from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from math import inf
from typing import Mapping

from stock_trading.engine import (
    AllocationIntent,
    FeatureSnapshot,
    Opportunity,
    PortfolioPosition,
    PortfolioSnapshot,
)
from stock_trading.engine.protocols import (
    OpportunityRiskPolicy,
    OpportunityStrategy,
    PortfolioPolicy,
    PortfolioRiskPolicy,
)


@dataclass(frozen=True, slots=True)
class HistoricalOutcome:
    horizon_sessions: int
    exit_date: date
    stock_return: float
    alpha: float
    downside: float

    def __post_init__(self) -> None:
        if self.horizon_sessions <= 0:
            raise ValueError("horizon_sessions must be > 0")
        if self.downside < 0:
            raise ValueError("downside must be >= 0")


@dataclass(frozen=True, slots=True)
class HistoricalCandidate:
    snapshot: FeatureSnapshot
    outcomes: Mapping[int, HistoricalOutcome]

    def __post_init__(self) -> None:
        if not self.outcomes:
            raise ValueError("historical candidate must have at least one outcome")
        for horizon, outcome in self.outcomes.items():
            if horizon != outcome.horizon_sessions:
                raise ValueError("historical outcome key/horizon mismatch")
            if outcome.exit_date < self.snapshot.execution_date:
                raise ValueError("historical outcome exits before execution")


@dataclass(frozen=True, slots=True)
class HistoricalTrade:
    strategy_id: str
    candidate_id: str
    event_id: str
    company_id: str
    security_id: str
    entry_date: date
    exit_date: date
    horizon_sessions: int
    allocation_pct: float
    allocated_capital: float
    gross_return: float
    net_return: float
    alpha: float
    downside: float
    pnl: float
    opportunity_score: float


@dataclass(frozen=True, slots=True)
class HistoricalBacktestResult:
    starting_capital: float
    ending_capital: float
    total_return: float
    profit_factor: float
    realized_max_drawdown: float
    trades: tuple[HistoricalTrade, ...]
    rejected_cash: int


@dataclass(slots=True)
class _OpenPosition:
    opportunity: Opportunity
    candidate: HistoricalCandidate
    outcome: HistoricalOutcome
    capital: float
    allocation_pct: float


class HistoricalStrategyBacktester:
    """Run a production strategy against outcomes hidden outside the strategy.

    The strategy sees only PIT ``FeatureSnapshot`` values and current portfolio
    state. Realized outcomes live in this research adapter and are selected only
    after the strategy has chosen a holding horizon.
    """

    def __init__(
        self,
        *,
        starting_capital: float = 10_000.0,
        round_trip_cost_bps: float = 20.0,
    ) -> None:
        if starting_capital <= 0:
            raise ValueError("starting_capital must be > 0")
        if round_trip_cost_bps < 0:
            raise ValueError("round_trip_cost_bps must be >= 0")
        self.starting_capital = starting_capital
        self.round_trip_cost_bps = round_trip_cost_bps

    def run(
        self,
        *,
        strategy: OpportunityStrategy,
        candidates: tuple[HistoricalCandidate, ...],
        opportunity_risk: OpportunityRiskPolicy,
        portfolio_policy: PortfolioPolicy,
        portfolio_risk: PortfolioRiskPolicy,
    ) -> HistoricalBacktestResult:
        candidates_by_date: dict[date, list[HistoricalCandidate]] = {}
        seen_candidates: set[str] = set()
        all_dates: set[date] = set()
        for candidate in candidates:
            candidate_id = candidate.snapshot.candidate_id
            if candidate_id in seen_candidates:
                raise ValueError(f"duplicate historical candidate {candidate_id}")
            seen_candidates.add(candidate_id)
            entry_date = candidate.snapshot.execution_date
            candidates_by_date.setdefault(entry_date, []).append(candidate)
            all_dates.add(entry_date)
            all_dates.update(outcome.exit_date for outcome in candidate.outcomes.values())

        cash = self.starting_capital
        open_positions: dict[str, _OpenPosition] = {}
        trades: list[HistoricalTrade] = []
        equity_peak = self.starting_capital
        max_drawdown = 0.0
        rejected_cash = 0

        for current_date in sorted(all_dates):
            cash = self._close_due(
                current_date,
                cash,
                open_positions,
                trades,
            )
            # Match the legacy backtester: drawdown is observed once after every
            # position due on this date has settled, before new entries.
            equity = cash + sum(item.capital for item in open_positions.values())
            equity_peak = max(equity_peak, equity)
            if equity_peak > 0:
                max_drawdown = max(max_drawdown, (equity_peak - equity) / equity_peak)

            batch = tuple(candidates_by_date.get(current_date, ()))
            if not batch:
                continue

            portfolio = _portfolio_snapshot(
                current_date,
                equity,
                cash,
                open_positions,
            )
            snapshots = tuple(item.snapshot for item in batch)
            by_id = {item.snapshot.candidate_id: item for item in batch}

            opportunities = strategy.evaluate(snapshots, portfolio)
            _validate_opportunities(strategy.strategy_id, opportunities, by_id)
            eligible = opportunity_risk.filter(opportunities, portfolio)
            _validate_opportunity_subset(opportunities, eligible)
            proposed_allocations = portfolio_policy.allocate(eligible, portfolio)
            _validate_allocations(proposed_allocations, eligible)
            allocations = portfolio_risk.filter(proposed_allocations, portfolio)
            _validate_portfolio_risk(proposed_allocations, allocations)

            for allocation in allocations:
                opportunity = allocation.opportunity
                historical = by_id.get(opportunity.candidate_id)
                if historical is None:
                    raise ValueError("portfolio allocation references non-current candidate")
                if opportunity.company_id in open_positions:
                    raise ValueError("portfolio policy opened duplicate active company")
                outcome = historical.outcomes.get(opportunity.horizon_sessions)
                if outcome is None:
                    raise ValueError(
                        f"missing realized {opportunity.horizon_sessions}d outcome for "
                        f"{opportunity.candidate_id}"
                    )

                # Preserve the legacy capital rule: target a fraction of current
                # equity and, if necessary, accept the available cash as a partial
                # allocation rather than fabricating leverage.
                target_capital = equity * allocation.allocation_pct
                capital = min(cash, target_capital)
                if capital <= 0:
                    rejected_cash += 1
                    continue
                realized_allocation_pct = capital / equity if equity > 0 else 0.0
                cash -= capital
                open_positions[opportunity.company_id] = _OpenPosition(
                    opportunity=opportunity,
                    candidate=historical,
                    outcome=outcome,
                    capital=capital,
                    allocation_pct=realized_allocation_pct,
                )

        if open_positions:
            raise RuntimeError("historical backtest ended with unclosed positions")

        positive_pnl = sum(trade.pnl for trade in trades if trade.pnl > 0)
        negative_pnl = -sum(trade.pnl for trade in trades if trade.pnl < 0)
        profit_factor = (
            positive_pnl / negative_pnl
            if negative_pnl > 0
            else (inf if positive_pnl > 0 else 0.0)
        )
        return HistoricalBacktestResult(
            starting_capital=self.starting_capital,
            ending_capital=cash,
            total_return=cash / self.starting_capital - 1.0,
            profit_factor=profit_factor,
            realized_max_drawdown=max_drawdown,
            trades=tuple(trades),
            rejected_cash=rejected_cash,
        )

    def _close_due(
        self,
        current_date: date,
        cash: float,
        open_positions: dict[str, _OpenPosition],
        trades: list[HistoricalTrade],
    ) -> float:
        exiting = sorted(
            (
                item
                for item in open_positions.values()
                if item.outcome.exit_date <= current_date
            ),
            key=lambda item: (item.outcome.exit_date, item.opportunity.event_id),
        )
        for item in exiting:
            cost = self.round_trip_cost_bps / 10_000.0
            net_return = item.outcome.stock_return - cost
            pnl = item.capital * net_return
            cash += item.capital + pnl
            opportunity = item.opportunity
            trades.append(
                HistoricalTrade(
                    strategy_id=opportunity.strategy_id,
                    candidate_id=opportunity.candidate_id,
                    event_id=opportunity.event_id,
                    company_id=opportunity.company_id,
                    security_id=opportunity.security_id,
                    entry_date=opportunity.execution_date,
                    exit_date=item.outcome.exit_date,
                    horizon_sessions=opportunity.horizon_sessions,
                    allocation_pct=item.allocation_pct,
                    allocated_capital=item.capital,
                    gross_return=item.outcome.stock_return,
                    net_return=net_return,
                    alpha=item.outcome.alpha,
                    downside=item.outcome.downside,
                    pnl=pnl,
                    opportunity_score=opportunity.score,
                )
            )
            del open_positions[opportunity.company_id]
        return cash


def _portfolio_snapshot(
    current_date: date,
    equity: float,
    cash: float,
    open_positions: Mapping[str, _OpenPosition],
) -> PortfolioSnapshot:
    gross_capital = sum(item.capital for item in open_positions.values())
    gross_exposure = gross_capital / equity if equity > 0 else 0.0
    return PortfolioSnapshot(
        as_of=datetime.combine(current_date, datetime.min.time(), tzinfo=timezone.utc),
        equity=equity,
        cash=cash,
        gross_exposure_pct=gross_exposure,
        positions=tuple(
            PortfolioPosition(
                position_id=f"hist:{item.opportunity.strategy_id}:{item.opportunity.candidate_id}",
                strategy_id=item.opportunity.strategy_id,
                company_id=item.opportunity.company_id,
                security_id=item.opportunity.security_id,
                allocation_pct=item.allocation_pct,
                opened_on=item.opportunity.execution_date,
                planned_exit_date=item.outcome.exit_date,
            )
            for item in sorted(
                open_positions.values(),
                key=lambda value: value.opportunity.candidate_id,
            )
        ),
    )


def _validate_opportunities(
    strategy_id: str,
    opportunities: tuple[Opportunity, ...],
    by_id: Mapping[str, HistoricalCandidate],
) -> None:
    seen: set[str] = set()
    for opportunity in opportunities:
        if opportunity.strategy_id != strategy_id:
            raise ValueError("strategy emitted foreign strategy_id")
        if opportunity.candidate_id in seen:
            raise ValueError("strategy emitted duplicate opportunity")
        seen.add(opportunity.candidate_id)
        historical = by_id.get(opportunity.candidate_id)
        if historical is None:
            raise ValueError("strategy emitted candidate outside current PIT batch")
        snapshot = historical.snapshot
        if (
            opportunity.event_id != snapshot.event_id
            or opportunity.company_id != snapshot.company_id
            or opportunity.security_id != snapshot.security_id
            or opportunity.execution_date != snapshot.execution_date
        ):
            raise ValueError("strategy mutated candidate identity")


def _validate_opportunity_subset(
    source: tuple[Opportunity, ...],
    subset: tuple[Opportunity, ...],
) -> None:
    allowed = {item.candidate_id: item for item in source}
    seen: set[str] = set()
    for opportunity in subset:
        if opportunity.candidate_id in seen:
            raise ValueError("opportunity risk duplicated candidate")
        seen.add(opportunity.candidate_id)
        if allowed.get(opportunity.candidate_id) != opportunity:
            raise ValueError("opportunity risk introduced or mutated opportunity")


def _validate_allocations(
    allocations: tuple[AllocationIntent, ...],
    opportunities: tuple[Opportunity, ...],
) -> None:
    allowed = {item.candidate_id: item for item in opportunities}
    seen: set[str] = set()
    for allocation in allocations:
        candidate_id = allocation.opportunity.candidate_id
        if candidate_id in seen:
            raise ValueError("portfolio policy duplicated candidate")
        seen.add(candidate_id)
        if allowed.get(candidate_id) != allocation.opportunity:
            raise ValueError("portfolio policy introduced or mutated opportunity")


def _validate_portfolio_risk(
    proposed: tuple[AllocationIntent, ...],
    filtered: tuple[AllocationIntent, ...],
) -> None:
    allowed = {item.opportunity.candidate_id: item for item in proposed}
    seen: set[str] = set()
    for allocation in filtered:
        candidate_id = allocation.opportunity.candidate_id
        if candidate_id in seen:
            raise ValueError("portfolio risk duplicated candidate")
        seen.add(candidate_id)
        original = allowed.get(candidate_id)
        if original is None or original.opportunity != allocation.opportunity:
            raise ValueError("portfolio risk introduced or mutated opportunity")
        if allocation.allocation_pct > original.allocation_pct + 1e-15:
            raise ValueError("portfolio risk may not upsize allocation")
