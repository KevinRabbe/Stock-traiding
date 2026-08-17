from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from math import inf
from pathlib import Path
from typing import Mapping

from stock_trading.engine import FeatureValue
from stock_trading.engine.protocols import (
    OpportunityRiskPolicy,
    OpportunityStrategy,
    PortfolioPolicy,
    PortfolioRiskPolicy,
)

from .historical import (
    HistoricalBacktestResult,
    HistoricalCandidate,
    HistoricalTrade,
    HistoricalStrategyBacktester,
    _OpenPosition,
    _portfolio_snapshot,
    _validate_allocations,
    _validate_opportunities,
    _validate_opportunity_subset,
    _validate_portfolio_risk,
)


@dataclass(frozen=True, slots=True)
class HistoricalExecutionLiquidity:
    """Execution-day liquidity kept outside the strategy's PIT feature snapshot."""

    candidate_id: str
    entry_price: float
    entry_volume: float

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if self.entry_price <= 0:
            raise ValueError("entry_price must be > 0")
        if self.entry_volume < 0:
            raise ValueError("entry_volume must be >= 0")

    @property
    def dollar_volume(self) -> float:
        return self.entry_price * self.entry_volume


@dataclass(frozen=True, slots=True)
class MarketQualityExclusion:
    security_id: str
    ticker: str
    start_date: date
    end_date: date
    reason: str

    def __post_init__(self) -> None:
        if not self.security_id or not self.ticker or not self.reason:
            raise ValueError("market quality exclusion fields must not be empty")
        if self.end_date < self.start_date:
            raise ValueError("market quality exclusion end_date precedes start_date")


@dataclass(frozen=True, slots=True)
class ExecutionRealisticBacktestDiagnostics:
    rejected_entry_liquidity: int


def load_market_quality_exclusions(path: str | Path) -> tuple[MarketQualityExclusion, ...]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "market-quality-verified-v1":
        raise ValueError("unsupported market quality manifest schema")

    result: list[MarketQualityExclusion] = []
    for item in payload.get("exclusions") or ():
        result.append(
            MarketQualityExclusion(
                security_id=str(item["security_id"]),
                ticker=str(item["ticker"]),
                start_date=date.fromisoformat(str(item["start_date"])),
                end_date=date.fromisoformat(str(item["end_date"])),
                reason=str(item["reason"]),
            )
        )
    return tuple(result)


def target_overlaps_exclusion(
    security_id: str,
    entry_date: date,
    exit_date: date,
    exclusions: tuple[MarketQualityExclusion, ...],
) -> bool:
    if exit_date < entry_date:
        raise ValueError("exit_date precedes entry_date")
    return any(
        item.security_id == security_id
        and entry_date <= item.end_date
        and exit_date >= item.start_date
        for item in exclusions
    )


def trailing_adv_supports(
    features: Mapping[str, FeatureValue],
    *,
    required_capital: float,
    max_participation_pct: float,
) -> bool:
    """Return whether trailing PIT average dollar volume can support a full fill."""

    if required_capital <= 0:
        raise ValueError("required_capital must be > 0")
    if not 0.0 < max_participation_pct <= 1.0:
        raise ValueError("max_participation_pct must be in (0, 1]")

    value = features.get("market.avg_dollar_volume_20d")
    if value is None or isinstance(value, bool):
        return False
    try:
        adv = float(value)
    except (TypeError, ValueError):
        return False
    if adv <= 0:
        return False
    return required_capital <= adv * max_participation_pct + 1e-12


class ExecutionRealisticHistoricalBacktester(HistoricalStrategyBacktester):
    """Historical runner that requires the intended entry to fit real day liquidity.

    The execution-day volume mapping is deliberately supplied outside
    ``HistoricalCandidate``. Strategies therefore cannot observe realized entry-day
    liquidity when they rank opportunities. The adapter only answers whether the
    already-decided order could plausibly receive a full fill at the modeled price.
    """

    def run(
        self,
        *,
        strategy: OpportunityStrategy,
        candidates: tuple[HistoricalCandidate, ...],
        opportunity_risk: OpportunityRiskPolicy,
        portfolio_policy: PortfolioPolicy,
        portfolio_risk: PortfolioRiskPolicy,
        entry_liquidity: Mapping[str, HistoricalExecutionLiquidity],
        max_entry_day_participation_pct: float = 0.01,
    ) -> tuple[HistoricalBacktestResult, ExecutionRealisticBacktestDiagnostics]:
        if not 0.0 < max_entry_day_participation_pct <= 1.0:
            raise ValueError("max_entry_day_participation_pct must be in (0, 1]")

        candidates_by_date: dict[date, list[HistoricalCandidate]] = {}
        seen_candidates: set[str] = set()
        all_dates: set[date] = set()
        for candidate in candidates:
            candidate_id = candidate.snapshot.candidate_id
            if candidate_id in seen_candidates:
                raise ValueError(f"duplicate historical candidate {candidate_id}")
            seen_candidates.add(candidate_id)
            if candidate_id not in entry_liquidity:
                raise ValueError(f"missing entry liquidity for {candidate_id}")
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
        rejected_entry_liquidity = 0

        for current_date in sorted(all_dates):
            cash = self._close_due(current_date, cash, open_positions, trades)
            equity = cash + sum(item.capital for item in open_positions.values())
            equity_peak = max(equity_peak, equity)
            if equity_peak > 0:
                max_drawdown = max(max_drawdown, (equity_peak - equity) / equity_peak)

            batch = tuple(candidates_by_date.get(current_date, ()))
            if not batch:
                continue

            portfolio = _portfolio_snapshot(current_date, equity, cash, open_positions)
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

                target_capital = equity * allocation.allocation_pct
                liquidity = entry_liquidity[opportunity.candidate_id]
                capacity = liquidity.dollar_volume * max_entry_day_participation_pct
                if target_capital > capacity + 1e-12:
                    rejected_entry_liquidity += 1
                    continue

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
        result = HistoricalBacktestResult(
            starting_capital=self.starting_capital,
            ending_capital=cash,
            total_return=cash / self.starting_capital - 1.0,
            profit_factor=profit_factor,
            realized_max_drawdown=max_drawdown,
            trades=tuple(trades),
            rejected_cash=rejected_cash,
        )
        return result, ExecutionRealisticBacktestDiagnostics(
            rejected_entry_liquidity=rejected_entry_liquidity,
        )
