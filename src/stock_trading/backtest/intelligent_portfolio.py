from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import sqrt

import numpy as np

from stock_trading.market import DuckDbMarketStore
from stock_trading.ml.lightgbm_models import OpportunityPrediction

from .portfolio import (
    BacktestConfig,
    BacktestResult,
    ScoredCandidate,
    TradeRecord,
    _build_result,
)


@dataclass(frozen=True, slots=True)
class PortfolioIntelligenceConfig:
    """Prediction-aware capital allocation with point-in-time diversification controls."""

    starting_capital: float = 10_000.0
    base_allocation_pct: float = 0.02
    min_allocation_pct: float = 0.0075
    max_allocation_pct: float = 0.03
    max_gross_exposure_pct: float = 0.35
    max_open_positions: int = 15
    signal_floor: float = 0.95
    max_expected_downside: float = 0.06
    correlation_lookback_sessions: int = 60
    min_correlation_observations: int = 30
    correlation_penalty_start: float = 0.50
    high_correlation_threshold: float = 0.75
    max_correlated_exposure_pct: float = 0.08
    round_trip_cost_bps: float = 20.0

    def __post_init__(self) -> None:
        if self.starting_capital <= 0:
            raise ValueError("starting_capital must be > 0")
        if not 0 < self.min_allocation_pct <= self.base_allocation_pct <= self.max_allocation_pct <= 1:
            raise ValueError("allocation percentages must satisfy 0 < min <= base <= max <= 1")
        if not self.max_allocation_pct <= self.max_gross_exposure_pct <= 1:
            raise ValueError("max_gross_exposure_pct must be >= max allocation and <= 1")
        if self.max_open_positions <= 0:
            raise ValueError("max_open_positions must be > 0")
        if not 0 <= self.signal_floor < 1:
            raise ValueError("signal_floor must be in [0, 1)")
        if self.max_expected_downside <= 0:
            raise ValueError("max_expected_downside must be > 0")
        if self.correlation_lookback_sessions < 2:
            raise ValueError("correlation_lookback_sessions must be >= 2")
        if self.min_correlation_observations < 2:
            raise ValueError("min_correlation_observations must be >= 2")
        if self.min_correlation_observations >= self.correlation_lookback_sessions:
            raise ValueError("min_correlation_observations must be below lookback")
        if not -1 <= self.correlation_penalty_start < 1:
            raise ValueError("correlation_penalty_start must be in [-1, 1)")
        if not self.correlation_penalty_start < self.high_correlation_threshold <= 1:
            raise ValueError("high_correlation_threshold must exceed penalty start and be <= 1")
        if not self.max_allocation_pct <= self.max_correlated_exposure_pct <= self.max_gross_exposure_pct:
            raise ValueError(
                "max_correlated_exposure_pct must be between max allocation and max gross exposure"
            )
        if self.round_trip_cost_bps < 0:
            raise ValueError("round_trip_cost_bps must be >= 0")


@dataclass(frozen=True, slots=True)
class PortfolioIntelligenceDiagnostics:
    rejected_duplicate_company: int
    rejected_slot_capacity: int
    rejected_gross_exposure: int
    rejected_correlation_exposure: int
    accepted_high_correlation: int
    entries_without_correlation_estimate: int
    average_allocation_pct: float | None
    minimum_allocation_pct: float | None
    maximum_allocation_pct: float | None
    average_entry_max_correlation: float | None
    maximum_entry_max_correlation: float | None
    max_gross_exposure_pct: float


@dataclass(slots=True)
class _OpenPosition:
    candidate: ScoredCandidate
    capital: float
    security_id: str


class IntelligentPortfolioBacktester:
    """One-position-per-company backtest with prediction/risk/correlation-aware sizing.

    Candidate selection remains upstream. This layer decides how much capital a
    qualified opportunity deserves given its score, predicted downside and the
    trailing PIT correlation of its security with positions already open.
    """

    def __init__(
        self,
        market_store: DuckDbMarketStore,
        config: PortfolioIntelligenceConfig | None = None,
    ) -> None:
        self.market_store = market_store
        self.config = config or PortfolioIntelligenceConfig()
        self._correlation_cache: dict[tuple[str, str, date], float | None] = {}

    def run(
        self,
        candidates: tuple[ScoredCandidate, ...] | list[ScoredCandidate],
    ) -> tuple[BacktestResult, PortfolioIntelligenceDiagnostics]:
        config = self.config
        candidates = tuple(candidates)
        self._correlation_cache.clear()
        by_entry: dict[date, list[ScoredCandidate]] = {}
        for candidate in candidates:
            by_entry.setdefault(candidate.row.execution_date, []).append(candidate)

        all_dates = sorted(
            set(by_entry)
            | {candidate.row.exit_date_20d for candidate in candidates}
        )
        cash = config.starting_capital
        open_positions: dict[str, _OpenPosition] = {}
        trades: list[TradeRecord] = []
        equity_peak = config.starting_capital
        realized_max_drawdown = 0.0

        rejected_duplicate = 0
        rejected_slot = 0
        rejected_gross = 0
        rejected_correlation = 0
        accepted_high_correlation = 0
        entries_without_correlation = 0
        allocation_pcts: list[float] = []
        entry_correlations: list[float] = []
        max_gross_observed = 0.0

        for current_date in all_dates:
            exiting = sorted(
                (
                    position
                    for position in open_positions.values()
                    if position.candidate.row.exit_date_20d <= current_date
                ),
                key=lambda position: (
                    position.candidate.row.exit_date_20d,
                    position.candidate.row.event_id,
                ),
            )
            for position in exiting:
                row = position.candidate.row
                gross_return = row.stock_return_20d
                net_return = gross_return - config.round_trip_cost_bps / 10_000.0
                pnl = position.capital * net_return
                cash += position.capital + pnl
                trades.append(
                    TradeRecord(
                        event_id=row.event_id,
                        company_id=row.company_id,
                        entry_date=row.execution_date,
                        exit_date=row.exit_date_20d,
                        allocated_capital=position.capital,
                        gross_return=gross_return,
                        net_return=net_return,
                        alpha_20d=row.alpha_20d,
                        max_adverse_excursion=-row.downside_20d,
                        pnl=pnl,
                        opportunity_score=position.candidate.prediction.opportunity_score,
                    )
                )
                del open_positions[row.company_id]

            equity = cash + sum(position.capital for position in open_positions.values())
            equity_peak = max(equity_peak, equity)
            if equity_peak > 0:
                realized_max_drawdown = max(
                    realized_max_drawdown,
                    (equity_peak - equity) / equity_peak,
                )

            daily_candidates = sorted(
                by_entry.get(current_date, ()),
                key=lambda candidate: (
                    -candidate.prediction.opportunity_score,
                    candidate.row.event_id,
                ),
            )
            for candidate in daily_candidates:
                row = candidate.row
                if row.company_id in open_positions:
                    rejected_duplicate += 1
                    continue
                if len(open_positions) >= config.max_open_positions:
                    rejected_slot += 1
                    continue

                security_id = self.market_store.security_for_company(
                    row.company_id,
                    row.execution_date,
                )
                if security_id is None:
                    raise RuntimeError(
                        f"selected company {row.company_id} has no security mapping on {row.execution_date}"
                    )

                correlations: list[tuple[float, _OpenPosition]] = []
                for position in open_positions.values():
                    correlation = self._correlation(
                        security_id,
                        position.security_id,
                        current_date,
                    )
                    if correlation is not None:
                        correlations.append((correlation, position))
                max_correlation = max(
                    (correlation for correlation, _ in correlations),
                    default=None,
                )
                if max_correlation is None and open_positions:
                    entries_without_correlation += 1
                elif max_correlation is not None:
                    entry_correlations.append(max_correlation)

                equity = cash + sum(position.capital for position in open_positions.values())
                if equity <= 0:
                    rejected_gross += 1
                    continue

                target_pct = dynamic_allocation_pct(
                    candidate.prediction,
                    config,
                    max_correlation=max_correlation,
                )
                target_capital = equity * target_pct
                min_capital = equity * config.min_allocation_pct

                gross_capital = sum(position.capital for position in open_positions.values())
                gross_headroom = max(0.0, equity * config.max_gross_exposure_pct - gross_capital)
                if gross_headroom < min_capital:
                    rejected_gross += 1
                    continue
                target_capital = min(target_capital, gross_headroom)

                correlated_capital = sum(
                    position.capital
                    for correlation, position in correlations
                    if correlation >= config.high_correlation_threshold
                )
                if max_correlation is not None and max_correlation >= config.high_correlation_threshold:
                    correlation_headroom = max(
                        0.0,
                        equity * config.max_correlated_exposure_pct - correlated_capital,
                    )
                    if correlation_headroom < min_capital:
                        rejected_correlation += 1
                        continue
                    target_capital = min(target_capital, correlation_headroom)
                    accepted_high_correlation += 1

                capital = min(cash, target_capital)
                if capital < min_capital:
                    rejected_gross += 1
                    continue

                allocation_pct = capital / equity
                allocation_pcts.append(allocation_pct)
                cash -= capital
                open_positions[row.company_id] = _OpenPosition(
                    candidate=candidate,
                    capital=capital,
                    security_id=security_id,
                )
                gross_after = (
                    sum(position.capital for position in open_positions.values()) / equity
                )
                max_gross_observed = max(max_gross_observed, gross_after)

        if open_positions:
            raise RuntimeError("intelligent backtest ended with positions that never reached their exit date")

        backtest_config = BacktestConfig(
            starting_capital=config.starting_capital,
            allocation_pct=config.base_allocation_pct,
            max_open_positions=config.max_open_positions,
            min_expected_alpha=-1_000_000.0,
            min_probability_positive=0.0,
            max_expected_downside=config.max_expected_downside,
            round_trip_cost_bps=config.round_trip_cost_bps,
        )
        result = _build_result(
            backtest_config,
            trades,
            ending_capital=cash,
            realized_max_drawdown=realized_max_drawdown,
            rejected_by_signal=0,
            rejected_duplicate=rejected_duplicate,
            rejected_capacity=rejected_slot + rejected_gross + rejected_correlation,
        )
        diagnostics = PortfolioIntelligenceDiagnostics(
            rejected_duplicate_company=rejected_duplicate,
            rejected_slot_capacity=rejected_slot,
            rejected_gross_exposure=rejected_gross,
            rejected_correlation_exposure=rejected_correlation,
            accepted_high_correlation=accepted_high_correlation,
            entries_without_correlation_estimate=entries_without_correlation,
            average_allocation_pct=_average(allocation_pcts),
            minimum_allocation_pct=min(allocation_pcts) if allocation_pcts else None,
            maximum_allocation_pct=max(allocation_pcts) if allocation_pcts else None,
            average_entry_max_correlation=_average(entry_correlations),
            maximum_entry_max_correlation=max(entry_correlations) if entry_correlations else None,
            max_gross_exposure_pct=max_gross_observed,
        )
        return result, diagnostics

    def _correlation(
        self,
        left_security_id: str,
        right_security_id: str,
        day: date,
    ) -> float | None:
        if left_security_id == right_security_id:
            return 1.0
        pair = tuple(sorted((left_security_id, right_security_id)))
        key = (pair[0], pair[1], day)
        if key not in self._correlation_cache:
            self._correlation_cache[key] = trailing_return_correlation(
                self.market_store,
                pair[0],
                pair[1],
                day,
                lookback_sessions=self.config.correlation_lookback_sessions,
                min_observations=self.config.min_correlation_observations,
            )
        return self._correlation_cache[key]


def dynamic_allocation_pct(
    prediction: OpportunityPrediction,
    config: PortfolioIntelligenceConfig,
    *,
    max_correlation: float | None,
) -> float:
    """Size from scale-free final rank, predicted downside and diversification."""

    score_strength = _clip(
        (prediction.opportunity_score - config.signal_floor) / (1.0 - config.signal_floor),
        0.0,
        1.0,
    )
    quality_multiplier = 0.75 + 0.75 * score_strength

    risk_ratio = _clip(
        prediction.expected_downside_20d / config.max_expected_downside,
        0.0,
        1.0,
    )
    # Convex-ish risk discount: very low predicted downside earns modestly more
    # capital, while a candidate near the hard risk ceiling is cut materially.
    risk_multiplier = 0.5 + 0.75 * sqrt(1.0 - risk_ratio)

    correlation_multiplier = 1.0
    if max_correlation is not None and max_correlation > config.correlation_penalty_start:
        span = 1.0 - config.correlation_penalty_start
        correlation_multiplier = 1.0 - 0.5 * _clip(
            (max_correlation - config.correlation_penalty_start) / span,
            0.0,
            1.0,
        )

    raw = (
        config.base_allocation_pct
        * quality_multiplier
        * risk_multiplier
        * correlation_multiplier
    )
    return _clip(raw, config.min_allocation_pct, config.max_allocation_pct)


def trailing_return_correlation(
    market_store: DuckDbMarketStore,
    left_security_id: str,
    right_security_id: str,
    day: date,
    *,
    lookback_sessions: int = 60,
    min_observations: int = 30,
) -> float | None:
    """Correlation of adjusted-close daily returns strictly before ``day``."""

    if left_security_id == right_security_id:
        return 1.0
    left = _return_map(
        market_store.bars_before(left_security_id, day, lookback_sessions + 1)
    )
    right = _return_map(
        market_store.bars_before(right_security_id, day, lookback_sessions + 1)
    )
    common = sorted(set(left) & set(right))
    if len(common) < min_observations:
        return None
    left_values = np.asarray([left[item] for item in common], dtype=np.float64)
    right_values = np.asarray([right[item] for item in common], dtype=np.float64)
    if float(left_values.std()) == 0.0 or float(right_values.std()) == 0.0:
        return None
    value = float(np.corrcoef(left_values, right_values)[0, 1])
    if not np.isfinite(value):
        return None
    return _clip(value, -1.0, 1.0)


def _return_map(bars) -> dict[date, float]:
    values: dict[date, float] = {}
    for previous, current in zip(bars, bars[1:]):
        previous_close = float(previous.adj_close)
        current_close = float(current.adj_close)
        if previous_close <= 0:
            continue
        values[current.date] = current_close / previous_close - 1.0
    return values


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))
