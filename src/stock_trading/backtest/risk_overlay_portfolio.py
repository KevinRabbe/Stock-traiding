from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from stock_trading.market import DuckDbMarketStore

from .intelligent_portfolio import trailing_return_correlation
from .portfolio import BacktestConfig, BacktestResult, ScoredCandidate, TradeRecord, _build_result


@dataclass(frozen=True, slots=True)
class RiskOverlayConfig:
    """A one-way risk overlay: qualified V5 trades may only be downsized, never upsized."""

    starting_capital: float = 10_000.0
    base_allocation_pct: float = 0.02
    min_allocation_pct: float = 0.01
    max_gross_exposure_pct: float = 0.20
    max_open_positions: int = 15
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
        if not 0 < self.min_allocation_pct <= self.base_allocation_pct <= 1:
            raise ValueError("allocation percentages must satisfy 0 < min <= base <= 1")
        if not self.base_allocation_pct <= self.max_gross_exposure_pct <= 1:
            raise ValueError("max_gross_exposure_pct must be >= base allocation and <= 1")
        if self.max_open_positions <= 0:
            raise ValueError("max_open_positions must be > 0")
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
        if not self.base_allocation_pct <= self.max_correlated_exposure_pct <= self.max_gross_exposure_pct:
            raise ValueError(
                "max_correlated_exposure_pct must be between base allocation and max gross exposure"
            )
        if self.round_trip_cost_bps < 0:
            raise ValueError("round_trip_cost_bps must be >= 0")


@dataclass(frozen=True, slots=True)
class RiskOverlayDiagnostics:
    rejected_duplicate_company: int
    rejected_slot_capacity: int
    rejected_gross_exposure: int
    rejected_correlation_exposure: int
    entries_without_correlation_estimate: int
    downsized_by_downside: int
    downsized_by_regime: int
    downsized_by_volatility: int
    downsized_by_correlation: int
    average_allocation_pct: float | None
    minimum_allocation_pct: float | None
    maximum_allocation_pct: float | None
    average_risk_multiplier: float | None
    average_entry_max_correlation: float | None
    maximum_entry_max_correlation: float | None
    max_gross_exposure_pct: float


@dataclass(slots=True)
class _OpenPosition:
    candidate: ScoredCandidate
    capital: float
    security_id: str


@dataclass(frozen=True, slots=True)
class _AllocationDecision:
    allocation_pct: float
    multiplier: float
    downside_multiplier: float
    regime_multiplier: float
    volatility_multiplier: float
    correlation_multiplier: float


class RiskOverlayBacktester:
    """Fixed-allocation V5 portfolio with a PIT one-way risk throttle.

    The base 2% policy remains the maximum. Predicted downside, benchmark regime,
    volatility expansion and correlation with already-open positions can only
    reduce capital. This avoids V6's failure mode where confidence-based upsizing
    amplified weak years.
    """

    def __init__(
        self,
        market_store: DuckDbMarketStore,
        config: RiskOverlayConfig | None = None,
    ) -> None:
        self.market_store = market_store
        self.config = config or RiskOverlayConfig()
        self._correlation_cache: dict[tuple[str, str, date], float | None] = {}

    def run(
        self,
        candidates: tuple[ScoredCandidate, ...] | list[ScoredCandidate],
    ) -> tuple[BacktestResult, RiskOverlayDiagnostics]:
        config = self.config
        candidates = tuple(candidates)
        self._correlation_cache.clear()
        by_entry: dict[date, list[ScoredCandidate]] = {}
        for candidate in candidates:
            by_entry.setdefault(candidate.row.execution_date, []).append(candidate)

        all_dates = sorted(
            set(by_entry) | {candidate.row.exit_date_20d for candidate in candidates}
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
        entries_without_correlation = 0
        downsized_downside = 0
        downsized_regime = 0
        downsized_volatility = 0
        downsized_correlation = 0
        allocation_pcts: list[float] = []
        risk_multipliers: list[float] = []
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

                equity = cash + sum(position.capital for position in open_positions.values())
                if equity <= 0:
                    rejected_gross += 1
                    continue

                decision = risk_overlay_allocation_pct(
                    candidate,
                    config,
                    max_correlation=max_correlation,
                )
                min_capital = equity * config.min_allocation_pct
                target_capital = equity * decision.allocation_pct

                gross_capital = sum(position.capital for position in open_positions.values())
                gross_headroom = max(
                    0.0,
                    equity * config.max_gross_exposure_pct - gross_capital,
                )
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

                capital = min(cash, target_capital)
                if capital < min_capital:
                    rejected_gross += 1
                    continue

                allocation_pct = capital / equity
                allocation_pcts.append(allocation_pct)
                risk_multipliers.append(allocation_pct / config.base_allocation_pct)
                if decision.downside_multiplier < 0.999999:
                    downsized_downside += 1
                if decision.regime_multiplier < 0.999999:
                    downsized_regime += 1
                if decision.volatility_multiplier < 0.999999:
                    downsized_volatility += 1
                if decision.correlation_multiplier < 0.999999:
                    downsized_correlation += 1
                if max_correlation is None and open_positions:
                    entries_without_correlation += 1
                elif max_correlation is not None:
                    entry_correlations.append(max_correlation)

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
            raise RuntimeError("risk-overlay backtest ended with positions that never reached their exit date")

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
        diagnostics = RiskOverlayDiagnostics(
            rejected_duplicate_company=rejected_duplicate,
            rejected_slot_capacity=rejected_slot,
            rejected_gross_exposure=rejected_gross,
            rejected_correlation_exposure=rejected_correlation,
            entries_without_correlation_estimate=entries_without_correlation,
            downsized_by_downside=downsized_downside,
            downsized_by_regime=downsized_regime,
            downsized_by_volatility=downsized_volatility,
            downsized_by_correlation=downsized_correlation,
            average_allocation_pct=_average(allocation_pcts),
            minimum_allocation_pct=min(allocation_pcts) if allocation_pcts else None,
            maximum_allocation_pct=max(allocation_pcts) if allocation_pcts else None,
            average_risk_multiplier=_average(risk_multipliers),
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


def risk_overlay_allocation_pct(
    candidate: ScoredCandidate,
    config: RiskOverlayConfig,
    *,
    max_correlation: float | None,
) -> _AllocationDecision:
    """Return a one-way PIT risk throttle in [min_allocation, base_allocation]."""

    prediction = candidate.prediction
    features = candidate.row.features

    downside_ratio = _clip(
        prediction.expected_downside_20d / config.max_expected_downside,
        0.0,
        1.0,
    )
    downside_multiplier = 1.0 - 0.40 * downside_ratio

    breadth = _number(features, "system.regime.benchmark_trend_breadth")
    regime_multiplier = 1.0
    if breadth is not None:
        regime_multiplier = 0.75 + 0.25 * _clip(breadth, 0.0, 1.0)
    benchmark_120d = _number(features, "market.benchmark_return_120d")
    if benchmark_120d is not None and benchmark_120d < 0:
        regime_multiplier = min(regime_multiplier, 0.80)

    volatility_ratio = _number(features, "system.volatility.ratio_20_60")
    volatility_multiplier = 1.0
    if volatility_ratio is not None and volatility_ratio > 1.0:
        volatility_multiplier = max(0.75, 1.0 / volatility_ratio)

    correlation_multiplier = 1.0
    if max_correlation is not None and max_correlation > config.correlation_penalty_start:
        span = 1.0 - config.correlation_penalty_start
        correlation_multiplier = 1.0 - 0.30 * _clip(
            (max_correlation - config.correlation_penalty_start) / span,
            0.0,
            1.0,
        )

    multiplier = min(
        1.0,
        downside_multiplier,
        regime_multiplier,
        volatility_multiplier,
        correlation_multiplier,
    )
    floor_multiplier = config.min_allocation_pct / config.base_allocation_pct
    multiplier = max(floor_multiplier, multiplier)
    allocation_pct = config.base_allocation_pct * multiplier
    return _AllocationDecision(
        allocation_pct=allocation_pct,
        multiplier=multiplier,
        downside_multiplier=downside_multiplier,
        regime_multiplier=regime_multiplier,
        volatility_multiplier=volatility_multiplier,
        correlation_multiplier=correlation_multiplier,
    )


def _number(features: dict[str, float | None], name: str) -> float | None:
    value = features.get(name)
    return float(value) if value is not None else None


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))
