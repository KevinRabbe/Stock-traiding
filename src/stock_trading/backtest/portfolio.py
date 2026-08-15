from dataclasses import dataclass
from datetime import date

from stock_trading.ml.dataset import TrainingRow
from stock_trading.ml.lightgbm_models import LightGbmModelBundle, OpportunityPrediction


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    starting_capital: float = 10_000.0
    allocation_pct: float = 0.02
    max_open_positions: int = 15
    min_expected_alpha: float = 0.03
    min_probability_positive: float = 0.60
    max_expected_downside: float = 0.06
    round_trip_cost_bps: float = 20.0

    def __post_init__(self) -> None:
        if self.starting_capital <= 0:
            raise ValueError("starting_capital must be > 0")
        if not 0 < self.allocation_pct <= 1:
            raise ValueError("allocation_pct must be in (0, 1]")
        if self.max_open_positions <= 0:
            raise ValueError("max_open_positions must be > 0")
        if not 0 <= self.min_probability_positive <= 1:
            raise ValueError("min_probability_positive must be in [0, 1]")
        if self.max_expected_downside < 0:
            raise ValueError("max_expected_downside must be >= 0")
        if self.round_trip_cost_bps < 0:
            raise ValueError("round_trip_cost_bps must be >= 0")


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    row: TrainingRow
    prediction: OpportunityPrediction


@dataclass(frozen=True, slots=True)
class TradeRecord:
    event_id: str
    company_id: str
    entry_date: date
    exit_date: date
    allocated_capital: float
    gross_return: float
    net_return: float
    alpha_20d: float
    max_adverse_excursion: float
    pnl: float
    opportunity_score: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    starting_capital: float
    ending_capital: float
    net_profit: float
    total_return: float
    trades: tuple[TradeRecord, ...]
    win_rate: float | None
    profit_factor: float | None
    average_trade_return: float | None
    realized_max_drawdown: float
    worst_trade_mae: float | None
    rejected_by_signal: int
    rejected_duplicate_company: int
    rejected_capacity: int


@dataclass(slots=True)
class _OpenPosition:
    candidate: ScoredCandidate
    capital: float


class FixedAllocationBacktester:
    """Long-only event backtest with intentionally simple capital rules."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def score_rows(
        self,
        rows: tuple[TrainingRow, ...] | list[TrainingRow],
        model: LightGbmModelBundle,
    ) -> tuple[ScoredCandidate, ...]:
        return tuple(
            ScoredCandidate(row=row, prediction=model.predict(row.features))
            for row in rows
        )

    def run(self, candidates: tuple[ScoredCandidate, ...] | list[ScoredCandidate]) -> BacktestResult:
        config = self.config
        candidates = tuple(candidates)
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
        rejected_by_signal = 0
        rejected_duplicate = 0
        rejected_capacity = 0

        for current_date in all_dates:
            # Capital from completed positions becomes available before new entries.
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
                candidate = position.candidate
                row = candidate.row
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
                        opportunity_score=candidate.prediction.opportunity_score,
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
                if not self._passes_signal(candidate.prediction):
                    rejected_by_signal += 1
                    continue
                if candidate.row.company_id in open_positions:
                    rejected_duplicate += 1
                    continue
                if len(open_positions) >= config.max_open_positions:
                    rejected_capacity += 1
                    continue

                equity = cash + sum(position.capital for position in open_positions.values())
                target_capital = equity * config.allocation_pct
                capital = min(cash, target_capital)
                if capital <= 0:
                    rejected_capacity += 1
                    continue

                cash -= capital
                open_positions[candidate.row.company_id] = _OpenPosition(
                    candidate=candidate,
                    capital=capital,
                )

        if open_positions:
            raise RuntimeError("backtest ended with positions that never reached their exit date")

        return _build_result(
            config,
            trades,
            ending_capital=cash,
            realized_max_drawdown=realized_max_drawdown,
            rejected_by_signal=rejected_by_signal,
            rejected_duplicate=rejected_duplicate,
            rejected_capacity=rejected_capacity,
        )

    def _passes_signal(self, prediction: OpportunityPrediction) -> bool:
        return _passes_signal(self.config, prediction)


class FixedAllocationTrancheBacktester:
    """Fixed allocation backtest that can add bounded overlapping tranches per company.

    The total portfolio slot cap is unchanged. Repeated high-scoring opportunities
    for one company can therefore consume another normal allocation slice instead
    of being discarded, up to ``max_company_tranches``. Each tranche keeps its own
    entry and 20-day exit date.
    """

    def __init__(
        self,
        config: BacktestConfig | None = None,
        *,
        max_company_tranches: int = 2,
    ) -> None:
        if max_company_tranches <= 0:
            raise ValueError("max_company_tranches must be > 0")
        self.config = config or BacktestConfig()
        self.max_company_tranches = max_company_tranches

    def run(self, candidates: tuple[ScoredCandidate, ...] | list[ScoredCandidate]) -> BacktestResult:
        config = self.config
        candidates = tuple(candidates)
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
        rejected_by_signal = 0
        rejected_company_limit = 0
        rejected_capacity = 0

        for current_date in all_dates:
            exiting_keys = sorted(
                (
                    key
                    for key, position in open_positions.items()
                    if position.candidate.row.exit_date_20d <= current_date
                ),
                key=lambda key: (
                    open_positions[key].candidate.row.exit_date_20d,
                    open_positions[key].candidate.row.event_id,
                ),
            )
            for key in exiting_keys:
                position = open_positions.pop(key)
                candidate = position.candidate
                row = candidate.row
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
                        opportunity_score=candidate.prediction.opportunity_score,
                    )
                )

            equity = cash + sum(position.capital for position in open_positions.values())
            equity_peak = max(equity_peak, equity)
            if equity_peak > 0:
                realized_max_drawdown = max(
                    realized_max_drawdown,
                    (equity_peak - equity) / equity_peak,
                )

            company_open_counts: dict[str, int] = {}
            for position in open_positions.values():
                company_id = position.candidate.row.company_id
                company_open_counts[company_id] = company_open_counts.get(company_id, 0) + 1

            daily_candidates = sorted(
                by_entry.get(current_date, ()),
                key=lambda candidate: (
                    -candidate.prediction.opportunity_score,
                    candidate.row.event_id,
                ),
            )
            for candidate in daily_candidates:
                if not _passes_signal(config, candidate.prediction):
                    rejected_by_signal += 1
                    continue
                company_id = candidate.row.company_id
                if company_open_counts.get(company_id, 0) >= self.max_company_tranches:
                    rejected_company_limit += 1
                    continue
                if len(open_positions) >= config.max_open_positions:
                    rejected_capacity += 1
                    continue

                equity = cash + sum(position.capital for position in open_positions.values())
                target_capital = equity * config.allocation_pct
                capital = min(cash, target_capital)
                if capital <= 0:
                    rejected_capacity += 1
                    continue

                cash -= capital
                key = _position_key(candidate, len(open_positions))
                open_positions[key] = _OpenPosition(candidate=candidate, capital=capital)
                company_open_counts[company_id] = company_open_counts.get(company_id, 0) + 1

        if open_positions:
            raise RuntimeError("backtest ended with tranches that never reached their exit date")

        return _build_result(
            config,
            trades,
            ending_capital=cash,
            realized_max_drawdown=realized_max_drawdown,
            rejected_by_signal=rejected_by_signal,
            rejected_duplicate=rejected_company_limit,
            rejected_capacity=rejected_capacity,
        )


def _position_key(candidate: ScoredCandidate, salt: int) -> str:
    row = candidate.row
    return f"{row.company_id}:{row.event_id}:{row.execution_date.isoformat()}:{salt}"


def _passes_signal(config: BacktestConfig, prediction: OpportunityPrediction) -> bool:
    return (
        prediction.expected_alpha_20d >= config.min_expected_alpha
        and prediction.probability_positive_alpha >= config.min_probability_positive
        and prediction.expected_downside_20d <= config.max_expected_downside
    )


def _build_result(
    config: BacktestConfig,
    trades: list[TradeRecord],
    *,
    ending_capital: float,
    realized_max_drawdown: float,
    rejected_by_signal: int,
    rejected_duplicate: int,
    rejected_capacity: int,
) -> BacktestResult:
    returns = [trade.net_return for trade in trades]
    wins = [trade for trade in trades if trade.pnl > 0]
    gains = sum(trade.pnl for trade in trades if trade.pnl > 0)
    losses = -sum(trade.pnl for trade in trades if trade.pnl < 0)
    return BacktestResult(
        starting_capital=config.starting_capital,
        ending_capital=ending_capital,
        net_profit=ending_capital - config.starting_capital,
        total_return=ending_capital / config.starting_capital - 1.0,
        trades=tuple(trades),
        win_rate=(len(wins) / len(trades) if trades else None),
        profit_factor=(gains / losses if losses > 0 else (float("inf") if gains > 0 else None)),
        average_trade_return=(sum(returns) / len(returns) if returns else None),
        realized_max_drawdown=realized_max_drawdown,
        worst_trade_mae=(
            min(trade.max_adverse_excursion for trade in trades)
            if trades
            else None
        ),
        rejected_by_signal=rejected_by_signal,
        rejected_duplicate_company=rejected_duplicate,
        rejected_capacity=rejected_capacity,
    )
