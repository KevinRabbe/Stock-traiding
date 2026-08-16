from __future__ import annotations

from dataclasses import dataclass

from .contracts import StrategyStage
from .protocols import OpportunityStrategy


@dataclass(frozen=True, slots=True)
class StrategyScorecard:
    compounded_return: float
    profit_factor: float
    worst_realized_drawdown: float
    total_trades: int
    profitable_year_rate: float
    average_trade_alpha: float | None = None

    def __post_init__(self) -> None:
        if self.profit_factor < 0:
            raise ValueError("profit_factor must be >= 0")
        if self.worst_realized_drawdown < 0:
            raise ValueError("worst_realized_drawdown must be >= 0")
        if self.total_trades < 0:
            raise ValueError("total_trades must be >= 0")
        if not 0.0 <= self.profitable_year_rate <= 1.0:
            raise ValueError("profitable_year_rate must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class StrategyRecord:
    strategy_id: str
    stage: StrategyStage
    artifact_ref: str | None = None
    scorecard: StrategyScorecard | None = None
    selection_score: float | None = None
    notes: str = ""


@dataclass(frozen=True, slots=True)
class ProfitabilityGate:
    """Configurable eligibility gate. Ranking remains an explicit research choice."""

    min_compounded_return: float = 0.0
    min_profit_factor: float = 1.0
    min_trades: int = 50
    max_realized_drawdown: float = 0.10
    min_profitable_year_rate: float = 0.0

    def eligible(self, record: StrategyRecord) -> bool:
        scorecard = record.scorecard
        if scorecard is None or record.stage is StrategyStage.RETIRED:
            return False
        return (
            scorecard.compounded_return > self.min_compounded_return
            and scorecard.profit_factor >= self.min_profit_factor
            and scorecard.total_trades >= self.min_trades
            and scorecard.worst_realized_drawdown <= self.max_realized_drawdown
            and scorecard.profitable_year_rate >= self.min_profitable_year_rate
        )


class StrategyRegistry:
    """In-process champion/challenger registry for strategy plugins.

    The registry can recommend an eligible challenger from an externally computed
    ``selection_score``, but promotion is always explicit through ``set_champion``.
    Backtest results therefore cannot silently deploy a live strategy.
    """

    def __init__(self) -> None:
        self._strategies: dict[str, OpportunityStrategy] = {}
        self._records: dict[str, StrategyRecord] = {}
        self._champion_id: str | None = None

    def register(self, strategy: OpportunityStrategy, record: StrategyRecord) -> None:
        if strategy.strategy_id != record.strategy_id:
            raise ValueError("strategy_id mismatch between plugin and record")
        self._strategies[record.strategy_id] = strategy
        self._records[record.strategy_id] = record

    def update_record(self, record: StrategyRecord) -> None:
        if record.strategy_id not in self._strategies:
            raise KeyError(f"unknown strategy {record.strategy_id}")
        self._records[record.strategy_id] = record

    def set_champion(self, strategy_id: str) -> None:
        record = self._records.get(strategy_id)
        if record is None:
            raise KeyError(f"unknown strategy {strategy_id}")
        if record.stage is StrategyStage.RETIRED:
            raise ValueError("retired strategy cannot be champion")
        self._champion_id = strategy_id

    def active(self) -> OpportunityStrategy:
        if self._champion_id is None:
            raise RuntimeError("no champion strategy configured")
        return self._strategies[self._champion_id]

    def record(self, strategy_id: str) -> StrategyRecord:
        try:
            return self._records[strategy_id]
        except KeyError as exc:
            raise KeyError(f"unknown strategy {strategy_id}") from exc

    def records(self) -> tuple[StrategyRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def challengers(self) -> tuple[StrategyRecord, ...]:
        return tuple(
            record
            for record in self.records()
            if record.strategy_id != self._champion_id
            and record.stage is not StrategyStage.RETIRED
        )

    def recommend_champion(self, gate: ProfitabilityGate) -> StrategyRecord | None:
        eligible = [
            record
            for record in self._records.values()
            if gate.eligible(record) and record.selection_score is not None
        ]
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda record: (
                float(record.selection_score),
                float(record.scorecard.compounded_return) if record.scorecard else float("-inf"),
                record.strategy_id,
            ),
        )
