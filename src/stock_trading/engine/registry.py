from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

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

    def __post_init__(self) -> None:
        if not self.strategy_id.strip():
            raise ValueError("strategy_id must not be empty")


@dataclass(frozen=True, slots=True)
class StrategyRegistrySnapshot:
    champion_id: str | None
    records: tuple[StrategyRecord, ...]


class StrategyMetadataStore(Protocol):
    def load(self) -> StrategyRegistrySnapshot | None: ...

    def save(self, snapshot: StrategyRegistrySnapshot) -> None: ...


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
    """Champion/challenger registry with optional durable deployment metadata.

    Durable metadata is kept even when a strategy plugin is not loaded in the
    current process. The champion ID may therefore be known after restart while
    ``active()`` still refuses to run until that exact plugin is explicitly loaded.
    """

    def __init__(self, metadata_store: StrategyMetadataStore | None = None) -> None:
        self._strategies: dict[str, OpportunityStrategy] = {}
        self._metadata_store = metadata_store
        persisted = metadata_store.load() if metadata_store is not None else None
        self._records: dict[str, StrategyRecord] = (
            {record.strategy_id: record for record in persisted.records}
            if persisted is not None
            else {}
        )
        self._champion_id = persisted.champion_id if persisted is not None else None

    def register(
        self,
        strategy: OpportunityStrategy,
        record: StrategyRecord | None = None,
    ) -> None:
        persisted = self._records.get(strategy.strategy_id)
        resolved = persisted or record
        if resolved is None:
            raise ValueError(
                f"no metadata supplied or persisted for strategy {strategy.strategy_id}"
            )
        if strategy.strategy_id != resolved.strategy_id:
            raise ValueError("strategy_id mismatch between plugin and record")
        if (
            self._champion_id == resolved.strategy_id
            and resolved.stage is StrategyStage.RETIRED
        ):
            raise ValueError("persisted champion strategy is retired")
        self._strategies[resolved.strategy_id] = strategy
        self._records[resolved.strategy_id] = resolved
        if persisted is None:
            self._persist()

    def update_record(self, record: StrategyRecord) -> None:
        if record.strategy_id not in self._records:
            raise KeyError(f"unknown strategy {record.strategy_id}")
        if (
            self._champion_id == record.strategy_id
            and record.stage is StrategyStage.RETIRED
        ):
            raise ValueError("champion strategy cannot be retired before replacement")
        self._records[record.strategy_id] = record
        self._persist()

    def set_champion(self, strategy_id: str) -> None:
        record = self._records.get(strategy_id)
        if record is None:
            raise KeyError(f"unknown strategy {strategy_id}")
        if strategy_id not in self._strategies:
            raise RuntimeError("champion strategy plugin must be loaded before promotion")
        if record.stage is StrategyStage.RETIRED:
            raise ValueError("retired strategy cannot be champion")
        self._champion_id = strategy_id
        self._persist()

    @property
    def champion_id(self) -> str | None:
        return self._champion_id

    def active(self) -> OpportunityStrategy:
        if self._champion_id is None:
            raise RuntimeError("no champion strategy configured")
        return self.loaded_strategy(self._champion_id)

    def loaded_strategy(self, strategy_id: str) -> OpportunityStrategy:
        if strategy_id not in self._records:
            raise KeyError(f"unknown strategy {strategy_id}")
        try:
            return self._strategies[strategy_id]
        except KeyError as exc:
            raise RuntimeError(
                f"strategy plugin {strategy_id} is not loaded"
            ) from exc

    def loaded_challenger_strategies(
        self,
        *,
        stages: Iterable[StrategyStage] = (StrategyStage.SHADOW, StrategyStage.PAPER),
    ) -> tuple[OpportunityStrategy, ...]:
        allowed = set(stages)
        return tuple(
            self._strategies[strategy_id]
            for strategy_id in sorted(self._strategies)
            if strategy_id != self._champion_id
            and self._records[strategy_id].stage in allowed
        )

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
                float(record.scorecard.compounded_return)
                if record.scorecard
                else float("-inf"),
                record.strategy_id,
            ),
        )

    def snapshot(self) -> StrategyRegistrySnapshot:
        return StrategyRegistrySnapshot(
            champion_id=self._champion_id,
            records=self.records(),
        )

    def _persist(self) -> None:
        if self._metadata_store is not None:
            self._metadata_store.save(self.snapshot())
