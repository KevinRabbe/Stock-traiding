from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from stock_trading.engine import (
    EngineCycleResult,
    FileStrategyMetadataStore,
    JsonlEngineAuditObserver,
    StrategyRecord,
    StrategyRegistry,
    StrategyScorecard,
    StrategyStage,
)


class _Strategy:
    def __init__(self, strategy_id: str):
        self._strategy_id = strategy_id

    @property
    def strategy_id(self) -> str:
        return self._strategy_id

    def evaluate(self, candidates, portfolio):
        del candidates, portfolio
        return ()


def _record(strategy_id: str, stage: StrategyStage, score: float) -> StrategyRecord:
    return StrategyRecord(
        strategy_id=strategy_id,
        stage=stage,
        artifact_ref=f"models/{strategy_id}",
        selection_score=score,
        scorecard=StrategyScorecard(
            compounded_return=0.05 + score / 100.0,
            profit_factor=1.4,
            worst_realized_drawdown=0.02,
            total_trades=150,
            profitable_year_rate=0.6,
        ),
    )


def test_strategy_registry_survives_restart_without_dropping_unloaded_challengers(tmp_path) -> None:
    path = tmp_path / "strategy_registry.json"
    store = FileStrategyMetadataStore(path)
    registry = StrategyRegistry(store)
    registry.register(_Strategy("v5"), _record("v5", StrategyStage.PAPER, 1.0))
    registry.register(
        _Strategy("challenger"),
        _record("challenger", StrategyStage.SHADOW, 2.0),
    )
    registry.set_champion("v5")

    restarted = StrategyRegistry(FileStrategyMetadataStore(path))
    assert restarted.champion_id == "v5"
    assert [item.strategy_id for item in restarted.records()] == ["challenger", "v5"]
    with pytest.raises(RuntimeError, match="not loaded"):
        restarted.active()

    # Loading only the champion must not erase the unloaded challenger metadata.
    restarted.register(_Strategy("v5"))
    assert restarted.active().strategy_id == "v5"
    restarted.register(_Strategy("new"), _record("new", StrategyStage.DEVELOPMENT, 0.5))

    second_restart = StrategyRegistry(FileStrategyMetadataStore(path))
    assert [item.strategy_id for item in second_restart.records()] == [
        "challenger",
        "new",
        "v5",
    ]
    assert second_restart.champion_id == "v5"


def test_persisted_champion_requires_loaded_plugin_before_repromotion(tmp_path) -> None:
    path = tmp_path / "strategy_registry.json"
    registry = StrategyRegistry(FileStrategyMetadataStore(path))
    registry.register(_Strategy("v5"), _record("v5", StrategyStage.PAPER, 1.0))
    registry.set_champion("v5")

    restarted = StrategyRegistry(FileStrategyMetadataStore(path))
    with pytest.raises(RuntimeError, match="plugin must be loaded"):
        restarted.set_champion("v5")


def test_engine_audit_observer_appends_durable_jsonl(tmp_path) -> None:
    path = tmp_path / "audit" / "engine.jsonl"
    observer = JsonlEngineAuditObserver(path)
    result = EngineCycleResult(
        as_of=datetime(2025, 1, 2, 12, tzinfo=timezone.utc),
        strategy_id="v5",
        candidate_count=10,
        opportunity_count=3,
        eligible_opportunity_count=2,
        allocation_count=1,
        position_orders=(),
        entry_orders=(),
        executions=(),
    )

    observer.record(result)
    observer.record(result)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["schema_version"] == 1
    assert rows[0]["kind"] == "engine_cycle"
    assert rows[0]["payload"]["strategy_id"] == "v5"
    assert rows[0]["payload"]["as_of"] == "2025-01-02T12:00:00+00:00"
