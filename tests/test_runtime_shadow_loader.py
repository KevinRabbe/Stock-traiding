from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from stock_trading.engine import (
    FileStrategyMetadataStore,
    StrategyRecord,
    StrategyRegistrySnapshot,
    StrategyStage,
    build_strategy_artifact_manifest,
    write_strategy_artifact_manifest,
)
from stock_trading.live import runtime_state
from stock_trading.live.runtime_state import (
    load_persisted_shadow_registry,
    verify_paper_shadow_runtime,
)
from stock_trading.live.service import ShadowStrategyResult
from stock_trading.live.shadow_persistence import JsonlShadowAuditObserver


@dataclass
class _Strategy:
    strategy_id: str

    def evaluate(self, candidates, portfolio):
        del candidates, portfolio
        return ()


def _artifact(runtime_dir: Path, strategy_id: str) -> Path:
    root = runtime_dir / "models" / strategy_id
    root.mkdir(parents=True)
    (root / "payload.txt").write_text(strategy_id, encoding="utf-8")
    manifest = build_strategy_artifact_manifest(strategy_id, root)
    path = runtime_dir / "artifacts" / f"{strategy_id}.json"
    write_strategy_artifact_manifest(manifest, path)
    return path


def _runtime(runtime_dir: Path) -> dict[str, Path]:
    champion = "champion"
    shadow_b = "shadow-b"
    shadow_a = "shadow-a"
    paths = {
        champion: _artifact(runtime_dir, champion),
        shadow_b: _artifact(runtime_dir, shadow_b),
        shadow_a: _artifact(runtime_dir, shadow_a),
    }
    FileStrategyMetadataStore(runtime_dir / "strategy_registry.json").save(
        StrategyRegistrySnapshot(
            champion_id=champion,
            records=(
                StrategyRecord(
                    strategy_id=shadow_b,
                    stage=StrategyStage.SHADOW,
                    artifact_ref=str(paths[shadow_b]),
                ),
                StrategyRecord(
                    strategy_id=champion,
                    stage=StrategyStage.PAPER,
                    artifact_ref=str(paths[champion]),
                ),
                StrategyRecord(
                    strategy_id=shadow_a,
                    stage=StrategyStage.SHADOW,
                    artifact_ref=str(paths[shadow_a]),
                ),
            ),
        )
    )
    return paths


def _restore(path) -> _Strategy:
    return _Strategy(Path(path).stem)


def test_runtime_verification_restores_all_shadows_in_deterministic_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _runtime(tmp_path)
    monkeypatch.setattr(
        runtime_state,
        "load_frozen_factory_strategy_from_manifest",
        _restore,
    )

    result = verify_paper_shadow_runtime(tmp_path)

    assert result.champion_id == "champion"
    assert result.shadow_strategy_ids == ("shadow-a", "shadow-b")
    by_id = {item.strategy_id: item for item in result.artifacts}
    assert by_id["champion"].manifest_verified is True
    assert by_id["champion"].plugin_restored is False
    assert by_id["shadow-a"].plugin_restored is True
    assert by_id["shadow-b"].plugin_restored is True


def test_runtime_registry_requires_exact_injected_champion_and_loads_shadows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _runtime(tmp_path)
    monkeypatch.setattr(
        runtime_state,
        "load_frozen_factory_strategy_from_manifest",
        _restore,
    )

    loaded = load_persisted_shadow_registry(
        _Strategy("champion"),
        runtime_dir=tmp_path,
    )

    assert loaded.champion_id == "champion"
    assert loaded.registry.active().strategy_id == "champion"
    assert loaded.shadow_strategy_ids == ("shadow-a", "shadow-b")

    with pytest.raises(ValueError, match="does not match persisted champion"):
        load_persisted_shadow_registry(
            _Strategy("foreign"),
            runtime_dir=tmp_path,
        )


def test_runtime_verification_fails_closed_when_artifact_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _runtime(tmp_path)
    monkeypatch.setattr(
        runtime_state,
        "load_frozen_factory_strategy_from_manifest",
        _restore,
    )
    (tmp_path / "models" / "shadow-a" / "payload.txt").write_text(
        "tampered",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strategy artifact file changed"):
        verify_paper_shadow_runtime(tmp_path)


def test_shadow_audit_observer_appends_complete_cycle_evidence(tmp_path: Path) -> None:
    path = tmp_path / "shadow_evaluations.jsonl"
    observer = JsonlShadowAuditObserver(path)
    result = ShadowStrategyResult(
        strategy_id="shadow-a",
        candidate_count=10,
        opportunity_count=3,
        eligible_opportunity_count=2,
        allocation_count=1,
        requested_exposure_pct=0.02,
        top_score=0.97,
        horizon_counts=((20, 1),),
        selected_candidate_ids=("candidate-1",),
    )
    as_of = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)

    observer.record(as_of, (result,))
    observer.record(as_of, ())

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["kind"] == "shadow_cycle"
    assert rows[0]["as_of"] == as_of.isoformat()
    assert rows[0]["results"][0]["strategy_id"] == "shadow-a"
    assert rows[0]["results"][0]["selected_candidate_ids"] == ["candidate-1"]
    assert rows[1]["results"] == []
