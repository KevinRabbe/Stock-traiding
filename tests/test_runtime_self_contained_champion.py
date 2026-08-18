from __future__ import annotations

from dataclasses import dataclass
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
    runtime_verification_payload,
    verify_paper_shadow_runtime,
)


@dataclass
class _Strategy:
    strategy_id: str

    def evaluate(self, candidates, portfolio):
        del candidates, portfolio
        return ()


def _artifact(
    runtime_dir: Path,
    strategy_id: str,
    *,
    self_contained: bool,
) -> Path:
    root = runtime_dir / "models" / strategy_id
    root.mkdir(parents=True)
    (root / "payload.txt").write_text(strategy_id, encoding="utf-8")
    if self_contained:
        (root / "strategy.json").write_text("{}\n", encoding="utf-8")
    manifest = build_strategy_artifact_manifest(strategy_id, root)
    path = runtime_dir / "artifacts" / f"{strategy_id}.json"
    write_strategy_artifact_manifest(manifest, path)
    return path


def _runtime(runtime_dir: Path, *, self_contained_champion: bool) -> None:
    champion_path = _artifact(
        runtime_dir,
        "champion",
        self_contained=self_contained_champion,
    )
    shadow_path = _artifact(runtime_dir, "shadow", self_contained=True)
    FileStrategyMetadataStore(runtime_dir / "strategy_registry.json").save(
        StrategyRegistrySnapshot(
            champion_id="champion",
            records=(
                StrategyRecord(
                    strategy_id="champion",
                    stage=StrategyStage.PAPER,
                    artifact_ref=str(champion_path),
                ),
                StrategyRecord(
                    strategy_id="shadow",
                    stage=StrategyStage.SHADOW,
                    artifact_ref=str(shadow_path),
                ),
            ),
        )
    )


def _restore(path) -> _Strategy:
    return _Strategy(Path(path).stem)


def test_self_contained_champion_is_verified_and_restored_without_injection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _runtime(tmp_path, self_contained_champion=True)
    monkeypatch.setattr(
        runtime_state,
        "load_frozen_factory_strategy_from_manifest",
        _restore,
    )

    verification = verify_paper_shadow_runtime(tmp_path)
    payload = runtime_verification_payload(verification)
    loaded = load_persisted_shadow_registry(runtime_dir=tmp_path)

    assert verification.champion_plugin_restored is True
    assert payload["champion_plugin_restore"] == "autonomous_from_verified_manifest"
    assert loaded.registry.active().strategy_id == "champion"
    assert loaded.shadow_strategy_ids == ("shadow",)


def test_legacy_champion_still_requires_explicit_injection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _runtime(tmp_path, self_contained_champion=False)
    monkeypatch.setattr(
        runtime_state,
        "load_frozen_factory_strategy_from_manifest",
        _restore,
    )

    verification = verify_paper_shadow_runtime(tmp_path)
    assert verification.champion_plugin_restored is False

    with pytest.raises(RuntimeError, match="not self-contained"):
        load_persisted_shadow_registry(runtime_dir=tmp_path)

    loaded = load_persisted_shadow_registry(
        _Strategy("champion"),
        runtime_dir=tmp_path,
    )
    assert loaded.registry.active().strategy_id == "champion"
