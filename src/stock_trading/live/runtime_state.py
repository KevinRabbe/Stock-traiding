from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stock_trading.engine import (
    FileStrategyMetadataStore,
    StrategyRecord,
    StrategyRegistry,
    StrategyStage,
    load_strategy_artifact_manifest,
    verify_strategy_artifact_manifest,
)
from stock_trading.engine.protocols import OpportunityStrategy
from stock_trading.strategies.frozen_factory import (
    load_frozen_factory_strategy_from_manifest,
)


@dataclass(frozen=True, slots=True)
class RuntimeArtifactStatus:
    strategy_id: str
    stage: StrategyStage
    champion: bool
    artifact_ref: str
    manifest_verified: bool
    plugin_restored: bool


@dataclass(frozen=True, slots=True)
class RuntimeVerification:
    runtime_dir: Path
    registry_path: Path
    champion_id: str
    artifacts: tuple[RuntimeArtifactStatus, ...]

    @property
    def shadow_strategy_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                item.strategy_id
                for item in self.artifacts
                if item.stage is StrategyStage.SHADOW and item.plugin_restored
            )
        )


@dataclass(frozen=True, slots=True)
class LoadedRuntimeRegistry:
    registry: StrategyRegistry
    champion_id: str
    shadow_strategy_ids: tuple[str, ...]


def verify_paper_shadow_runtime(
    runtime_dir: str | Path = "data/runtime",
) -> RuntimeVerification:
    """Verify persisted strategy metadata and every active artifact.

    Frozen SHADOW plugins are also deserialized from their verified manifests.
    The legacy V5 PAPER champion is intentionally manifest-verified only because
    its calibration state is not yet serialized in the bootstrap artifact. Runtime
    execution must inject the already-constructed champion plugin explicitly.
    """

    runtime_root = Path(runtime_dir)
    registry_path = runtime_root / "strategy_registry.json"
    snapshot = FileStrategyMetadataStore(registry_path).load()
    if snapshot is None or snapshot.champion_id is None:
        raise RuntimeError("runtime has no persisted champion")

    records = {item.strategy_id: item for item in snapshot.records}
    champion_id = snapshot.champion_id
    champion = records.get(champion_id)
    if champion is None:
        raise ValueError("persisted champion is missing from strategy records")
    if champion.stage not in (StrategyStage.PAPER, StrategyStage.LIVE):
        raise ValueError("persisted champion must be PAPER or LIVE")

    statuses: list[RuntimeArtifactStatus] = []
    for record in snapshot.records:
        if record.stage is StrategyStage.RETIRED:
            continue
        manifest_path = _artifact_path(record)
        manifest = load_strategy_artifact_manifest(manifest_path)
        if manifest.strategy_id != record.strategy_id:
            raise ValueError(
                f"artifact manifest strategy_id mismatch for {record.strategy_id}"
            )
        verify_strategy_artifact_manifest(manifest)

        restored = False
        if record.strategy_id == champion_id:
            restored = False
        elif record.stage is StrategyStage.SHADOW:
            strategy = load_frozen_factory_strategy_from_manifest(manifest_path)
            if strategy.strategy_id != record.strategy_id:
                raise ValueError(
                    f"restored strategy_id mismatch for {record.strategy_id}"
                )
            restored = True
        else:
            raise RuntimeError(
                f"non-champion {record.stage.value} strategy {record.strategy_id} "
                "has no configured runtime plugin loader"
            )

        statuses.append(
            RuntimeArtifactStatus(
                strategy_id=record.strategy_id,
                stage=record.stage,
                champion=record.strategy_id == champion_id,
                artifact_ref=str(manifest_path),
                manifest_verified=True,
                plugin_restored=restored,
            )
        )

    return RuntimeVerification(
        runtime_dir=runtime_root,
        registry_path=registry_path,
        champion_id=champion_id,
        artifacts=tuple(statuses),
    )


def load_persisted_shadow_registry(
    champion_strategy: OpportunityStrategy,
    *,
    runtime_dir: str | Path = "data/runtime",
) -> LoadedRuntimeRegistry:
    """Restore the persisted champion/challenger registry for a service process.

    The V5 champion is supplied by the caller because bootstrap currently freezes
    only its model directory, not the rolling calibration state needed to restore
    the plugin autonomously. Every SHADOW challenger is restored directly from its
    immutable manifest. No metadata or lifecycle state is modified here.
    """

    verification = verify_paper_shadow_runtime(runtime_dir)
    if champion_strategy.strategy_id != verification.champion_id:
        raise ValueError(
            "injected champion strategy_id does not match persisted champion"
        )

    metadata_store = FileStrategyMetadataStore(verification.registry_path)
    registry = StrategyRegistry(metadata_store=metadata_store)
    registry.register(champion_strategy)

    snapshot = metadata_store.load()
    if snapshot is None:
        raise RuntimeError("strategy registry disappeared during runtime load")
    for record in snapshot.records:
        if record.strategy_id == verification.champion_id:
            continue
        if record.stage is StrategyStage.RETIRED:
            continue
        if record.stage is not StrategyStage.SHADOW:
            raise RuntimeError(
                f"unsupported non-champion runtime stage {record.stage.value} "
                f"for {record.strategy_id}"
            )
        strategy = load_frozen_factory_strategy_from_manifest(_artifact_path(record))
        registry.register(strategy)

    if registry.active().strategy_id != verification.champion_id:
        raise RuntimeError("loaded runtime champion identity changed")
    loaded_shadows = tuple(
        strategy.strategy_id
        for strategy in registry.loaded_challenger_strategies(
            stages=(StrategyStage.SHADOW,)
        )
    )
    if loaded_shadows != verification.shadow_strategy_ids:
        raise RuntimeError("loaded SHADOW strategy set differs from verified runtime")
    return LoadedRuntimeRegistry(
        registry=registry,
        champion_id=verification.champion_id,
        shadow_strategy_ids=loaded_shadows,
    )


def runtime_verification_payload(result: RuntimeVerification) -> dict[str, Any]:
    return {
        "runtime_dir": str(result.runtime_dir),
        "registry_path": str(result.registry_path),
        "champion_id": result.champion_id,
        "champion_plugin_restore": (
            "injected_required_until_v5_calibration_is_serialized"
        ),
        "shadow_strategy_ids": list(result.shadow_strategy_ids),
        "artifacts": [
            {
                "strategy_id": item.strategy_id,
                "stage": item.stage.value,
                "champion": item.champion,
                "artifact_ref": item.artifact_ref,
                "manifest_verified": item.manifest_verified,
                "plugin_restored": item.plugin_restored,
            }
            for item in result.artifacts
        ],
    }


def _artifact_path(record: StrategyRecord) -> Path:
    if not record.artifact_ref:
        raise ValueError(
            f"active strategy {record.strategy_id} has no artifact_ref"
        )
    return Path(record.artifact_ref)
