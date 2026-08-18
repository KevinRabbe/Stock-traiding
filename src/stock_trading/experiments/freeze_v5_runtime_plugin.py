from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from math import isclose
from pathlib import Path

from stock_trading.engine import (
    FileStrategyMetadataStore,
    FixedAllocationPortfolioPolicy,
    PassThroughOpportunityRiskPolicy,
    PassThroughPortfolioRiskPolicy,
    StrategyStage,
    build_strategy_artifact_manifest,
    load_strategy_artifact_manifest,
    verify_strategy_artifact_manifest,
    write_strategy_artifact_manifest,
)
from stock_trading.market import DuckDbMarketStore
from stock_trading.ml.multi_horizon import build_multi_horizon_targets
from stock_trading.ml.system_context import augment_system_context_features
from stock_trading.ml.walk_forward import annual_walk_forward_splits
from stock_trading.research import HistoricalStrategyBacktester
from stock_trading.strategies import V5StrategyConfig, build_v5_strategy_from_saved_models
from stock_trading.strategies.frozen_factory import (
    load_frozen_factory_strategy,
    load_frozen_factory_strategy_from_manifest,
    write_frozen_factory_strategy,
)

from .lightgbm_diagnostics import _load_training_rows
from .strategy_engine_v5_replay import _feature_snapshots, _historical_candidates


DEFAULT_STRATEGY_ID = "lightgbm-v5-adaptive-horizon"
REPLAY_SCHEMA = "strategy-engine-v5-exact-replay"


@dataclass(frozen=True, slots=True)
class V5RuntimeFreezeResult:
    strategy_id: str
    model_year: int
    artifact_root: Path
    manifest_path: Path
    verified_test_return: float
    verified_test_trade_count: int
    already_frozen: bool


def freeze_v5_runtime_plugin(
    experiment_dir: str | Path,
    *,
    runtime_dir: str | Path = "data/runtime",
    model_year: int = 2026,
    starting_capital: float = 10_000.0,
    market_read_cache_series: int = 160,
    strategy_id: str = DEFAULT_STRATEGY_ID,
) -> V5RuntimeFreezeResult:
    """Make the legacy V5 PAPER champion restart-restorable without retraining it.

    The migration reconstructs the exact saved-model V5 plugin for ``model_year``
    using the same legacy validation/calibration split as the locked exact replay.
    It serializes that initial calibration state, reloads the plugin from disk, and
    requires the reloaded artifact to reproduce the stored V5 test-year return and
    trade count before replacing the champion manifest.

    This deliberately preserves V5 as a legacy comparison champion. It does not
    claim the later G002m full-horizon maturity correction for V5 and it does not
    retrain any predictor.
    """

    if not strategy_id.strip():
        raise ValueError("strategy_id must not be empty")
    if model_year <= 0:
        raise ValueError("model_year must be positive")
    if starting_capital <= 0:
        raise ValueError("starting_capital must be > 0")
    if market_read_cache_series <= 0:
        raise ValueError("market_read_cache_series must be > 0")

    experiment_root = Path(experiment_dir)
    runtime_root = Path(runtime_dir)
    replay_path = experiment_root / "strategy_engine_v5_replay.json"
    models_root = experiment_root / "profit_models_v5"
    rows_path = experiment_root / "training_rows.jsonl"
    for required in (replay_path, models_root, rows_path):
        if not required.exists():
            raise FileNotFoundError(f"missing V5 freeze prerequisite: {required}")

    replay = _load_replay(replay_path, strategy_id=strategy_id, model_year=model_year)
    config_payload = dict(replay["strategy"])
    config_payload["horizons"] = tuple(int(item) for item in config_payload["horizons"])
    config = V5StrategyConfig(**config_payload)
    if config.strategy_id != strategy_id:
        raise ValueError("V5 replay strategy_id does not match requested champion")

    registry_path = runtime_root / "strategy_registry.json"
    metadata_store = FileStrategyMetadataStore(registry_path)
    snapshot = metadata_store.load()
    if snapshot is None or snapshot.champion_id is None:
        raise RuntimeError("runtime has no persisted champion")
    if snapshot.champion_id != strategy_id:
        raise RuntimeError(
            f"runtime champion is {snapshot.champion_id}; refusing V5 artifact migration"
        )
    records = {item.strategy_id: item for item in snapshot.records}
    record = records.get(strategy_id)
    if record is None:
        raise RuntimeError("persisted V5 champion record is missing")
    if record.stage not in (StrategyStage.PAPER, StrategyStage.LIVE):
        raise RuntimeError("persisted V5 champion must be PAPER or LIVE")
    if not record.artifact_ref:
        raise RuntimeError("persisted V5 champion has no artifact manifest")

    manifest_path = Path(record.artifact_ref)
    current_manifest = load_strategy_artifact_manifest(manifest_path)
    if current_manifest.strategy_id != strategy_id:
        raise ValueError("persisted V5 artifact manifest strategy_id mismatch")
    verify_strategy_artifact_manifest(current_manifest)

    artifact_root = runtime_root / "models" / strategy_id
    if artifact_root.is_dir() and (artifact_root / "strategy.json").is_file():
        if Path(current_manifest.root) != artifact_root:
            raise RuntimeError(
                "self-contained V5 artifact directory exists but registry manifest still "
                "points elsewhere; refusing ambiguous recovery"
            )
        strategy = load_frozen_factory_strategy_from_manifest(manifest_path)
        if strategy.strategy_id != strategy_id:
            raise RuntimeError("existing self-contained V5 artifact restored wrong strategy")
        verification = _verification_from_metadata(artifact_root / "strategy.json")
        return V5RuntimeFreezeResult(
            strategy_id=strategy_id,
            model_year=model_year,
            artifact_root=artifact_root,
            manifest_path=manifest_path,
            verified_test_return=verification["return"],
            verified_test_trade_count=verification["trades"],
            already_frozen=True,
        )
    if artifact_root.exists():
        raise RuntimeError(f"V5 runtime artifact directory already exists: {artifact_root}")

    market_store = DuckDbMarketStore(replay["market_db"])
    market_store.enable_read_cache(max_series=market_read_cache_series)
    source_rows = _load_training_rows(rows_path)
    targets = build_multi_horizon_targets(
        source_rows,
        market_store,
        benchmark_security_id=str(replay["benchmark_security_id"]),
        horizons=tuple(config.horizons),
        verify_existing_20d=True,
    )
    complete_rows = tuple(row for row in source_rows if row.event_id in targets)
    rows = augment_system_context_features(complete_rows)
    splits = annual_walk_forward_splits(rows)
    split = next((item for item in splits if item.test_year == model_year), None)
    if split is None:
        raise RuntimeError(f"V5 replay has no walk-forward split for {model_year}")

    validation_candidates = _feature_snapshots(split.validation_rows, market_store)
    strategy = build_v5_strategy_from_saved_models(
        models_root,
        model_year=model_year,
        validation_candidates=validation_candidates,
        config=config,
    )
    expected = _expected_year(replay, model_year)
    round_trip_cost_bps = config.min_expected_return * 10_000.0
    portfolio_policy = FixedAllocationPortfolioPolicy(**dict(replay["portfolio_policy"]))
    historical = _historical_candidates(split.test_rows, targets, market_store)
    backtester = HistoricalStrategyBacktester(
        starting_capital=starting_capital,
        round_trip_cost_bps=round_trip_cost_bps,
    )

    created_root = False
    pending_manifest = manifest_path.with_name(manifest_path.name + ".pending")
    try:
        write_frozen_factory_strategy(
            artifact_root,
            strategy_id=strategy_id,
            model_year=model_year,
            models=strategy.models,
            calibration=strategy.calibration,
            config=strategy.config,
            source={
                "kind": "legacy_v5_exact_replay",
                "experiment_dir": str(experiment_root),
                "replay_path": str(replay_path),
                "source_models_root": str(models_root),
                "predictor_retrained": False,
                "full_horizon_maturity_required": False,
                "legacy_maturity_method": "stored_20d_exit_date_boundary",
            },
            training={
                "saved_models_reused": True,
                "calibration_reconstructed_from_validation": True,
                "validation_year": model_year - 1,
                "model_year": model_year,
                "verification": {
                    "expected_test_return": float(expected["return"]),
                    "expected_test_trade_count": int(expected["trades"]),
                },
            },
        )
        created_root = True

        restored = load_frozen_factory_strategy(artifact_root)
        if restored.strategy_id != strategy_id:
            raise RuntimeError("serialized V5 artifact restored wrong strategy_id")
        result = backtester.run(
            strategy=restored,
            candidates=historical,
            opportunity_risk=PassThroughOpportunityRiskPolicy(),
            portfolio_policy=portfolio_policy,
            portfolio_risk=PassThroughPortfolioRiskPolicy(),
        )
        _require_year_identity(result.total_return, len(result.trades), expected, model_year)
        _write_verified_result(
            artifact_root / "strategy.json",
            verified_return=result.total_return,
            verified_trades=len(result.trades),
        )

        manifest = build_strategy_artifact_manifest(strategy_id, artifact_root)
        verify_strategy_artifact_manifest(manifest)
        write_strategy_artifact_manifest(manifest, pending_manifest)
        pending = load_strategy_artifact_manifest(pending_manifest)
        verify_strategy_artifact_manifest(pending)
        if pending.strategy_id != strategy_id or Path(pending.root) != artifact_root:
            raise RuntimeError("pending V5 artifact manifest identity mismatch")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(pending_manifest, manifest_path)
        created_root = False
        return V5RuntimeFreezeResult(
            strategy_id=strategy_id,
            model_year=model_year,
            artifact_root=artifact_root,
            manifest_path=manifest_path,
            verified_test_return=result.total_return,
            verified_test_trade_count=len(result.trades),
            already_frozen=False,
        )
    finally:
        pending_manifest.unlink(missing_ok=True)
        if created_root:
            shutil.rmtree(artifact_root, ignore_errors=True)


def _load_replay(path: Path, *, strategy_id: str, model_year: int) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid V5 exact replay: {path}") from exc
    if payload.get("schema_version") != REPLAY_SCHEMA:
        raise ValueError("unexpected V5 exact replay schema")
    architecture = payload.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("V5 exact replay is missing architecture metadata")
    for field in (
        "generic_strategy_plugin",
        "generic_historical_backtester",
        "saved_models_reused",
        "exact_v5_identity_verified",
    ):
        if architecture.get(field) is not True:
            raise ValueError(f"V5 exact replay does not verify {field}")
    if architecture.get("predictor_retrained") is not False:
        raise ValueError("V5 exact replay unexpectedly retrained predictor")
    strategy = payload.get("strategy")
    if not isinstance(strategy, dict) or strategy.get("strategy_id") != strategy_id:
        raise ValueError("V5 exact replay strategy metadata mismatch")
    if not payload.get("market_db") or not payload.get("benchmark_security_id"):
        raise ValueError("V5 exact replay is missing market metadata")
    _expected_year(payload, model_year)
    return payload


def _expected_year(replay: dict, model_year: int) -> dict:
    for item in replay.get("years", ()):  # exact replay has one row per test year
        if int(item.get("year", -1)) == model_year:
            if item.get("identity_verified_against_v5") is not True:
                raise ValueError(f"V5 replay year {model_year} is not identity-verified")
            return item
    raise ValueError(f"V5 exact replay does not contain model year {model_year}")


def _require_year_identity(
    observed_return: float,
    observed_trades: int,
    expected: dict,
    model_year: int,
) -> None:
    expected_trades = int(expected["trades"])
    expected_return = float(expected["return"])
    if observed_trades != expected_trades:
        raise RuntimeError(
            f"serialized V5 {model_year} trade count diverged: "
            f"{observed_trades} != {expected_trades}"
        )
    if not isclose(observed_return, expected_return, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"serialized V5 {model_year} return diverged: "
            f"{observed_return} != {expected_return}"
        )


def _write_verified_result(
    metadata_path: Path,
    *,
    verified_return: float,
    verified_trades: int,
) -> None:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    training = payload.setdefault("training", {})
    training["serialized_replay_verified"] = True
    training["verified_test_return"] = float(verified_return)
    training["verified_test_trade_count"] = int(verified_trades)
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _verification_from_metadata(metadata_path: Path) -> dict[str, float | int]:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        training = payload["training"]
        if training.get("serialized_replay_verified") is not True:
            raise ValueError("existing V5 artifact lacks serialized replay verification")
        return {
            "return": float(training["verified_test_return"]),
            "trades": int(training["verified_test_trade_count"]),
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid self-contained V5 metadata: {metadata_path}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the exact legacy V5 PAPER champion into a self-contained runtime "
            "artifact with serialized calibration and exact test-year replay proof."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, default=Path("data/runtime"))
    parser.add_argument("--model-year", type=int, default=2026)
    parser.add_argument("--starting-capital", type=float, default=10_000.0)
    parser.add_argument("--market-read-cache-series", type=int, default=160)
    parser.add_argument("--strategy-id", default=DEFAULT_STRATEGY_ID)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = freeze_v5_runtime_plugin(
        args.experiment_dir,
        runtime_dir=args.runtime_dir,
        model_year=args.model_year,
        starting_capital=args.starting_capital,
        market_read_cache_series=args.market_read_cache_series,
        strategy_id=args.strategy_id,
    )
    print(
        json.dumps(
            {
                "strategy_id": result.strategy_id,
                "model_year": result.model_year,
                "artifact_root": str(result.artifact_root),
                "manifest_path": str(result.manifest_path),
                "verified_test_return": result.verified_test_return,
                "verified_test_trade_count": result.verified_test_trade_count,
                "already_frozen": result.already_frozen,
                "predictor_retrained": False,
                "legacy_full_horizon_maturity_required": False,
                "champion_stage_changed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
