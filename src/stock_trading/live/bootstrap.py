from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import isclose
from pathlib import Path

from stock_trading.engine import (
    FileStrategyMetadataStore,
    StrategyRecord,
    StrategyRegistrySnapshot,
    StrategyScorecard,
    StrategyStage,
    build_strategy_artifact_manifest,
    verify_strategy_artifact_manifest,
    write_strategy_artifact_manifest,
)
from stock_trading.execution import FilePaperLedger, PaperLedgerState


DEFAULT_STRATEGY_ID = "lightgbm-v5-adaptive-horizon"


@dataclass(frozen=True, slots=True)
class PaperRuntimeConfig:
    schema_version: int
    strategy_id: str
    experiment_dir: str
    market_db: str
    benchmark_security_id: str
    strategy_registry: str
    artifact_manifest: str
    paper_ledger: str


@dataclass(frozen=True, slots=True)
class PaperChampionBootstrapResult:
    strategy_id: str
    runtime_dir: Path
    registry_path: Path
    artifact_manifest_path: Path
    paper_ledger_path: Path
    runtime_config_path: Path


def bootstrap_v5_paper_champion(
    experiment_dir: str | Path,
    *,
    runtime_dir: str | Path = "data/runtime",
    starting_capital: float = 10_000.0,
    strategy_id: str = DEFAULT_STRATEGY_ID,
) -> PaperChampionBootstrapResult:
    """Initialize V5 as PAPER champion only after exact architecture replay passes."""

    if starting_capital <= 0:
        raise ValueError("starting_capital must be > 0")
    if not strategy_id.strip():
        raise ValueError("strategy_id must not be empty")

    experiment_root = Path(experiment_dir)
    replay_path = experiment_root / "strategy_engine_v5_replay.json"
    models_root = experiment_root / "profit_models_v5"
    if not replay_path.exists():
        raise FileNotFoundError(
            f"missing exact V5 architecture replay: {replay_path}; run "
            "stock_trading.experiments.strategy_engine_v5_replay first"
        )
    if not models_root.is_dir():
        raise FileNotFoundError(f"missing saved V5 model directory: {models_root}")

    replay = _load_verified_replay(replay_path)
    observed = replay["observed"]
    runtime_root = Path(runtime_dir)
    runtime_root.mkdir(parents=True, exist_ok=True)
    artifacts_dir = runtime_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_strategy_artifact_manifest(strategy_id, models_root)
    manifest_path = artifacts_dir / f"{strategy_id}.json"
    write_strategy_artifact_manifest(manifest, manifest_path)
    verify_strategy_artifact_manifest(manifest)

    record = StrategyRecord(
        strategy_id=strategy_id,
        stage=StrategyStage.PAPER,
        artifact_ref=str(manifest_path),
        scorecard=StrategyScorecard(
            compounded_return=float(observed["compounded_return"]),
            profit_factor=float(observed["aggregate_profit_factor"]),
            worst_realized_drawdown=float(observed["worst_realized_drawdown"]),
            total_trades=int(observed["total_trades"]),
            profitable_year_rate=float(observed["profitable_year_rate"]),
            average_trade_alpha=(
                float(observed["average_trade_alpha"])
                if observed.get("average_trade_alpha") is not None
                else None
            ),
        ),
        notes="Bootstrapped from exact generic-engine V5 replay",
    )

    registry_path = runtime_root / "strategy_registry.json"
    metadata_store = FileStrategyMetadataStore(registry_path)
    existing = metadata_store.load()
    records = {item.strategy_id: item for item in existing.records} if existing else {}
    champion_id = existing.champion_id if existing else None
    if champion_id is not None and champion_id != strategy_id:
        raise RuntimeError(
            f"runtime already has champion {champion_id}; bootstrap refuses to replace it"
        )
    existing_record = records.get(strategy_id)
    if existing_record is not None:
        if (
            existing_record.stage is not StrategyStage.PAPER
            or existing_record.artifact_ref != record.artifact_ref
            or existing_record.scorecard != record.scorecard
        ):
            raise RuntimeError(
                f"existing {strategy_id} registry metadata differs; bootstrap refuses overwrite"
            )
    else:
        records[strategy_id] = record
    metadata_store.save(
        StrategyRegistrySnapshot(
            champion_id=strategy_id,
            records=tuple(records[key] for key in sorted(records)),
        )
    )

    ledger_path = runtime_root / "paper_ledger.json"
    ledger = FilePaperLedger(ledger_path, starting_cash=starting_capital)
    if ledger_path.exists():
        ledger.load()  # validate existing state; never reset paper state
    else:
        ledger.save(PaperLedgerState(cash=starting_capital))

    config = PaperRuntimeConfig(
        schema_version=1,
        strategy_id=strategy_id,
        experiment_dir=str(experiment_root),
        market_db=str(replay["market_db"]),
        benchmark_security_id=str(replay["benchmark_security_id"]),
        strategy_registry=str(registry_path),
        artifact_manifest=str(manifest_path),
        paper_ledger=str(ledger_path),
    )
    config_path = runtime_root / "paper_runtime.json"
    _write_json(config_path, asdict(config))

    return PaperChampionBootstrapResult(
        strategy_id=strategy_id,
        runtime_dir=runtime_root,
        registry_path=registry_path,
        artifact_manifest_path=manifest_path,
        paper_ledger_path=ledger_path,
        runtime_config_path=config_path,
    )


def _load_verified_replay(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid V5 architecture replay: {path}") from exc
    try:
        if payload["schema_version"] != "strategy-engine-v5-exact-replay":
            raise ValueError("unexpected V5 replay schema")
        architecture = payload["architecture"]
        if architecture.get("exact_v5_identity_verified") is not True:
            raise ValueError("V5 architecture replay did not verify exact identity")
        if architecture.get("generic_strategy_plugin") is not True:
            raise ValueError("V5 replay did not use generic strategy plugin")
        if architecture.get("generic_historical_backtester") is not True:
            raise ValueError("V5 replay did not use generic historical backtester")
        observed = payload["observed"]
        baseline = payload["v5_baseline"]
        if int(observed["total_trades"]) != int(baseline["total_trades"]):
            raise ValueError("V5 replay/baseline mismatch for total_trades")
        for field in (
            "compounded_return",
            "profitable_year_rate",
            "average_trade_alpha",
            "aggregate_profit_factor",
            "worst_realized_drawdown",
        ):
            left = observed.get(field)
            right = baseline.get(field)
            if left is None or right is None:
                if left != right:
                    raise ValueError(f"V5 replay/baseline mismatch for {field}")
                continue
            if not isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"V5 replay/baseline mismatch for {field}")
        if not payload.get("market_db") or not payload.get("benchmark_security_id"):
            raise ValueError("V5 replay is missing runtime market metadata")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid V5 architecture replay: {path}") from exc
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize the durable V5 PAPER champion only after the generic engine "
            "has reproduced the saved V5 backtest exactly."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, default=Path("data/runtime"))
    parser.add_argument("--starting-capital", type=float, default=10_000.0)
    parser.add_argument("--strategy-id", default=DEFAULT_STRATEGY_ID)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = bootstrap_v5_paper_champion(
        args.experiment_dir,
        runtime_dir=args.runtime_dir,
        starting_capital=args.starting_capital,
        strategy_id=args.strategy_id,
    )
    print(
        json.dumps(
            {
                "strategy_id": result.strategy_id,
                "runtime_dir": str(result.runtime_dir),
                "registry_path": str(result.registry_path),
                "artifact_manifest_path": str(result.artifact_manifest_path),
                "paper_ledger_path": str(result.paper_ledger_path),
                "runtime_config_path": str(result.runtime_config_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
