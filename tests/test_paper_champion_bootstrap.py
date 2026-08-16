from __future__ import annotations

import json

import pytest

from stock_trading.engine import FileStrategyMetadataStore, StrategyStage
from stock_trading.execution import FilePaperLedger, PaperLedgerState
from stock_trading.live.bootstrap import (
    DEFAULT_STRATEGY_ID,
    _parser,
    bootstrap_v5_paper_champion,
)


def _replay_payload(*, exact=True):
    observed = {
        "compounded_return": 0.0683167704686507,
        "profitable_year_rate": 0.5384615384615384,
        "total_trades": 193,
        "average_trade_alpha": 0.013107422301346023,
        "aggregate_profit_factor": 1.6546587797165533,
        "worst_realized_drawdown": 0.012307985477586224,
    }
    return {
        "schema_version": "strategy-engine-v5-exact-replay",
        "market_db": "data/normalized/market.duckdb",
        "benchmark_security_id": "benchmark_spy",
        "architecture": {
            "generic_strategy_plugin": True,
            "generic_historical_backtester": True,
            "exact_v5_identity_verified": exact,
        },
        "observed": observed,
        "v5_baseline": dict(observed),
    }


def _experiment(tmp_path, *, exact=True):
    root = tmp_path / "experiment"
    models = root / "profit_models_v5" / "2015" / "20d"
    models.mkdir(parents=True)
    (models / "model.txt").write_text("saved model", encoding="utf-8")
    (root / "strategy_engine_v5_replay.json").write_text(
        json.dumps(_replay_payload(exact=exact)),
        encoding="utf-8",
    )
    return root


def test_paper_bootstrap_requires_exact_replay_and_initializes_durable_runtime(tmp_path) -> None:
    experiment = _experiment(tmp_path)
    runtime = tmp_path / "runtime"

    result = bootstrap_v5_paper_champion(
        experiment,
        runtime_dir=runtime,
        starting_capital=10_000.0,
    )

    registry = FileStrategyMetadataStore(result.registry_path).load()
    assert registry is not None
    assert registry.champion_id == DEFAULT_STRATEGY_ID
    assert len(registry.records) == 1
    record = registry.records[0]
    assert record.stage is StrategyStage.PAPER
    assert record.artifact_ref == str(result.artifact_manifest_path)
    assert record.scorecard is not None
    assert record.scorecard.total_trades == 193

    ledger = FilePaperLedger(result.paper_ledger_path).load()
    assert ledger.cash == pytest.approx(10_000.0)
    assert ledger.positions == ()
    assert result.artifact_manifest_path.exists()

    config = json.loads(result.runtime_config_path.read_text(encoding="utf-8"))
    assert config["strategy_id"] == DEFAULT_STRATEGY_ID
    assert config["benchmark_security_id"] == "benchmark_spy"
    assert config["market_db"] == "data/normalized/market.duckdb"


def test_paper_bootstrap_is_idempotent_and_never_resets_existing_ledger(tmp_path) -> None:
    experiment = _experiment(tmp_path)
    runtime = tmp_path / "runtime"
    first = bootstrap_v5_paper_champion(experiment, runtime_dir=runtime)
    FilePaperLedger(first.paper_ledger_path).save(PaperLedgerState(cash=9_123.45))

    second = bootstrap_v5_paper_champion(
        experiment,
        runtime_dir=runtime,
        starting_capital=50_000.0,
    )

    assert second.registry_path == first.registry_path
    assert FilePaperLedger(second.paper_ledger_path).load().cash == pytest.approx(9_123.45)


def test_paper_bootstrap_rejects_unverified_replay(tmp_path) -> None:
    experiment = _experiment(tmp_path, exact=False)

    with pytest.raises(ValueError, match="did not verify exact identity"):
        bootstrap_v5_paper_champion(experiment, runtime_dir=tmp_path / "runtime")


def test_paper_bootstrap_cli_defaults() -> None:
    args = _parser().parse_args(["--experiment-dir", "data/experiments/example"])

    assert args.runtime_dir.as_posix() == "data/runtime"
    assert args.starting_capital == 10_000.0
    assert args.strategy_id == DEFAULT_STRATEGY_ID
