from __future__ import annotations

import json
from pathlib import Path

import pytest

from stock_trading.experiments.freeze_v5_runtime_plugin import (
    _load_replay,
    _require_year_identity,
    _verification_from_metadata,
)


def _replay() -> dict:
    return {
        "schema_version": "strategy-engine-v5-exact-replay",
        "market_db": "data/normalized/market.duckdb",
        "benchmark_security_id": "benchmark_spy",
        "strategy": {
            "strategy_id": "lightgbm-v5-adaptive-horizon",
            "horizons": [5, 20, 60],
        },
        "architecture": {
            "generic_strategy_plugin": True,
            "generic_historical_backtester": True,
            "saved_models_reused": True,
            "predictor_retrained": False,
            "exact_v5_identity_verified": True,
        },
        "years": [
            {
                "year": 2026,
                "trades": 2,
                "return": 0.0125,
                "identity_verified_against_v5": True,
            }
        ],
    }


def test_load_replay_requires_exact_saved_model_identity(tmp_path: Path) -> None:
    path = tmp_path / "replay.json"
    payload = _replay()
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = _load_replay(
        path,
        strategy_id="lightgbm-v5-adaptive-horizon",
        model_year=2026,
    )
    assert loaded["architecture"]["predictor_retrained"] is False

    payload["architecture"]["predictor_retrained"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpectedly retrained"):
        _load_replay(
            path,
            strategy_id="lightgbm-v5-adaptive-horizon",
            model_year=2026,
        )


def test_serialized_year_identity_is_exact() -> None:
    expected = {
        "trades": 2,
        "return": 0.0125,
    }
    _require_year_identity(0.0125, 2, expected, 2026)

    with pytest.raises(RuntimeError, match="trade count diverged"):
        _require_year_identity(0.0125, 3, expected, 2026)
    with pytest.raises(RuntimeError, match="return diverged"):
        _require_year_identity(0.01250000001, 2, expected, 2026)


def test_existing_self_contained_artifact_requires_replay_verification(tmp_path: Path) -> None:
    metadata = tmp_path / "strategy.json"
    metadata.write_text(
        json.dumps(
            {
                "training": {
                    "serialized_replay_verified": True,
                    "verified_test_return": 0.0,
                    "verified_test_trade_count": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    result = _verification_from_metadata(metadata)
    assert result == {"return": 0.0, "trades": 0}

    metadata.write_text(json.dumps({"training": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid self-contained V5 metadata"):
        _verification_from_metadata(metadata)
