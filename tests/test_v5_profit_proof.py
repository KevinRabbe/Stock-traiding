from __future__ import annotations

import json
from datetime import date

import pytest

from stock_trading.experiments.v5_profit_proof import (
    _AnnualV5StrategyRouter,
    _resolve_market_inputs,
    current_v5_profit_proof_spec,
    minimum_viability,
)


def test_current_v5_profit_proof_spec_is_fixed_to_paper_design() -> None:
    spec = current_v5_profit_proof_spec()

    assert spec.variant_id == "lightgbm-v5-adaptive-horizon"
    assert spec.feature_profile == "full"
    assert spec.training_window_years is None
    assert spec.tree_profile == "baseline"
    assert spec.horizons == (5, 20, 60)
    assert spec.alpha_rank_weight == pytest.approx(0.25)
    assert spec.seed == 42
    assert spec.validation_top_fraction == pytest.approx(0.05)
    assert spec.calibration_window_days == 365
    assert spec.max_expected_downside == pytest.approx(0.06)


def test_minimum_viability_requires_profit_robustness_and_sample_size() -> None:
    passing = minimum_viability(
        {
            "total_return": 0.12,
            "profit_factor": 1.4,
            "total_trades": 100,
            "realized_max_drawdown": 0.04,
            "average_trade_alpha": 0.01,
            "net_profit_excluding_best_entry_year": 500.0,
        }
    )
    assert passing["passes"] is True
    assert passing["verdict"] == "passes_minimum_viability"
    assert all(passing["checks"].values())

    failing = minimum_viability(
        {
            "total_return": 0.12,
            "profit_factor": 1.4,
            "total_trades": 74,
            "realized_max_drawdown": 0.04,
            "average_trade_alpha": 0.01,
            "net_profit_excluding_best_entry_year": 500.0,
        }
    )
    assert failing["passes"] is False
    assert failing["verdict"] == "fails_minimum_viability"
    assert failing["checks"]["minimum_75_trades"] is False


def test_market_inputs_default_to_paper_runtime_config(tmp_path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    market = tmp_path / "market.duckdb"
    (runtime / "paper_runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "market_db": str(market),
                "benchmark_security_id": "benchmark_spy",
                "paper_ledger": str(runtime / "paper_ledger.json"),
            }
        ),
        encoding="utf-8",
    )

    resolved_market, benchmark = _resolve_market_inputs(
        runtime,
        market_db=None,
        benchmark_security_id=None,
    )

    assert resolved_market == market
    assert benchmark == "benchmark_spy"


def test_market_inputs_explicit_pair_does_not_require_runtime_config(tmp_path) -> None:
    market = tmp_path / "market.duckdb"
    resolved_market, benchmark = _resolve_market_inputs(
        tmp_path / "missing-runtime",
        market_db=market,
        benchmark_security_id="benchmark_spy",
    )
    assert resolved_market == market
    assert benchmark == "benchmark_spy"


class _FakeCandidate:
    def __init__(self, year: int) -> None:
        self.execution_date = date(year, 1, 2)


class _FakeStrategy:
    strategy_id = "lightgbm-v5-adaptive-horizon"

    def __init__(self, marker: str) -> None:
        self.marker = marker
        self.calls = 0

    def evaluate(self, candidates, portfolio):
        del candidates, portfolio
        self.calls += 1
        return (self.marker,)


def test_annual_router_uses_execution_year_without_blending_models() -> None:
    older = _FakeStrategy("2025")
    newer = _FakeStrategy("2026")
    router = _AnnualV5StrategyRouter({2025: older, 2026: newer})  # type: ignore[arg-type]

    assert router.evaluate((_FakeCandidate(2026),), object()) == ("2026",)  # type: ignore[arg-type]
    assert newer.calls == 1
    assert older.calls == 0

    with pytest.raises(ValueError, match="crosses model years"):
        router.evaluate(  # type: ignore[arg-type]
            (_FakeCandidate(2025), _FakeCandidate(2026)),
            object(),
        )
