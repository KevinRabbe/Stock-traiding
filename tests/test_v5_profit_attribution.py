from __future__ import annotations

import pytest

from stock_trading.experiments.v5_profit_attribution import _attribute_break


def _scenario(name: str, *, ret: float, pf: float, alpha: float, trades: int, ids=None):
    return {
        "scenario": name,
        "return": ret,
        "profit_factor": pf,
        "average_trade_alpha": alpha,
        "total_trades": trades,
        "trade_candidate_ids": ids,
    }


def test_attribution_marks_controlled_maturity_step() -> None:
    scenarios = [
        _scenario(
            "legacy_saved_models_reference",
            ret=0.06,
            pf=1.6,
            alpha=0.013,
            trades=190,
        ),
        _scenario(
            "retrained_executable_20d_maturity",
            ret=0.04,
            pf=1.4,
            alpha=0.010,
            trades=180,
            ids=["a", "b", "c"],
        ),
        _scenario(
            "retrained_executable_full_horizon_maturity",
            ret=-0.03,
            pf=0.8,
            alpha=-0.015,
            trades=175,
            ids=["b", "c", "d"],
        ),
        _scenario(
            "strict_continuous_30pct_reference",
            ret=-0.028,
            pf=0.79,
            alpha=-0.014,
            trades=174,
        ),
    ]

    result = _attribute_break(scenarios)

    assert result["primary_break"] == "retrained_executable_full_horizon_maturity"
    controlled = result["controlled_maturity_step"]
    assert controlled["controlled_maturity_step"] is True
    assert controlled["average_trade_alpha_delta"] == pytest.approx(-0.025)
    assert controlled["return_delta"] == pytest.approx(-0.07)
    assert controlled["trade_jaccard_overlap"] == pytest.approx(0.5)


def test_attribution_handles_missing_trade_identity_reference() -> None:
    scenarios = [
        _scenario("legacy_saved_models_reference", ret=0.01, pf=1.1, alpha=0.002, trades=80),
        _scenario(
            "retrained_executable_20d_maturity",
            ret=0.005,
            pf=1.05,
            alpha=0.001,
            trades=78,
            ids=["a"],
        ),
        _scenario(
            "retrained_executable_full_horizon_maturity",
            ret=0.004,
            pf=1.04,
            alpha=0.0005,
            trades=77,
            ids=["a"],
        ),
        _scenario(
            "strict_continuous_30pct_reference",
            ret=0.003,
            pf=1.03,
            alpha=0.0004,
            trades=76,
        ),
    ]

    result = _attribute_break(scenarios)

    assert result["steps"][0]["trade_jaccard_overlap"] is None
    assert result["steps"][1]["trade_jaccard_overlap"] == pytest.approx(1.0)
