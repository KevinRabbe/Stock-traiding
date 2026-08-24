from __future__ import annotations

import pytest

from stock_trading.experiments.v5_retrain_attribution import _attribute_retrain_break


def _scenario(name: str, *, ret: float, pf: float, alpha: float, trades: int, ids=None):
    return {
        "scenario": name,
        "return": ret,
        "profit_factor": pf,
        "average_trade_alpha": alpha,
        "total_trades": trades,
        "trade_candidate_ids": ids,
    }


def test_retrain_attribution_identifies_saved_model_break() -> None:
    scenarios = [
        _scenario("legacy_saved_models_reference", ret=0.07, pf=1.6, alpha=0.013, trades=190),
        _scenario(
            "fresh_retrain_unfiltered_20d",
            ret=-0.02,
            pf=0.9,
            alpha=-0.010,
            trades=188,
            ids=["a", "b", "c"],
        ),
        _scenario(
            "fresh_retrain_executable_20d",
            ret=-0.03,
            pf=0.8,
            alpha=-0.012,
            trades=180,
            ids=["a", "b", "d"],
        ),
        _scenario(
            "fresh_retrain_executable_full_maturity",
            ret=-0.025,
            pf=0.85,
            alpha=-0.009,
            trades=177,
            ids=["a", "b", "e"],
        ),
        _scenario(
            "strict_continuous_30pct_reference",
            ret=-0.024,
            pf=0.84,
            alpha=-0.0089,
            trades=176,
        ),
    ]

    result = _attribute_retrain_break(scenarios)

    assert result["primary_break"] == "saved_models_to_fresh_retrain"
    retrain = result["saved_models_to_fresh_retrain"]
    assert retrain["average_trade_alpha_delta"] == pytest.approx(-0.023)
    assert retrain["return_delta"] == pytest.approx(-0.09)
    execution = result["fresh_retrain_to_executable_data"]
    assert execution["trade_jaccard_overlap"] == pytest.approx(0.5)


def test_retrain_attribution_requires_ordered_five_scenarios() -> None:
    with pytest.raises(ValueError, match="exactly five"):
        _attribute_retrain_break(
            [
                _scenario("a", ret=0.0, pf=1.0, alpha=0.0, trades=1),
                _scenario("b", ret=0.0, pf=1.0, alpha=0.0, trades=1),
            ]
        )
