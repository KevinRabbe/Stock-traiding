from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from stock_trading.experiments.v5_execution_filter_attribution import (
    _attribute_filter_break,
    _classify_adv_row,
    _quality_valid_rows,
)


def _row(event_id: str, value=100_000.0, *, year: int = 2020):
    features = {}
    if value is not ...:
        features["market.avg_dollar_volume_20d"] = value
    return SimpleNamespace(
        event_id=event_id,
        execution_date=date(year, 6, 1),
        features=features,
    )


def _scenario(name: str, *, ret: float, pf: float, alpha: float, trades: int, ids):
    return {
        "scenario": name,
        "return": ret,
        "profit_factor": pf,
        "average_trade_alpha": alpha,
        "total_trades": trades,
        "trade_candidate_ids": list(ids),
    }


def test_adv_classification_separates_missing_from_true_low_liquidity() -> None:
    missing = _classify_adv_row(
        _row("missing", ...),
        required_capital=200.0,
        max_participation_pct=0.01,
    )
    low = _classify_adv_row(
        _row("low", 10_000.0),
        required_capital=200.0,
        max_participation_pct=0.01,
    )
    passing = _classify_adv_row(
        _row("pass", 20_000.0),
        required_capital=200.0,
        max_participation_pct=0.01,
    )

    assert missing.reason == "missing"
    assert missing.value is None
    assert low.reason == "below_required"
    assert low.value == 10_000.0
    assert passing.reason == "passes"
    assert passing.value == 20_000.0


@pytest.mark.parametrize("value", [True, "bad", float("nan"), float("inf")])
def test_adv_classification_rejects_invalid_values(value) -> None:
    result = _classify_adv_row(
        _row("invalid", value),
        required_capital=200.0,
        max_participation_pct=0.01,
    )
    assert result.reason == "invalid"


def test_quality_filter_requires_all_requested_horizons_to_be_valid() -> None:
    rows = (_row("a"), _row("b"), _row("c"))
    result = _quality_valid_rows(
        rows,
        horizons=(5, 20, 60),
        invalid_target_keys=frozenset({("b", 60), ("c", 20)}),
    )
    assert [row.event_id for row in result] == ["a"]


def test_filter_attribution_identifies_adv_break_and_reports_overlap() -> None:
    scenarios = [
        _scenario("unfiltered_20d", ret=0.07, pf=1.6, alpha=0.013, trades=4, ids=["a", "b", "c", "d"]),
        _scenario("quality_only_20d", ret=0.068, pf=1.58, alpha=0.012, trades=4, ids=["a", "b", "c", "d"]),
        _scenario(
            "quality_plus_trailing_adv_20d",
            ret=-0.04,
            pf=0.7,
            alpha=-0.02,
            trades=3,
            ids=["a", "x", "y"],
        ),
        _scenario(
            "quality_adv_plus_entry_fill_20d",
            ret=-0.042,
            pf=0.69,
            alpha=-0.021,
            trades=2,
            ids=["a", "x"],
        ),
    ]

    result = _attribute_filter_break(scenarios)

    assert result["primary_break"] == "pit_trailing_adv_filter"
    assert result["steps"][0]["trade_jaccard_overlap"] == pytest.approx(1.0)
    assert result["steps"][1]["trade_jaccard_overlap"] == pytest.approx(1 / 6)
    assert result["steps"][2]["trade_jaccard_overlap"] == pytest.approx(2 / 3)
    assert result["steps"][1]["average_trade_alpha_delta"] == pytest.approx(-0.032)
