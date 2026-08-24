from datetime import date
from types import SimpleNamespace

import pytest

from stock_trading.experiments.v5_edge_attribution import (
    _HorizonExcludingStrategy,
    _alpha_concentration,
    _remaining_alpha_shape,
    _score_deciles,
    _trade_summary,
    _trigger_attribution,
)
from stock_trading.research.historical import HistoricalTrade


def _trade(
    *,
    candidate_id: str,
    horizon: int = 20,
    score: float = 1.0,
    alpha: float = 0.01,
    gross_return: float = 0.02,
    net_return: float = 0.018,
    pnl: float = 3.6,
    year: int = 2020,
) -> HistoricalTrade:
    return HistoricalTrade(
        strategy_id="lightgbm-v5-adaptive-horizon",
        candidate_id=candidate_id,
        event_id=f"evt-{candidate_id}",
        company_id=f"cmp-{candidate_id}",
        security_id=f"sec-{candidate_id}",
        entry_date=date(year, 1, 2),
        exit_date=date(year, 2, 3),
        horizon_sessions=horizon,
        allocation_pct=0.02,
        allocated_capital=200.0,
        gross_return=gross_return,
        net_return=net_return,
        alpha=alpha,
        downside=0.01,
        pnl=pnl,
        opportunity_score=score,
    )


def _candidate(candidate_id: str, **features: float) -> SimpleNamespace:
    return SimpleNamespace(
        snapshot=SimpleNamespace(
            candidate_id=candidate_id,
            features=features,
        )
    )


def test_trade_summary_separates_stock_benchmark_and_alpha() -> None:
    trades = (
        _trade(
            candidate_id="a",
            alpha=0.01,
            gross_return=0.03,
            net_return=0.028,
            pnl=5.6,
        ),
        _trade(
            candidate_id="b",
            alpha=-0.02,
            gross_return=-0.01,
            net_return=-0.012,
            pnl=-2.4,
        ),
    )

    summary = _trade_summary(trades)

    assert summary["trade_count"] == 2
    assert summary["average_gross_stock_return"] == pytest.approx(0.01)
    assert summary["average_benchmark_return"] == pytest.approx(0.015)
    assert summary["average_trade_alpha"] == pytest.approx(-0.005)
    assert summary["alpha_sum"] == pytest.approx(-0.01)
    assert summary["capital_weighted_alpha_sum"] == pytest.approx(-2.0)
    assert summary["profit_factor"] == pytest.approx(5.6 / 2.4)


def test_score_deciles_are_equal_count_and_ordered_low_to_high() -> None:
    trades = tuple(
        _trade(candidate_id=str(index), score=float(index))
        for index in range(20)
    )

    buckets = _score_deciles(trades)

    assert list(buckets) == [f"D{index:02d}" for index in range(1, 11)]
    assert all(item["trade_count"] == 2 for item in buckets.values())
    assert buckets["D01"]["minimum_opportunity_score"] == 0.0
    assert buckets["D01"]["maximum_opportunity_score"] == 1.0
    assert buckets["D10"]["minimum_opportunity_score"] == 18.0
    assert buckets["D10"]["maximum_opportunity_score"] == 19.0


def test_trigger_attribution_reports_overlaps_and_exclusive_signatures() -> None:
    trades = (
        _trade(candidate_id="a"),
        _trade(candidate_id="b"),
        _trade(candidate_id="c"),
    )
    candidates = {
        "a": _candidate(
            "a",
            **{
                "trigger.is_insider": 1.0,
                "trigger.is_contract": 1.0,
            },
        ),
        "b": _candidate("b", **{"trigger.is_contract": 1.0}),
        "c": _candidate("c"),
    }

    result = _trigger_attribution(trades, candidates)

    assert result["overlapping_families"]["insider"]["trade_count"] == 1
    assert result["overlapping_families"]["contract"]["trade_count"] == 2
    assert result["overlapping_families"]["lobbying"]["trade_count"] == 0
    assert result["exclusive_signatures"]["insider+contract"]["trade_count"] == 1
    assert result["exclusive_signatures"]["contract"]["trade_count"] == 1
    assert result["exclusive_signatures"]["none"]["trade_count"] == 1


def test_remaining_alpha_shape_uses_horizon_then_period_concentration() -> None:
    horizon = {
        "5": {"alpha_sum": -0.8},
        "20": {"alpha_sum": -0.1},
        "60": {"alpha_sum": 0.2},
    }
    years = {
        "2020": {"alpha_sum": -0.3},
        "2021": {"alpha_sum": -0.2},
        "2022": {"alpha_sum": 0.1},
    }
    concentration = _alpha_concentration(horizon, years)

    assert (
        _remaining_alpha_shape(-0.001, concentration)
        == "concentrated_by_horizon"
    )


def test_horizon_excluding_strategy_suppresses_only_selected_horizon() -> None:
    opportunities = (
        SimpleNamespace(horizon_sessions=5),
        SimpleNamespace(horizon_sessions=20),
        SimpleNamespace(horizon_sessions=60),
    )
    delegate = SimpleNamespace(
        strategy_id="v5",
        evaluate=lambda candidates, portfolio: opportunities,
    )
    strategy = _HorizonExcludingStrategy(delegate, 20)

    result = strategy.evaluate((), SimpleNamespace())

    assert [item.horizon_sessions for item in result] == [5, 60]
