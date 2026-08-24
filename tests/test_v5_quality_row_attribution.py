from datetime import UTC, date, datetime

import pytest

from stock_trading.experiments.v5_quality_row_attribution import (
    _jaccard,
    _scorecard_delta,
    _walk_forward_roles,
)
from stock_trading.ml.dataset import TrainingRow


def _row(event_id: str, year: int) -> TrainingRow:
    return TrainingRow(
        event_id=event_id,
        company_id=f"cmp_{event_id}",
        decision_time=datetime(year, 6, 1, tzinfo=UTC),
        execution_date=date(year, 6, 2),
        exit_date_20d=date(year, 7, 1),
        features={"x": 1.0},
        stock_return_20d=0.01,
        benchmark_return_20d=0.0,
        alpha_20d=0.01,
        downside_20d=0.0,
        mfe_20d=0.02,
        positive_alpha_20d=0,
    )


def test_walk_forward_roles_show_validation_then_training_influence() -> None:
    rows = tuple(_row(f"evt_{year}", year) for year in range(2012, 2016))

    roles = _walk_forward_roles(rows)

    assert roles["evt_2013"]["validation_for_test_years"] == [2014]
    assert roles["evt_2013"]["train_for_test_years"] == [2015]
    assert roles["evt_2013"]["test_years"] == [2013]


def test_scorecard_delta_reports_direction_and_trade_change() -> None:
    previous = {
        "return": 0.06,
        "profit_factor": 1.6,
        "average_trade_alpha": 0.013,
        "total_trades": 193,
    }
    current = {
        "return": 0.025,
        "profit_factor": 1.2,
        "average_trade_alpha": -0.005,
        "total_trades": 184,
    }

    delta = _scorecard_delta(previous, current)

    assert delta["return_delta"] == pytest.approx(-0.035)
    assert delta["profit_factor_delta"] == pytest.approx(-0.4)
    assert delta["average_trade_alpha_delta"] == pytest.approx(-0.018)
    assert delta["trade_count_delta"] == -9


def test_jaccard_measures_trade_set_instability() -> None:
    assert _jaccard(("a", "b", "c"), ("b", "c", "d")) == pytest.approx(0.5)
    assert _jaccard((), ()) == 1.0
