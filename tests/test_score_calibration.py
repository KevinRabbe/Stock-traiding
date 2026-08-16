from datetime import date, datetime, timezone

import numpy as np
import pytest

from stock_trading.ml.dataset import TrainingRow
from stock_trading.ml.score_calibration import rolling_score_percentiles


def _row(event_id: str, execution_day: int) -> TrainingRow:
    return TrainingRow(
        event_id=event_id,
        company_id=event_id,
        decision_time=datetime(2024, 1, execution_day - 1, 12, tzinfo=timezone.utc),
        execution_date=date(2024, 1, execution_day),
        exit_date_20d=date(2024, 2, min(execution_day, 28)),
        features={"x": 1.0},
        stock_return_20d=0.0,
        benchmark_return_20d=0.0,
        alpha_20d=0.0,
        downside_20d=0.0,
        mfe_20d=0.0,
        positive_alpha_20d=0,
    )


def test_same_day_scores_do_not_calibrate_each_other() -> None:
    validation = (_row("v1", 1), _row("v2", 2))
    test = (_row("high", 5), _row("low", 5), _row("next", 6))

    result = rolling_score_percentiles(
        validation,
        np.asarray([0.0, 1.0]),
        test,
        np.asarray([100.0, -100.0, 50.0]),
        window_days=365,
    )

    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.0)
    # The next date can see both prior test scores as legitimate score history.
    assert result[2] == pytest.approx(0.75)


def test_calibration_discards_scores_outside_window() -> None:
    old_validation = TrainingRow(
        **{
            **_row("old", 1).__dict__,
        }
    ) if False else _row("old", 1)
    recent_validation = _row("recent", 20)
    test = (_row("test", 21),)

    result = rolling_score_percentiles(
        (old_validation, recent_validation),
        (1000.0, 0.0),
        test,
        (1.0,),
        window_days=5,
    )
    assert result[0] == pytest.approx(1.0)
