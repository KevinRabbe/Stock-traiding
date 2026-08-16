from datetime import date, datetime, time, timedelta, timezone

import numpy as np
import pytest

from stock_trading.ml.dataset import TrainingRow
from stock_trading.ml.score_calibration import (
    rolling_filtered_score_percentiles,
    rolling_score_percentiles,
    static_score_percentiles,
)


def _row(event_id: str, execution_day: int) -> TrainingRow:
    execution_date = date(2024, 1, execution_day)
    decision_date = execution_date - timedelta(days=1)
    return TrainingRow(
        event_id=event_id,
        company_id=event_id,
        decision_time=datetime.combine(
            decision_date,
            time(hour=12),
            tzinfo=timezone.utc,
        ),
        execution_date=execution_date,
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
    old_validation = _row("old", 1)
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


def test_static_percentiles_use_mid_ranks_for_ties() -> None:
    result = static_score_percentiles((0.0, 1.0, 1.0, 2.0))
    assert result.tolist() == pytest.approx([0.125, 0.5, 0.5, 0.875])


def test_filtered_calibration_ignores_ineligible_extreme_scores() -> None:
    validation = (_row("eligible-low", 1), _row("blocked-extreme", 2))
    test = (_row("candidate", 5), _row("blocked-test", 5), _row("next", 6))

    result = rolling_filtered_score_percentiles(
        validation,
        (0.0, 1000.0),
        (True, False),
        test,
        (1.0, 5000.0, 2.0),
        (True, False, True),
        window_days=365,
    )

    # The blocked validation extreme never enters history, so 1.0 is top-ranked.
    assert result[0] == pytest.approx(1.0)
    # Ineligible rows receive the explicit non-qualifying percentile and are not
    # appended for future dates.
    assert result[1] == pytest.approx(0.0)
    # The next date sees eligible history [0.0, 1.0], not the blocked 5000.0 row.
    assert result[2] == pytest.approx(1.0)
