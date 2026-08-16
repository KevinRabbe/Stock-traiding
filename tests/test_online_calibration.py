from datetime import date

import pytest

from stock_trading.ml.online_calibration import RollingScoreHistory


def test_rolling_score_history_keeps_same_day_candidates_out_of_each_others_rank() -> None:
    history = RollingScoreHistory(window_days=365)
    history.seed(((date(2025, 1, 1), 0.1), (date(2025, 1, 1), 0.9)))

    first_batch = history.percentiles(date(2025, 1, 2), (0.5, 0.8))
    next_day = history.percentiles(date(2025, 1, 3), (0.8,), update=False)

    assert first_batch == pytest.approx((0.5, 0.5))
    # Jan-2 scores are visible on Jan-3. Exact 0.8 has mid-rank 2.5 / 4.
    assert next_day == pytest.approx((0.625,))


def test_rolling_score_history_updates_only_eligible_values() -> None:
    history = RollingScoreHistory(window_days=30)
    history.seed(((date(2025, 1, 1), 0.2),))

    percentiles = history.percentiles(
        date(2025, 1, 2),
        (0.3, 0.9),
        eligible=(True, False),
        ineligible_percentile=0.0,
    )

    assert percentiles == pytest.approx((1.0, 0.0))
    assert history.snapshot()[-1] == (date(2025, 1, 2), 0.3)
    assert all(score != 0.9 for _, score in history.snapshot())


def test_rolling_score_history_prunes_values_outside_window() -> None:
    history = RollingScoreHistory(window_days=10)
    history.seed(((date(2025, 1, 1), 0.99), (date(2025, 1, 15), 0.1)))

    result = history.percentiles(date(2025, 1, 20), (0.5,), update=False)

    assert result == pytest.approx((1.0,))
    assert history.snapshot() == ((date(2025, 1, 15), 0.1),)
