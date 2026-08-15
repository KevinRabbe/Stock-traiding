from dataclasses import dataclass
from datetime import date

import numpy as np
import pytest

from stock_trading.experiments.lightgbm_profit_rolling_rank import _rolling_selected_indices


@dataclass(frozen=True)
class _Row:
    execution_date: date


def test_rolling_gate_uses_only_prior_sessions() -> None:
    validation_rows = tuple(_Row(date(2023, 12, day)) for day in (1, 2, 3, 4))
    validation_scores = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    test_rows = (
        _Row(date(2024, 1, 2)),
        _Row(date(2024, 1, 2)),
        _Row(date(2024, 1, 3)),
    )
    test_scores = np.asarray([100.0, 50.0, 20.0], dtype=np.float64)

    selected, thresholds = _rolling_selected_indices(
        validation_rows,
        validation_scores,
        test_rows,
        test_scores,
        target_fraction=0.25,
        calibration_days=365,
    )

    # The first session sees only validation scores, so both same-session scores
    # are compared with the same 75th-percentile threshold. Those extreme scores
    # only affect the following session's threshold.
    assert thresholds[date(2024, 1, 2)] == pytest.approx(2.25)
    assert thresholds[date(2024, 1, 3)] > 20.0
    assert selected == (0, 1)


def test_rolling_gate_drops_scores_outside_calibration_window() -> None:
    validation_rows = (
        _Row(date(2023, 1, 1)),
        _Row(date(2023, 12, 31)),
    )
    validation_scores = np.asarray([100.0, 0.0], dtype=np.float64)
    test_rows = (
        _Row(date(2024, 1, 2)),
        _Row(date(2024, 12, 31)),
    )
    test_scores = np.asarray([1.0, 2.0], dtype=np.float64)

    selected, thresholds = _rolling_selected_indices(
        validation_rows,
        validation_scores,
        test_rows,
        test_scores,
        target_fraction=0.5,
        calibration_days=365,
    )

    # The old 100 score is already outside the one-year window by the first test
    # session; the later threshold is therefore calibrated from recent history.
    assert thresholds[date(2024, 1, 2)] == pytest.approx(0.0)
    assert thresholds[date(2024, 12, 31)] == pytest.approx(1.0)
    assert selected == (0, 1)
