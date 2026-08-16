from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import date, timedelta
from typing import Sequence

import numpy as np

from .dataset import TrainingRow


def rolling_score_percentiles(
    validation_rows: Sequence[TrainingRow],
    validation_scores: Sequence[float],
    test_rows: Sequence[TrainingRow],
    test_scores: Sequence[float],
    *,
    window_days: int = 365,
) -> np.ndarray:
    """Map raw model scores to PIT trailing-distribution percentiles.

    Validation scores seed the history. Test execution dates are processed in
    chronological batches. Every candidate on a date is calibrated only against
    scores from strictly earlier execution dates, then the whole current batch is
    appended for future dates. No outcome labels are used.
    """

    if window_days <= 0:
        raise ValueError("window_days must be > 0")
    if len(validation_rows) != len(validation_scores):
        raise ValueError("validation row/score lengths differ")
    if len(test_rows) != len(test_scores):
        raise ValueError("test row/score lengths differ")

    history = sorted(
        (
            (row.execution_date, float(score))
            for row, score in zip(validation_rows, validation_scores, strict=True)
        ),
        key=lambda item: item[0],
    )
    indices_by_date: dict[date, list[int]] = defaultdict(list)
    for index, row in enumerate(test_rows):
        indices_by_date[row.execution_date].append(index)

    result = np.empty(len(test_rows), dtype=np.float64)
    for current_date in sorted(indices_by_date):
        cutoff = current_date - timedelta(days=window_days)
        history = [item for item in history if cutoff <= item[0] < current_date]
        sorted_history_scores = sorted(score for _, score in history)
        current_indices = indices_by_date[current_date]

        for index in current_indices:
            result[index] = _percentile(sorted_history_scores, float(test_scores[index]))

        history.extend(
            (current_date, float(test_scores[index])) for index in current_indices
        )
    return result


def _percentile(sorted_values: list[float], value: float) -> float:
    if not sorted_values:
        return 0.5
    left = bisect_left(sorted_values, value)
    right = bisect_right(sorted_values, value)
    return (left + right) / (2.0 * len(sorted_values))
