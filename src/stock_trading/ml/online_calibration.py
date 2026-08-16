from __future__ import annotations

from bisect import bisect_left, bisect_right
from datetime import date, timedelta
from typing import Iterable, Sequence


class RollingScoreHistory:
    """Stateful PIT percentile calibration for live or chronological replay.

    A batch for ``current_date`` is always ranked only against strictly earlier
    dates. The entire batch is appended after all percentiles are calculated, so
    candidates on the same execution date cannot influence one another.
    """

    def __init__(self, *, window_days: int = 365) -> None:
        if window_days <= 0:
            raise ValueError("window_days must be > 0")
        self.window_days = window_days
        self._history: list[tuple[date, float]] = []

    def seed(self, values: Iterable[tuple[date, float]]) -> None:
        self._history.extend((day, float(score)) for day, score in values)
        self._history.sort(key=lambda item: item[0])

    def percentiles(
        self,
        current_date: date,
        scores: Sequence[float],
        *,
        eligible: Sequence[bool] | None = None,
        ineligible_percentile: float = 0.0,
        update: bool = True,
    ) -> tuple[float, ...]:
        if not 0.0 <= ineligible_percentile <= 1.0:
            raise ValueError("ineligible_percentile must be in [0, 1]")
        if eligible is None:
            eligible = [True] * len(scores)
        if len(scores) != len(eligible):
            raise ValueError("score/eligibility lengths differ")

        cutoff = current_date - timedelta(days=self.window_days)
        self._history = [
            item for item in self._history if cutoff <= item[0] < current_date
        ]
        sorted_scores = sorted(score for _, score in self._history)
        result = tuple(
            _percentile(sorted_scores, float(score))
            if is_eligible
            else ineligible_percentile
            for score, is_eligible in zip(scores, eligible, strict=True)
        )

        if update:
            self._history.extend(
                (current_date, float(score))
                for score, is_eligible in zip(scores, eligible, strict=True)
                if is_eligible
            )
        return result

    def snapshot(self) -> tuple[tuple[date, float], ...]:
        return tuple(self._history)


def _percentile(sorted_values: list[float], value: float) -> float:
    if not sorted_values:
        return 0.5
    left = bisect_left(sorted_values, value)
    right = bisect_right(sorted_values, value)
    return (left + right) / (2.0 * len(sorted_values))
