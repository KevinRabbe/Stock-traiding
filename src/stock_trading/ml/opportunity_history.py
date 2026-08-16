from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import replace
from datetime import timedelta
from typing import Iterable

from .dataset import TrainingRow


OPPORTUNITY_HISTORY_FEATURES = (
    "opportunity_history.has_previous",
    "opportunity_history.days_since_previous",
    "opportunity_history.count_7d",
    "opportunity_history.count_14d",
    "opportunity_history.count_30d",
    "opportunity_history.count_90d",
    "opportunity_history.trigger_count_7d",
    "opportunity_history.trigger_count_30d",
    "opportunity_history.trigger_count_90d",
    "opportunity_history.prior_within_20d",
    "opportunity_history.previous_trigger_count",
    "opportunity_history.trigger_count_change_vs_previous",
)


def augment_opportunity_history_features(
    rows: Iterable[TrainingRow],
) -> tuple[TrainingRow, ...]:
    """Add same-company opportunity-history state using strictly prior rows only.

    The augmentation intentionally uses no realized return, alpha, downside, exit
    date, or other label-derived value. For each company, only opportunities with
    ``decision_time`` strictly earlier than the current opportunity are visible.
    Rows that share the same decision timestamp therefore cannot observe one
    another, regardless of input ordering.

    The returned tuple preserves the caller's original row order so existing
    walk-forward splitting behavior is unchanged.
    """

    materialized = tuple(rows)
    if not materialized:
        return ()

    indices_by_company: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(materialized):
        indices_by_company[row.company_id].append(index)

    additions: list[dict[str, float | None] | None] = [None] * len(materialized)
    for company_indices in indices_by_company.values():
        ordered_indices = sorted(
            company_indices,
            key=lambda index: (
                materialized[index].decision_time,
                materialized[index].execution_date,
                materialized[index].event_id,
            ),
        )
        times = [materialized[index].decision_time for index in ordered_indices]
        trigger_counts = [_trigger_count(materialized[index]) for index in ordered_indices]
        trigger_prefix = [0.0]
        for count in trigger_counts:
            trigger_prefix.append(trigger_prefix[-1] + count)

        for ordered_position, row_index in enumerate(ordered_indices):
            row = materialized[row_index]
            current_time = row.decision_time
            # bisect_left excludes every row at the same decision timestamp, not
            # just the current row, which keeps the feature state strictly PIT.
            prior_end = bisect_left(times, current_time, 0, ordered_position + 1)
            previous_index = ordered_indices[prior_end - 1] if prior_end > 0 else None
            previous_row = (
                materialized[previous_index] if previous_index is not None else None
            )
            previous_trigger_count = (
                _trigger_count(previous_row) if previous_row is not None else None
            )
            current_trigger_count = _trigger_count(row)
            days_since_previous = (
                (current_time - previous_row.decision_time).total_seconds() / 86_400.0
                if previous_row is not None
                else None
            )

            counts: dict[int, int] = {}
            trigger_sums: dict[int, float] = {}
            for window_days in (7, 14, 30, 90):
                window_start = current_time - timedelta(days=window_days)
                start = bisect_left(times, window_start, 0, prior_end)
                counts[window_days] = prior_end - start
                trigger_sums[window_days] = trigger_prefix[prior_end] - trigger_prefix[start]

            additions[row_index] = {
                "opportunity_history.has_previous": float(previous_row is not None),
                "opportunity_history.days_since_previous": days_since_previous,
                "opportunity_history.count_7d": float(counts[7]),
                "opportunity_history.count_14d": float(counts[14]),
                "opportunity_history.count_30d": float(counts[30]),
                "opportunity_history.count_90d": float(counts[90]),
                "opportunity_history.trigger_count_7d": trigger_sums[7],
                "opportunity_history.trigger_count_30d": trigger_sums[30],
                "opportunity_history.trigger_count_90d": trigger_sums[90],
                "opportunity_history.prior_within_20d": (
                    float(days_since_previous <= 20.0)
                    if days_since_previous is not None
                    else 0.0
                ),
                "opportunity_history.previous_trigger_count": previous_trigger_count,
                "opportunity_history.trigger_count_change_vs_previous": (
                    current_trigger_count - previous_trigger_count
                    if previous_trigger_count is not None
                    else None
                ),
            }

    return tuple(
        replace(row, features={**row.features, **(additions[index] or {})})
        for index, row in enumerate(materialized)
    )


def _trigger_count(row: TrainingRow) -> float:
    if row.trigger_event_ids:
        return float(len(row.trigger_event_ids))
    feature_value = row.features.get("trigger.event_count")
    if feature_value is not None:
        return float(feature_value)
    return 1.0
