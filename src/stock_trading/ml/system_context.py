from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import replace
from typing import Iterable

from .dataset import TrainingRow
from .opportunity_history import augment_opportunity_history_features


SYSTEM_CONTEXT_FEATURES = (
    "system.regime.benchmark_positive_5d",
    "system.regime.benchmark_positive_20d",
    "system.regime.benchmark_positive_60d",
    "system.regime.benchmark_trend_breadth",
    "system.momentum.stock_5_minus_20",
    "system.momentum.stock_20_minus_60",
    "system.momentum.relative_5_minus_20",
    "system.momentum.relative_20_minus_60",
    "system.volatility.ratio_5_20",
    "system.volatility.ratio_20_60",
    "system.interaction.repeat_x_volatility_20d",
    "system.interaction.repeat_x_benchmark_return_20d",
    "system.interaction.repeat_x_relative_return_20d",
    "system.interaction.trigger_change_x_relative_return_20d",
    "system.cross_section.opportunity_count",
    "system.cross_section.relative_return_5d_percentile",
    "system.cross_section.relative_return_20d_percentile",
    "system.cross_section.relative_return_60d_percentile",
    "system.cross_section.stock_return_20d_percentile",
    "system.cross_section.volatility_20d_percentile",
    "system.cross_section.volume_zscore_20d_percentile",
    "system.cross_section.trigger_count_percentile",
    "system.cross_section.history_count_30d_percentile",
)


_CROSS_SECTION_SOURCES = {
    "system.cross_section.relative_return_5d_percentile": "market.relative_return_5d",
    "system.cross_section.relative_return_20d_percentile": "market.relative_return_20d",
    "system.cross_section.relative_return_60d_percentile": "market.relative_return_60d",
    "system.cross_section.stock_return_20d_percentile": "market.return_20d",
    "system.cross_section.volatility_20d_percentile": "market.volatility_20d",
    "system.cross_section.volume_zscore_20d_percentile": "market.volume_zscore_20d",
    "system.cross_section.trigger_count_percentile": "trigger.event_count",
    "system.cross_section.history_count_30d_percentile": "opportunity_history.count_30d",
}


def augment_system_context_features(
    rows: Iterable[TrainingRow],
) -> tuple[TrainingRow, ...]:
    """Add broad PIT system state without using outcomes or future rows.

    V3 deliberately bundles several cheap, high-leverage context families:
    same-company opportunity history, benchmark regime, cross-horizon momentum,
    volatility shape, repeat/regime interactions, and same-session cross-sectional
    ranks. Cross-sectional ranks only compare opportunities sharing the same
    execution session, so every value is knowable before that session opens.
    """

    materialized = augment_opportunity_history_features(rows)
    if not materialized:
        return ()

    additions: list[dict[str, float | None]] = []
    for row in materialized:
        features = row.features
        benchmark_returns = [
            _number(features, "market.benchmark_return_5d"),
            _number(features, "market.benchmark_return_20d"),
            _number(features, "market.benchmark_return_60d"),
        ]
        present_benchmark_returns = [
            value for value in benchmark_returns if value is not None
        ]
        prior_within_20d = _number(features, "opportunity_history.prior_within_20d")
        trigger_change = _number(
            features, "opportunity_history.trigger_count_change_vs_previous"
        )
        relative_20d = _number(features, "market.relative_return_20d")
        volatility_20d = _number(features, "market.volatility_20d")
        benchmark_20d = _number(features, "market.benchmark_return_20d")

        additions.append(
            {
                "system.regime.benchmark_positive_5d": _positive(
                    _number(features, "market.benchmark_return_5d")
                ),
                "system.regime.benchmark_positive_20d": _positive(benchmark_20d),
                "system.regime.benchmark_positive_60d": _positive(
                    _number(features, "market.benchmark_return_60d")
                ),
                "system.regime.benchmark_trend_breadth": (
                    sum(value > 0 for value in present_benchmark_returns)
                    / len(present_benchmark_returns)
                    if present_benchmark_returns
                    else None
                ),
                "system.momentum.stock_5_minus_20": _difference(
                    _number(features, "market.return_5d"),
                    _number(features, "market.return_20d"),
                ),
                "system.momentum.stock_20_minus_60": _difference(
                    _number(features, "market.return_20d"),
                    _number(features, "market.return_60d"),
                ),
                "system.momentum.relative_5_minus_20": _difference(
                    _number(features, "market.relative_return_5d"),
                    relative_20d,
                ),
                "system.momentum.relative_20_minus_60": _difference(
                    relative_20d,
                    _number(features, "market.relative_return_60d"),
                ),
                "system.volatility.ratio_5_20": _ratio(
                    _number(features, "market.volatility_5d"),
                    volatility_20d,
                ),
                "system.volatility.ratio_20_60": _ratio(
                    volatility_20d,
                    _number(features, "market.volatility_60d"),
                ),
                "system.interaction.repeat_x_volatility_20d": _product(
                    prior_within_20d, volatility_20d
                ),
                "system.interaction.repeat_x_benchmark_return_20d": _product(
                    prior_within_20d, benchmark_20d
                ),
                "system.interaction.repeat_x_relative_return_20d": _product(
                    prior_within_20d, relative_20d
                ),
                "system.interaction.trigger_change_x_relative_return_20d": _product(
                    trigger_change, relative_20d
                ),
            }
        )

    indices_by_execution_date: dict[object, list[int]] = defaultdict(list)
    for index, row in enumerate(materialized):
        indices_by_execution_date[row.execution_date].append(index)

    for indices in indices_by_execution_date.values():
        for index in indices:
            additions[index]["system.cross_section.opportunity_count"] = float(len(indices))
        for target_name, source_name in _CROSS_SECTION_SOURCES.items():
            values = [
                (index, _number(materialized[index].features, source_name))
                for index in indices
            ]
            sorted_values = sorted(value for _, value in values if value is not None)
            for index, value in values:
                additions[index][target_name] = (
                    _percentile_rank(sorted_values, value) if value is not None else None
                )

    return tuple(
        replace(row, features={**row.features, **additions[index]})
        for index, row in enumerate(materialized)
    )


def _number(features: dict[str, float | None], name: str) -> float | None:
    value = features.get(name)
    return float(value) if value is not None else None


def _positive(value: float | None) -> float | None:
    return float(value > 0) if value is not None else None


def _difference(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _product(left: float | None, right: float | None) -> float | None:
    return left * right if left is not None and right is not None else None


def _percentile_rank(sorted_values: list[float], value: float) -> float | None:
    if not sorted_values:
        return None
    left = bisect_left(sorted_values, value)
    right = bisect_right(sorted_values, value)
    # Mid-rank keeps ties deterministic and maps a singleton to the neutral 0.5.
    return (left + right) / (2.0 * len(sorted_values))
