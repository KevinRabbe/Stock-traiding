from datetime import date, datetime, timezone

import pytest

from stock_trading.ml.dataset import TrainingRow
from stock_trading.ml.system_context import augment_system_context_features


def _row(
    event_id: str,
    company_id: str,
    day: int,
    *,
    relative_20d: float,
    stock_return: float = 0.01,
    alpha: float = 0.02,
) -> TrainingRow:
    return TrainingRow(
        event_id=event_id,
        company_id=company_id,
        decision_time=datetime(2024, 1, day, 12, tzinfo=timezone.utc),
        execution_date=date(2024, 1, day + 1),
        exit_date_20d=date(2024, 2, min(day + 1, 28)),
        features={
            "market.return_5d": 0.03,
            "market.return_20d": 0.05,
            "market.return_60d": 0.10,
            "market.relative_return_5d": relative_20d + 0.01,
            "market.relative_return_20d": relative_20d,
            "market.relative_return_60d": relative_20d - 0.02,
            "market.benchmark_return_5d": 0.01,
            "market.benchmark_return_20d": 0.02,
            "market.benchmark_return_60d": -0.01,
            "market.volatility_5d": 0.20,
            "market.volatility_20d": 0.25,
            "market.volatility_60d": 0.30,
            "market.volume_zscore_20d": 1.0,
            "trigger.event_count": 1.0,
        },
        stock_return_20d=stock_return,
        benchmark_return_20d=0.0,
        alpha_20d=alpha,
        downside_20d=0.01,
        mfe_20d=0.02,
        positive_alpha_20d=1,
        trigger_event_ids=(f"{event_id}-trigger",),
    )


def test_system_context_adds_regime_momentum_and_repeat_interactions() -> None:
    first = _row("first", "company-a", 1, relative_20d=0.02)
    second = _row("second", "company-a", 5, relative_20d=0.04)

    augmented = augment_system_context_features((first, second))
    second_features = augmented[1].features

    assert second_features["opportunity_history.prior_within_20d"] == 1.0
    assert second_features["system.regime.benchmark_trend_breadth"] == pytest.approx(2 / 3)
    assert second_features["system.momentum.stock_5_minus_20"] == pytest.approx(-0.02)
    assert second_features["system.volatility.ratio_5_20"] == pytest.approx(0.8)
    assert second_features["system.interaction.repeat_x_volatility_20d"] == pytest.approx(0.25)
    assert second_features["system.interaction.repeat_x_relative_return_20d"] == pytest.approx(0.04)


def test_cross_section_uses_same_execution_session_without_labels() -> None:
    low = _row("low", "company-a", 1, relative_20d=-0.10, stock_return=0.90, alpha=0.80)
    high = _row("high", "company-b", 1, relative_20d=0.20, stock_return=-0.90, alpha=-0.80)

    augmented = augment_system_context_features((high, low))
    by_id = {row.event_id: row for row in augmented}

    assert by_id["low"].features["system.cross_section.opportunity_count"] == 2.0
    assert by_id["low"].features["system.cross_section.relative_return_20d_percentile"] == pytest.approx(0.25)
    assert by_id["high"].features["system.cross_section.relative_return_20d_percentile"] == pytest.approx(0.75)

    changed_low = _row("low", "company-a", 1, relative_20d=-0.10, stock_return=-0.10, alpha=-0.20)
    changed_high = _row("high", "company-b", 1, relative_20d=0.20, stock_return=0.10, alpha=0.20)
    changed = augment_system_context_features((changed_high, changed_low))
    changed_by_id = {row.event_id: row for row in changed}

    for event_id in ("low", "high"):
        original_system = {
            key: value
            for key, value in by_id[event_id].features.items()
            if key.startswith("system.") or key.startswith("opportunity_history.")
        }
        changed_system = {
            key: value
            for key, value in changed_by_id[event_id].features.items()
            if key.startswith("system.") or key.startswith("opportunity_history.")
        }
        assert original_system == changed_system
