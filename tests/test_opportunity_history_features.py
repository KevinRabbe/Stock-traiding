from datetime import date, datetime, timezone

import pytest

from stock_trading.ml.dataset import TrainingRow
from stock_trading.ml.opportunity_history import augment_opportunity_history_features


def _row(
    event_id: str,
    day: int,
    *,
    trigger_count: int = 1,
    company_id: str = "company-a",
    decision_hour: int = 12,
    stock_return: float = 0.01,
    alpha: float = 0.02,
) -> TrainingRow:
    decision = datetime(2024, 1, day, decision_hour, tzinfo=timezone.utc)
    return TrainingRow(
        event_id=event_id,
        company_id=company_id,
        decision_time=decision,
        execution_date=date(2024, 1, day),
        exit_date_20d=date(2024, 2, min(day, 28)),
        features={"trigger.event_count": float(trigger_count), "base": float(day)},
        stock_return_20d=stock_return,
        benchmark_return_20d=0.0,
        alpha_20d=alpha,
        downside_20d=0.01,
        mfe_20d=0.02,
        positive_alpha_20d=1,
        trigger_event_ids=tuple(f"{event_id}-trigger-{i}" for i in range(trigger_count)),
    )


def test_opportunity_history_uses_strictly_prior_company_rows_and_preserves_order() -> None:
    first = _row("first", 1, trigger_count=1)
    second = _row("second", 5, trigger_count=2)
    third = _row("third", 12, trigger_count=1)

    # Deliberately pass rows out of chronological order. Feature state must be
    # derived from timestamps, not caller order, and returned order must be stable.
    augmented = augment_opportunity_history_features((third, first, second))
    assert [row.event_id for row in augmented] == ["third", "first", "second"]
    by_id = {row.event_id: row for row in augmented}

    assert by_id["first"].features["opportunity_history.has_previous"] == 0.0
    assert by_id["first"].features["opportunity_history.days_since_previous"] is None
    assert by_id["first"].features["opportunity_history.count_90d"] == 0.0

    second_features = by_id["second"].features
    assert second_features["opportunity_history.days_since_previous"] == pytest.approx(4.0)
    assert second_features["opportunity_history.count_7d"] == 1.0
    assert second_features["opportunity_history.count_14d"] == 1.0
    assert second_features["opportunity_history.trigger_count_7d"] == 1.0
    assert second_features["opportunity_history.previous_trigger_count"] == 1.0
    assert second_features["opportunity_history.trigger_count_change_vs_previous"] == 1.0
    assert second_features["opportunity_history.prior_within_20d"] == 1.0

    third_features = by_id["third"].features
    assert third_features["opportunity_history.days_since_previous"] == pytest.approx(7.0)
    assert third_features["opportunity_history.count_7d"] == 1.0
    assert third_features["opportunity_history.count_14d"] == 2.0
    assert third_features["opportunity_history.count_30d"] == 2.0
    assert third_features["opportunity_history.trigger_count_7d"] == 2.0
    assert third_features["opportunity_history.trigger_count_30d"] == 3.0
    assert third_features["opportunity_history.previous_trigger_count"] == 2.0
    assert third_features["opportunity_history.trigger_count_change_vs_previous"] == -1.0


def test_same_timestamp_rows_cannot_observe_each_other() -> None:
    first = _row("first", 1)
    same_a = _row("same-a", 5, trigger_count=2)
    same_b = _row("same-b", 5, trigger_count=4)

    augmented = augment_opportunity_history_features((same_b, first, same_a))
    by_id = {row.event_id: row for row in augmented}

    for event_id in ("same-a", "same-b"):
        features = by_id[event_id].features
        assert features["opportunity_history.count_7d"] == 1.0
        assert features["opportunity_history.previous_trigger_count"] == 1.0
        assert features["opportunity_history.days_since_previous"] == pytest.approx(4.0)


def test_history_features_do_not_depend_on_realized_labels() -> None:
    prior = _row("prior", 1, trigger_count=3, stock_return=0.50, alpha=0.40)
    current = _row("current", 5, trigger_count=1, stock_return=-0.20, alpha=-0.30)
    changed_prior = _row(
        "prior",
        1,
        trigger_count=3,
        stock_return=-0.90,
        alpha=-0.80,
    )

    original = augment_opportunity_history_features((prior, current))[1]
    changed = augment_opportunity_history_features((changed_prior, current))[1]

    history_keys = [
        key for key in original.features if key.startswith("opportunity_history.")
    ]
    assert history_keys
    assert {key: original.features[key] for key in history_keys} == {
        key: changed.features[key] for key in history_keys
    }
