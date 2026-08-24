from datetime import UTC, date, datetime

from stock_trading.experiments.v5_horizon_quality_profit_proof import (
    _adv_filter_rows,
    _candidate_all_horizons_quality_valid,
    _invalid_counts_by_horizon,
    _label_is_eligible,
)
from stock_trading.ml.dataset import TrainingRow
from stock_trading.ml.multi_horizon import HorizonTarget


def _row(*, event_id: str = "evt", adv: float | None = 50_000.0) -> TrainingRow:
    features = {"x": 1.0}
    if adv is not None:
        features["market.avg_dollar_volume_20d"] = adv
    return TrainingRow(
        event_id=event_id,
        company_id="cmp",
        decision_time=datetime(2015, 12, 1, tzinfo=UTC),
        execution_date=date(2015, 12, 2),
        exit_date_20d=date(2015, 12, 30),
        features=features,
        stock_return_20d=0.01,
        benchmark_return_20d=0.0,
        alpha_20d=0.01,
        downside_20d=0.0,
        mfe_20d=0.02,
        positive_alpha_20d=0,
    )


def _target(*, horizon: int, exit_date: date) -> HorizonTarget:
    return HorizonTarget(
        horizon=horizon,
        exit_date=exit_date,
        stock_return=0.01,
        benchmark_return=0.0,
        alpha=0.01,
        downside=0.0,
        mfe=0.02,
    )


def test_quality_is_horizon_local_not_row_wide() -> None:
    row = _row()
    invalid = frozenset({("evt", 60)})
    test_start = date(2016, 1, 1)

    assert _label_is_eligible(
        row,
        _target(horizon=20, exit_date=date(2015, 12, 30)),
        horizon=20,
        test_start=test_start,
        invalid_target_keys=invalid,
    )
    assert not _label_is_eligible(
        row,
        _target(horizon=60, exit_date=date(2015, 12, 30)),
        horizon=60,
        test_start=test_start,
        invalid_target_keys=invalid,
    )
    assert not _candidate_all_horizons_quality_valid(
        "evt",
        (5, 20, 60),
        invalid,
    )


def test_maturity_is_horizon_local() -> None:
    row = _row()
    invalid = frozenset()
    test_start = date(2016, 1, 1)

    assert _label_is_eligible(
        row,
        _target(horizon=20, exit_date=date(2015, 12, 30)),
        horizon=20,
        test_start=test_start,
        invalid_target_keys=invalid,
    )
    assert not _label_is_eligible(
        row,
        _target(horizon=60, exit_date=date(2016, 2, 1)),
        horizon=60,
        test_start=test_start,
        invalid_target_keys=invalid,
    )


def test_adv_filter_can_be_applied_only_to_candidate_roles() -> None:
    liquid = _row(event_id="liquid", adv=50_000.0)
    illiquid = _row(event_id="illiquid", adv=10_000.0)
    missing = _row(event_id="missing", adv=None)

    kept, removed = _adv_filter_rows(
        (liquid, illiquid, missing),
        required_capital=200.0,
        max_participation_pct=0.01,
    )

    assert [row.event_id for row in kept] == ["liquid"]
    assert removed == 2


def test_invalid_counts_are_reported_per_horizon() -> None:
    assert _invalid_counts_by_horizon(
        frozenset(
            {
                ("a", 60),
                ("b", 60),
                ("c", 20),
            }
        )
    ) == {"20": 1, "60": 2}
