from __future__ import annotations

from datetime import date, datetime

import pytest

from stock_trading.ml.dataset import TrainingRow
from stock_trading.ml.multi_horizon import HorizonTarget, multi_horizon_maturity_dates
from stock_trading.ml.walk_forward import annual_walk_forward_splits


def _row(event_id: str, decision_day: date, exit_20d: date) -> TrainingRow:
    return TrainingRow(
        event_id=event_id,
        company_id=f"company-{event_id}",
        decision_time=datetime.combine(decision_day, datetime.min.time()),
        execution_date=decision_day,
        exit_date_20d=exit_20d,
        features={"x": 1.0},
        stock_return_20d=0.01,
        benchmark_return_20d=0.0,
        alpha_20d=0.01,
        downside_20d=0.01,
        mfe_20d=0.02,
        positive_alpha_20d=0,
    )


def _target(horizon: int, exit_day: date) -> HorizonTarget:
    return HorizonTarget(
        horizon=horizon,
        exit_date=exit_day,
        stock_return=0.01,
        benchmark_return=0.0,
        alpha=0.01,
        downside=0.01,
        mfe=0.02,
    )


def test_full_horizon_maturity_excludes_validation_row_crossing_test_year() -> None:
    train = _row("train", date(2023, 6, 1), date(2023, 7, 1))
    safe_validation = _row("safe", date(2024, 5, 1), date(2024, 6, 1))
    leaking_validation = _row("leak", date(2024, 11, 15), date(2024, 12, 20))
    test = _row("test", date(2025, 3, 1), date(2025, 4, 1))
    rows = (train, safe_validation, leaking_validation, test)

    legacy = annual_walk_forward_splits(rows, first_test_year=2025)
    assert len(legacy) == 1
    assert {row.event_id for row in legacy[0].validation_rows} == {"safe", "leak"}

    maturity_dates = {
        "train": date(2023, 8, 1),
        "safe": date(2024, 8, 1),
        "leak": date(2025, 2, 15),
        "test": date(2025, 6, 1),
    }
    safe = annual_walk_forward_splits(
        rows,
        first_test_year=2025,
        maturity_dates=maturity_dates,
    )

    assert len(safe) == 1
    assert [row.event_id for row in safe[0].validation_rows] == ["safe"]
    assert all(maturity_dates[row.event_id] < date(2025, 1, 1) for row in safe[0].validation_rows)


def test_multi_horizon_maturity_uses_latest_requested_exit() -> None:
    row = _row("event", date(2024, 10, 1), date(2024, 11, 1))
    targets = {
        "event": {
            5: _target(5, date(2024, 10, 8)),
            20: _target(20, date(2024, 11, 1)),
            60: _target(60, date(2025, 1, 15)),
        }
    }

    assert multi_horizon_maturity_dates((row,), targets, horizons=(5, 20))["event"] == date(2024, 11, 1)
    assert multi_horizon_maturity_dates((row,), targets, horizons=(5, 20, 60))["event"] == date(2025, 1, 15)


def test_walk_forward_requires_complete_explicit_maturity_mapping() -> None:
    row = _row("event", date(2024, 1, 1), date(2024, 2, 1))

    with pytest.raises(ValueError, match="maturity_dates missing rows"):
        annual_walk_forward_splits((row,), maturity_dates={})
