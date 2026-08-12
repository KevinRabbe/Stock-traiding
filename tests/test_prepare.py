from datetime import date

import pytest

from stock_trading.experiments.prepare import latest_completed_quarter, quarter_range
from stock_trading.ml import TrainingDatasetBuilder


def test_latest_completed_quarter() -> None:
    assert latest_completed_quarter(date(2026, 8, 12)) == (2026, 2)
    assert latest_completed_quarter(date(2026, 4, 1)) == (2026, 1)
    assert latest_completed_quarter(date(2026, 1, 15)) == (2025, 4)


def test_quarter_range_crosses_year_boundary() -> None:
    assert quarter_range(2025, 3, 2026, 2) == (
        (2025, 3),
        (2025, 4),
        (2026, 1),
        (2026, 2),
    )


def test_training_dataset_rejects_non_20_day_target() -> None:
    with pytest.raises(ValueError, match="20-day"):
        TrainingDatasetBuilder(object(), target_horizon=60)
