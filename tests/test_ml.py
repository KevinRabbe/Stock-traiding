from dataclasses import replace
from datetime import date, datetime, timezone

import numpy as np
import pytest

from stock_trading.ml import (
    FeatureSchema,
    LightGbmModelBundle,
    LightGbmTrainer,
    LightGbmTrainingConfig,
    ProfitLightGbmModelBundle,
    ProfitLightGbmTrainer,
    TrainingRow,
)
from stock_trading.ml.lightgbm_models import _company_balanced_weights
from stock_trading.ml.walk_forward import annual_walk_forward_splits


def _row(
    index: int,
    *,
    year: int = 2024,
    signal: float | None = None,
    exit_date: date | None = None,
) -> TrainingRow:
    value = (index % 20) / 19.0 if signal is None else signal
    alpha = (value - 0.5) * 0.10
    benchmark_return = 0.01
    stock_return = benchmark_return + alpha
    execution = date(year, 6, 1)
    return TrainingRow(
        event_id=f"event-{year}-{index}",
        company_id=f"cmp-{index % 7}",
        decision_time=datetime(year, 5, 31, 20, 0, tzinfo=timezone.utc),
        execution_date=execution,
        exit_date_20d=exit_date or date(year, 6, 28),
        features={
            "signal": value,
            "sometimes_missing": None if index % 3 == 0 else value * 2,
        },
        stock_return_20d=stock_return,
        benchmark_return_20d=benchmark_return,
        alpha_20d=alpha,
        downside_20d=0.07 - 0.05 * value,
        mfe_20d=max(0.0, stock_return + 0.02),
        positive_alpha_20d=int(alpha >= 0.01),
    )


def test_feature_schema_preserves_missing_values_as_nan() -> None:
    rows = (_row(0), _row(1))
    schema = FeatureSchema.from_rows(rows)
    matrix = schema.matrix(rows)

    assert schema.names == ("signal", "sometimes_missing")
    assert matrix.shape == (2, 2)
    assert np.isnan(matrix[0, 1])
    assert not np.isnan(matrix[1, 1])


def test_company_balanced_weights_equalize_company_mass() -> None:
    rows = (
        replace(_row(0), company_id="cmp_a"),
        replace(_row(1), company_id="cmp_a"),
        replace(_row(2), company_id="cmp_a"),
        replace(_row(3), company_id="cmp_b"),
    )
    weights = _company_balanced_weights(rows)

    assert weights.mean() == pytest.approx(1.0)
    assert weights[:3].sum() == pytest.approx(weights[3])


def test_lightgbm_bundle_learns_signal_and_round_trips(tmp_path) -> None:
    rows = tuple(_row(index) for index in range(100))
    trainer = LightGbmTrainer(
        LightGbmTrainingConfig(
            num_boost_round=80,
            early_stopping_rounds=10,
            learning_rate=0.1,
            num_leaves=7,
            min_data_in_leaf=3,
            feature_fraction=1.0,
            bagging_fraction=1.0,
            bagging_freq=0,
            downside_penalty=0.5,
            seed=7,
        )
    )
    bundle = trainer.train(rows[:80], rows[80:])

    low = bundle.predict({"signal": 0.05, "sometimes_missing": None})
    high = bundle.predict({"signal": 0.95, "sometimes_missing": 1.9})

    assert high.expected_alpha_20d > low.expected_alpha_20d
    assert high.probability_positive_alpha > low.probability_positive_alpha
    assert high.expected_downside_20d < low.expected_downside_20d
    assert set(bundle.feature_importance()) == {"signal", "sometimes_missing"}

    bundle.save(tmp_path / "model")
    loaded = LightGbmModelBundle.load(tmp_path / "model")
    reloaded = loaded.predict({"signal": 0.95, "sometimes_missing": 1.9})

    assert reloaded.expected_alpha_20d == pytest.approx(high.expected_alpha_20d)
    assert reloaded.expected_downside_20d == pytest.approx(high.expected_downside_20d)
    assert reloaded.probability_positive_alpha == pytest.approx(
        high.probability_positive_alpha
    )


def test_profit_lightgbm_bundle_targets_realized_stock_return_and_round_trips(tmp_path) -> None:
    rows = tuple(_row(index) for index in range(100))
    trainer = ProfitLightGbmTrainer(
        LightGbmTrainingConfig(
            num_boost_round=80,
            early_stopping_rounds=10,
            learning_rate=0.1,
            num_leaves=7,
            min_data_in_leaf=3,
            feature_fraction=1.0,
            bagging_fraction=1.0,
            bagging_freq=0,
            downside_penalty=0.5,
            seed=7,
        )
    )
    bundle = trainer.train(
        rows[:80],
        rows[80:],
        profitable_return_threshold=0.002,
    )

    low = bundle.predict({"signal": 0.05, "sometimes_missing": None})
    high = bundle.predict({"signal": 0.95, "sometimes_missing": 1.9})

    assert high.expected_stock_return_20d > low.expected_stock_return_20d
    assert high.probability_profitable_return > low.probability_profitable_return
    assert high.expected_downside_20d < low.expected_downside_20d
    assert high.profit_score > low.profit_score
    assert set(bundle.feature_importance()) == {"signal", "sometimes_missing"}

    bundle.save(tmp_path / "profit-model")
    loaded = ProfitLightGbmModelBundle.load(tmp_path / "profit-model")
    reloaded = loaded.predict({"signal": 0.95, "sometimes_missing": 1.9})

    assert reloaded.expected_stock_return_20d == pytest.approx(
        high.expected_stock_return_20d
    )
    assert reloaded.expected_downside_20d == pytest.approx(high.expected_downside_20d)
    assert reloaded.probability_profitable_return == pytest.approx(
        high.probability_profitable_return
    )
    assert reloaded.profit_score == pytest.approx(high.profit_score)


def test_walk_forward_excludes_labels_not_mature_by_test_year() -> None:
    rows = [
        _row(1, year=2022, exit_date=date(2022, 6, 28)),
        _row(2, year=2023, exit_date=date(2023, 6, 28)),
        _row(3, year=2023, exit_date=date(2024, 1, 10)),
        _row(4, year=2024, exit_date=date(2024, 6, 28)),
    ]

    splits = annual_walk_forward_splits(
        rows,
        first_test_year=2024,
        min_train_rows=1,
        min_validation_rows=1,
        min_test_rows=1,
    )

    assert len(splits) == 1
    split = splits[0]
    assert split.test_year == 2024
    assert [row.event_id for row in split.train_rows] == ["event-2022-1"]
    assert [row.event_id for row in split.validation_rows] == ["event-2023-2"]
    assert [row.event_id for row in split.test_rows] == ["event-2024-4"]
