from dataclasses import dataclass
from datetime import date

from stock_trading.backtest.portfolio import BacktestResult, FixedAllocationBacktester

from .dataset import TrainingRow
from .lightgbm_models import LightGbmTrainer


@dataclass(frozen=True, slots=True)
class WalkForwardSplit:
    test_year: int
    train_rows: tuple[TrainingRow, ...]
    validation_rows: tuple[TrainingRow, ...]
    test_rows: tuple[TrainingRow, ...]


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    test_year: int
    train_count: int
    validation_count: int
    test_count: int
    backtest: BacktestResult


def annual_walk_forward_splits(
    rows: tuple[TrainingRow, ...] | list[TrainingRow],
    *,
    first_test_year: int | None = None,
    min_train_rows: int = 1,
    min_validation_rows: int = 1,
    min_test_rows: int = 1,
) -> tuple[WalkForwardSplit, ...]:
    if min_train_rows <= 0 or min_validation_rows <= 0 or min_test_rows <= 0:
        raise ValueError("minimum row counts must be > 0")

    rows = tuple(sorted(rows, key=lambda row: (row.decision_time, row.event_id)))
    years = sorted({row.decision_time.year for row in rows})
    if first_test_year is not None:
        years = [year for year in years if year >= first_test_year]

    splits: list[WalkForwardSplit] = []
    for test_year in years:
        validation_year = test_year - 1
        test_start = date(test_year, 1, 1)

        # Training happens immediately before the test year. Every training and
        # validation target must therefore have matured before Jan 1 test_year.
        eligible = [row for row in rows if row.exit_date_20d < test_start]
        train_rows = tuple(
            row for row in eligible if row.decision_time.year < validation_year
        )
        validation_rows = tuple(
            row for row in eligible if row.decision_time.year == validation_year
        )
        test_rows = tuple(row for row in rows if row.decision_time.year == test_year)

        if (
            len(train_rows) < min_train_rows
            or len(validation_rows) < min_validation_rows
            or len(test_rows) < min_test_rows
        ):
            continue
        splits.append(
            WalkForwardSplit(
                test_year=test_year,
                train_rows=train_rows,
                validation_rows=validation_rows,
                test_rows=test_rows,
            )
        )
    return tuple(splits)


def run_annual_walk_forward(
    rows: tuple[TrainingRow, ...] | list[TrainingRow],
    *,
    trainer: LightGbmTrainer,
    backtester: FixedAllocationBacktester,
    first_test_year: int | None = None,
    min_train_rows: int = 100,
    min_validation_rows: int = 20,
    min_test_rows: int = 1,
    positive_alpha_threshold: float = 0.02,
) -> tuple[WalkForwardResult, ...]:
    results: list[WalkForwardResult] = []
    for split in annual_walk_forward_splits(
        rows,
        first_test_year=first_test_year,
        min_train_rows=min_train_rows,
        min_validation_rows=min_validation_rows,
        min_test_rows=min_test_rows,
    ):
        model = trainer.train(
            split.train_rows,
            split.validation_rows,
            positive_alpha_threshold=positive_alpha_threshold,
        )
        scored = backtester.score_rows(split.test_rows, model)
        results.append(
            WalkForwardResult(
                test_year=split.test_year,
                train_count=len(split.train_rows),
                validation_count=len(split.validation_rows),
                test_count=len(split.test_rows),
                backtest=backtester.run(scored),
            )
        )
    return tuple(results)
