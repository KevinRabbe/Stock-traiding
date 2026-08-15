from datetime import date, datetime, timezone

import pytest

from stock_trading.backtest import (
    BacktestConfig,
    FixedAllocationBacktester,
    FixedAllocationTrancheBacktester,
    ScoredCandidate,
)
from stock_trading.ml import OpportunityPrediction, TrainingRow


def _candidate(index: int, *, company_id: str = "cmp-a") -> ScoredCandidate:
    execution = date(2024, 1, 2 + index)
    row = TrainingRow(
        event_id=f"event-{index}",
        company_id=company_id,
        decision_time=datetime(2024, 1, 1 + index, 20, 0, tzinfo=timezone.utc),
        execution_date=execution,
        exit_date_20d=date(2024, 1, 31),
        features={"signal": 1.0},
        stock_return_20d=0.05,
        benchmark_return_20d=0.01,
        alpha_20d=0.04,
        downside_20d=0.02,
        mfe_20d=0.08,
        positive_alpha_20d=1,
    )
    return ScoredCandidate(
        row=row,
        prediction=OpportunityPrediction(
            expected_alpha_20d=0.05,
            expected_downside_20d=0.02,
            probability_positive_alpha=0.8,
            opportunity_score=0.04,
        ),
    )


def test_tranche_backtester_allows_second_overlapping_company_signal() -> None:
    config = BacktestConfig(
        allocation_pct=0.02,
        max_open_positions=15,
        min_expected_alpha=0.0,
        min_probability_positive=0.0,
        max_expected_downside=0.06,
        round_trip_cost_bps=20.0,
    )
    candidates = (_candidate(0), _candidate(1), _candidate(2))

    single = FixedAllocationBacktester(config).run(candidates)
    tranche = FixedAllocationTrancheBacktester(
        config,
        max_company_tranches=2,
    ).run(candidates)

    assert len(single.trades) == 1
    assert single.rejected_duplicate_company == 2
    assert len(tranche.trades) == 2
    assert tranche.rejected_duplicate_company == 1
    assert tranche.rejected_capacity == 0
    assert tranche.total_return > single.total_return


def test_tranche_backtester_keeps_total_slot_cap() -> None:
    config = BacktestConfig(
        allocation_pct=0.02,
        max_open_positions=2,
        min_expected_alpha=0.0,
        min_probability_positive=0.0,
        max_expected_downside=0.06,
    )
    candidates = (
        _candidate(0, company_id="cmp-a"),
        _candidate(1, company_id="cmp-a"),
        _candidate(2, company_id="cmp-b"),
    )

    result = FixedAllocationTrancheBacktester(
        config,
        max_company_tranches=2,
    ).run(candidates)

    assert len(result.trades) == 2
    assert result.rejected_capacity == 1
    assert result.rejected_duplicate_company == 0
    assert result.ending_capital == pytest.approx(
        config.starting_capital * (1.0 + 2 * config.allocation_pct * (0.05 - 0.002))
    )
