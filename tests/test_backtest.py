from datetime import date, datetime, timezone

import pytest

from stock_trading.backtest import BacktestConfig, FixedAllocationBacktester, ScoredCandidate
from stock_trading.ml import OpportunityPrediction, TrainingRow


def _row(
    event_id: str,
    company_id: str,
    *,
    entry: date,
    exit_: date,
    stock_return: float,
    alpha: float,
    downside: float = 0.03,
) -> TrainingRow:
    return TrainingRow(
        event_id=event_id,
        company_id=company_id,
        decision_time=datetime(entry.year, entry.month, entry.day, tzinfo=timezone.utc),
        execution_date=entry,
        exit_date_20d=exit_,
        features={"signal": 1.0},
        stock_return_20d=stock_return,
        benchmark_return_20d=stock_return - alpha,
        alpha_20d=alpha,
        downside_20d=downside,
        mfe_20d=max(0.0, stock_return),
        positive_alpha_20d=int(alpha >= 0.02),
    )


def _candidate(
    row: TrainingRow,
    *,
    expected_alpha: float,
    expected_downside: float,
    probability: float,
    score: float,
) -> ScoredCandidate:
    return ScoredCandidate(
        row=row,
        prediction=OpportunityPrediction(
            expected_alpha_20d=expected_alpha,
            expected_downside_20d=expected_downside,
            probability_positive_alpha=probability,
            opportunity_score=score,
        ),
    )


def test_fixed_allocation_backtest_ranks_filters_and_applies_costs() -> None:
    entry = date(2026, 8, 10)
    exit_ = date(2026, 9, 4)
    best = _candidate(
        _row("best", "cmp_a", entry=entry, exit_=exit_, stock_return=0.10, alpha=0.08),
        expected_alpha=0.07,
        expected_downside=0.03,
        probability=0.85,
        score=0.05,
    )
    duplicate = _candidate(
        _row("duplicate", "cmp_a", entry=entry, exit_=exit_, stock_return=0.20, alpha=0.18),
        expected_alpha=0.06,
        expected_downside=0.03,
        probability=0.80,
        score=0.04,
    )
    filtered = _candidate(
        _row("filtered", "cmp_b", entry=entry, exit_=exit_, stock_return=-0.20, alpha=-0.15),
        expected_alpha=0.01,
        expected_downside=0.02,
        probability=0.70,
        score=0.01,
    )

    backtester = FixedAllocationBacktester(
        BacktestConfig(
            starting_capital=10_000,
            allocation_pct=0.02,
            max_open_positions=2,
            min_expected_alpha=0.03,
            min_probability_positive=0.60,
            max_expected_downside=0.06,
            round_trip_cost_bps=20,
        )
    )
    result = backtester.run([duplicate, filtered, best])

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.event_id == "best"
    assert trade.allocated_capital == pytest.approx(200.0)
    assert trade.net_return == pytest.approx(0.098)
    assert trade.pnl == pytest.approx(19.6)
    assert result.ending_capital == pytest.approx(10_019.6)
    assert result.rejected_by_signal == 1
    assert result.rejected_duplicate_company == 1
    assert result.rejected_capacity == 0
    assert result.win_rate == 1.0


def test_capacity_keeps_highest_ranked_candidate() -> None:
    entry = date(2026, 8, 10)
    exit_ = date(2026, 9, 4)
    high = _candidate(
        _row("high", "cmp_a", entry=entry, exit_=exit_, stock_return=0.05, alpha=0.04),
        expected_alpha=0.05,
        expected_downside=0.02,
        probability=0.80,
        score=0.04,
    )
    low = _candidate(
        _row("low", "cmp_b", entry=entry, exit_=exit_, stock_return=0.30, alpha=0.29),
        expected_alpha=0.04,
        expected_downside=0.02,
        probability=0.75,
        score=0.02,
    )

    result = FixedAllocationBacktester(
        BacktestConfig(max_open_positions=1)
    ).run([low, high])

    assert [trade.event_id for trade in result.trades] == ["high"]
    assert result.rejected_capacity == 1
