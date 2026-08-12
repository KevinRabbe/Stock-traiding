from datetime import date, datetime, timezone

import pytest

from stock_trading.backtest import (
    BacktestResult,
    ScoredCandidate,
    TradeRecord,
    evaluate_score_buckets,
    profit_without_best_trades,
)
from stock_trading.ml import OpportunityPrediction, TrainingRow


def _candidate(index: int, score: float, alpha: float) -> ScoredCandidate:
    row = TrainingRow(
        event_id=f"event-{index}",
        company_id=f"cmp-{index}",
        decision_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        execution_date=date(2026, 1, 2),
        exit_date_20d=date(2026, 1, 30),
        features={"signal": score},
        stock_return_20d=alpha + 0.01,
        benchmark_return_20d=0.01,
        alpha_20d=alpha,
        downside_20d=0.02,
        mfe_20d=max(0.0, alpha + 0.03),
        positive_alpha_20d=int(alpha >= 0.02),
    )
    return ScoredCandidate(
        row=row,
        prediction=OpportunityPrediction(
            expected_alpha_20d=alpha,
            expected_downside_20d=0.02,
            probability_positive_alpha=0.8,
            opportunity_score=score,
        ),
    )


def test_score_buckets_measure_the_top_ranked_candidates() -> None:
    candidates = [
        _candidate(index, score=100 - index, alpha=(100 - index) / 1000)
        for index in range(100)
    ]

    buckets = evaluate_score_buckets(candidates)
    by_fraction = {bucket.top_fraction: bucket for bucket in buckets}

    assert by_fraction[0.10].candidate_count == 10
    assert by_fraction[0.01].candidate_count == 1
    assert by_fraction[0.01].average_alpha_20d > by_fraction[0.10].average_alpha_20d
    assert by_fraction[0.10].average_alpha_20d > by_fraction[0.20].average_alpha_20d


def test_profit_without_best_trades_exposes_outlier_dependence() -> None:
    trades = (
        TradeRecord(
            event_id="winner",
            company_id="cmp-a",
            entry_date=date(2026, 1, 2),
            exit_date=date(2026, 1, 30),
            allocated_capital=100,
            gross_return=1.0,
            net_return=1.0,
            alpha_20d=0.9,
            max_adverse_excursion=-0.02,
            pnl=100,
            opportunity_score=1.0,
        ),
        TradeRecord(
            event_id="normal",
            company_id="cmp-b",
            entry_date=date(2026, 2, 2),
            exit_date=date(2026, 2, 27),
            allocated_capital=100,
            gross_return=0.10,
            net_return=0.10,
            alpha_20d=0.08,
            max_adverse_excursion=-0.03,
            pnl=10,
            opportunity_score=0.5,
        ),
    )
    result = BacktestResult(
        starting_capital=1000,
        ending_capital=1110,
        net_profit=110,
        total_return=0.11,
        trades=trades,
        win_rate=1.0,
        profit_factor=float("inf"),
        average_trade_return=0.55,
        realized_max_drawdown=0.0,
        worst_trade_mae=-0.03,
        rejected_by_signal=0,
        rejected_duplicate_company=0,
        rejected_capacity=0,
    )

    assert profit_without_best_trades(result, 0) == pytest.approx(110)
    assert profit_without_best_trades(result, 1) == pytest.approx(10)
    assert profit_without_best_trades(result, 2) == pytest.approx(0)
