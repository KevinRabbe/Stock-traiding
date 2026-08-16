from datetime import date, datetime, timezone

import pytest

from stock_trading.backtest.portfolio import ScoredCandidate
from stock_trading.experiments.lightgbm_repeat_signal_diagnostics import (
    _candidate_summary,
    _classify_overlap_signals,
    _ordinal_summaries,
    _trace_acceptance,
)
from stock_trading.ml import OpportunityPrediction, TrainingRow


def _candidate(
    event_id: str,
    company_id: str,
    entry: date,
    exit_: date,
    *,
    stock_return: float,
    alpha: float,
    score: float,
    downside: float = 0.02,
) -> ScoredCandidate:
    row = TrainingRow(
        event_id=event_id,
        company_id=company_id,
        decision_time=datetime.combine(entry, datetime.min.time(), tzinfo=timezone.utc),
        execution_date=entry,
        exit_date_20d=exit_,
        features={"feature": 1.0},
        stock_return_20d=stock_return,
        benchmark_return_20d=stock_return - alpha,
        alpha_20d=alpha,
        downside_20d=downside,
        mfe_20d=max(stock_return, 0.0),
        positive_alpha_20d=int(alpha > 0.02),
        trigger_event_ids=(event_id,),
    )
    return ScoredCandidate(
        row=row,
        prediction=OpportunityPrediction(
            expected_alpha_20d=stock_return,
            expected_downside_20d=downside,
            probability_positive_alpha=0.7,
            opportunity_score=score,
        ),
    )


def test_overlap_signal_classification_tracks_second_and_third_signal_quality() -> None:
    candidates = [
        _candidate(
            "a1",
            "A",
            date(2020, 1, 2),
            date(2020, 1, 30),
            stock_return=0.10,
            alpha=0.06,
            score=0.9,
        ),
        _candidate(
            "a2",
            "A",
            date(2020, 1, 9),
            date(2020, 2, 6),
            stock_return=0.02,
            alpha=0.01,
            score=0.8,
        ),
        _candidate(
            "a3",
            "A",
            date(2020, 1, 16),
            date(2020, 2, 13),
            stock_return=-0.03,
            alpha=-0.02,
            score=0.7,
        ),
        _candidate(
            "a4",
            "A",
            date(2020, 3, 2),
            date(2020, 3, 30),
            stock_return=0.04,
            alpha=0.03,
            score=0.6,
        ),
    ]

    observations = _classify_overlap_signals(candidates)

    assert [item.overlap_ordinal for item in observations] == [1, 2, 3, 1]
    assert observations[1].days_since_previous_signal == 7
    assert observations[1].stock_return_delta_vs_previous == pytest.approx(-0.08)
    assert observations[1].alpha_delta_vs_previous == pytest.approx(-0.05)
    assert observations[2].stock_return_delta_vs_previous == pytest.approx(-0.05)
    assert observations[3].days_since_previous_signal is None

    summary = _ordinal_summaries(observations)
    assert summary["first_active_signal"]["count"] == 2
    assert summary["second_overlapping_signal"]["count"] == 1
    assert summary["third_or_later_overlapping_signal"]["count"] == 1
    assert summary["second_overlapping_signal"][
        "average_incremental_stock_return_vs_previous_signal"
    ] == pytest.approx(-0.08)


def test_trace_exposes_capacity_displacement_created_by_repeat_tranche() -> None:
    # Day 1 opens A. On day 2, A's repeat outranks B. The one-position policy
    # rejects A2 and accepts B, while the two-tranche policy accepts A2 first and
    # then has no portfolio slot left for B.
    candidates = [
        _candidate(
            "a1",
            "A",
            date(2020, 1, 2),
            date(2020, 1, 30),
            stock_return=0.03,
            alpha=0.02,
            score=0.9,
        ),
        _candidate(
            "a2",
            "A",
            date(2020, 1, 3),
            date(2020, 1, 31),
            stock_return=-0.05,
            alpha=-0.04,
            score=0.95,
        ),
        _candidate(
            "b1",
            "B",
            date(2020, 1, 3),
            date(2020, 1, 31),
            stock_return=0.08,
            alpha=0.06,
            score=0.8,
        ),
    ]

    single = _trace_acceptance(
        candidates,
        max_open_positions=2,
        max_company_tranches=1,
    )
    tranche = _trace_acceptance(
        candidates,
        max_open_positions=2,
        max_company_tranches=2,
    )

    assert single.accepted_event_ids == ("a1", "b1")
    assert single.rejection_reason_by_event_id["a2"] == "company_limit"
    assert tranche.accepted_event_ids == ("a1", "a2")
    assert tranche.rejection_reason_by_event_id["b1"] == "capacity"


def test_candidate_summary_keeps_outcome_and_model_score_separate() -> None:
    candidates = [
        _candidate(
            "a1",
            "A",
            date(2020, 1, 2),
            date(2020, 1, 30),
            stock_return=0.04,
            alpha=0.03,
            score=0.8,
        ),
        _candidate(
            "b1",
            "B",
            date(2020, 1, 3),
            date(2020, 1, 31),
            stock_return=-0.01,
            alpha=-0.02,
            score=0.6,
        ),
    ]

    summary = _candidate_summary(candidates)

    assert summary["count"] == 2
    assert summary["average_stock_return_20d"] == pytest.approx(0.015)
    assert summary["average_alpha_20d"] == pytest.approx(0.005)
    assert summary["average_opportunity_score"] == pytest.approx(0.7)
    assert summary["profitable_after_cost_rate"] == pytest.approx(0.5)
