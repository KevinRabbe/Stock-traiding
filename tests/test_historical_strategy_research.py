from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from stock_trading.engine import (
    AllocationIntent,
    FeatureSnapshot,
    FixedAllocationPortfolioPolicy,
    Opportunity,
    PassThroughOpportunityRiskPolicy,
    PassThroughPortfolioRiskPolicy,
)
from stock_trading.research import (
    HistoricalCandidate,
    HistoricalOutcome,
    HistoricalStrategyBacktester,
    HistoricalYearResult,
    summarize_historical_years,
)


class _Strategy:
    strategy_id = "test-strategy"

    def evaluate(self, candidates, portfolio):
        del portfolio
        results = []
        for candidate in candidates:
            # Realized outcomes are intentionally not part of FeatureSnapshot.
            assert "realized_return" not in candidate.features
            results.append(
                Opportunity(
                    strategy_id=self.strategy_id,
                    candidate_id=candidate.candidate_id,
                    event_id=candidate.event_id,
                    company_id=candidate.company_id,
                    security_id=candidate.security_id,
                    execution_date=candidate.execution_date,
                    score=float(candidate.features["score"]),
                    expected_return=0.03,
                    expected_alpha=0.02,
                    expected_downside=0.01,
                    probability_positive=0.8,
                    horizon_sessions=int(candidate.features["horizon"]),
                )
            )
        return tuple(results)


class _UpsizingRisk:
    def filter(self, allocations, portfolio):
        del portfolio
        return tuple(
            replace(item, allocation_pct=item.allocation_pct + 0.01)
            for item in allocations
        )


def _candidate(
    candidate_id: str,
    company_id: str,
    entry: date,
    exit_day: date,
    *,
    score: float,
    stock_return: float,
    alpha: float = 0.0,
) -> HistoricalCandidate:
    snapshot = FeatureSnapshot(
        candidate_id=candidate_id,
        event_id=candidate_id,
        company_id=company_id,
        security_id=f"sec-{company_id}",
        decision_time=datetime.combine(
            entry - timedelta(days=1),
            datetime.min.time(),
            tzinfo=timezone.utc,
        ),
        execution_date=entry,
        features={"score": score, "horizon": 5},
    )
    return HistoricalCandidate(
        snapshot=snapshot,
        outcomes={
            5: HistoricalOutcome(
                horizon_sessions=5,
                exit_date=exit_day,
                stock_return=stock_return,
                alpha=alpha,
                downside=0.01,
            )
        },
    )


def _backtester():
    return HistoricalStrategyBacktester(
        starting_capital=10_000.0,
        round_trip_cost_bps=0.0,
    )


def _portfolio_policy():
    return FixedAllocationPortfolioPolicy(
        allocation_pct=0.02,
        max_open_positions=15,
        max_gross_exposure_pct=1.0,
    )


def test_historical_runner_closes_positions_before_same_day_entries() -> None:
    candidates = (
        _candidate(
            "a1",
            "company-a",
            date(2025, 1, 2),
            date(2025, 1, 4),
            score=0.99,
            stock_return=0.10,
        ),
        # Same company while a1 is active: shared portfolio policy rejects it.
        _candidate(
            "a2",
            "company-a",
            date(2025, 1, 3),
            date(2025, 1, 8),
            score=1.0,
            stock_return=0.50,
        ),
        # a1 exits before this batch is evaluated, so its gain is available to b1.
        _candidate(
            "b1",
            "company-b",
            date(2025, 1, 4),
            date(2025, 1, 9),
            score=0.98,
            stock_return=0.0,
        ),
    )

    result = _backtester().run(
        strategy=_Strategy(),
        candidates=candidates,
        opportunity_risk=PassThroughOpportunityRiskPolicy(),
        portfolio_policy=_portfolio_policy(),
        portfolio_risk=PassThroughPortfolioRiskPolicy(),
    )

    assert [trade.candidate_id for trade in result.trades] == ["a1", "b1"]
    assert result.trades[0].allocated_capital == pytest.approx(200.0)
    assert result.trades[0].pnl == pytest.approx(20.0)
    assert result.trades[1].allocated_capital == pytest.approx(200.4)
    assert result.ending_capital == pytest.approx(10_020.0)
    assert result.total_return == pytest.approx(0.002)


def test_historical_runner_rejects_portfolio_risk_upsizing() -> None:
    candidate = _candidate(
        "a1",
        "company-a",
        date(2025, 1, 2),
        date(2025, 1, 4),
        score=0.99,
        stock_return=0.10,
    )

    with pytest.raises(ValueError, match="may not upsize"):
        _backtester().run(
            strategy=_Strategy(),
            candidates=(candidate,),
            opportunity_risk=PassThroughOpportunityRiskPolicy(),
            portfolio_policy=_portfolio_policy(),
            portfolio_risk=_UpsizingRisk(),
        )


def test_fixed_allocation_ties_use_candidate_identity_ascending() -> None:
    candidates = tuple(
        _candidate(
            candidate_id,
            f"company-{candidate_id}",
            date(2025, 1, 2),
            date(2025, 1, 4),
            score=0.95,
            stock_return=0.0,
        )
        for candidate_id in ("z", "a", "m")
    )
    result = HistoricalStrategyBacktester(
        starting_capital=10_000.0,
        round_trip_cost_bps=0.0,
    ).run(
        strategy=_Strategy(),
        candidates=candidates,
        opportunity_risk=PassThroughOpportunityRiskPolicy(),
        portfolio_policy=FixedAllocationPortfolioPolicy(
            allocation_pct=0.02,
            max_open_positions=2,
            max_gross_exposure_pct=1.0,
        ),
        portfolio_risk=PassThroughPortfolioRiskPolicy(),
    )

    assert [trade.candidate_id for trade in result.trades] == ["a", "m"]


def test_historical_summary_produces_strategy_scorecard() -> None:
    positive = HistoricalStrategyBacktester(
        starting_capital=10_000.0,
        round_trip_cost_bps=0.0,
    ).run(
        strategy=_Strategy(),
        candidates=(
            _candidate(
                "a",
                "a",
                date(2024, 1, 2),
                date(2024, 1, 8),
                score=0.99,
                stock_return=0.10,
                alpha=0.05,
            ),
        ),
        opportunity_risk=PassThroughOpportunityRiskPolicy(),
        portfolio_policy=_portfolio_policy(),
        portfolio_risk=PassThroughPortfolioRiskPolicy(),
    )
    negative = HistoricalStrategyBacktester(
        starting_capital=10_000.0,
        round_trip_cost_bps=0.0,
    ).run(
        strategy=_Strategy(),
        candidates=(
            _candidate(
                "b",
                "b",
                date(2025, 1, 2),
                date(2025, 1, 8),
                score=0.99,
                stock_return=-0.05,
                alpha=-0.02,
            ),
        ),
        opportunity_risk=PassThroughOpportunityRiskPolicy(),
        portfolio_policy=_portfolio_policy(),
        portfolio_risk=PassThroughPortfolioRiskPolicy(),
    )

    summary = summarize_historical_years(
        (
            HistoricalYearResult(2024, positive),
            HistoricalYearResult(2025, negative),
        )
    )

    assert summary.scorecard.total_trades == 2
    assert summary.scorecard.profitable_year_rate == pytest.approx(0.5)
    assert summary.scorecard.profit_factor == pytest.approx(2.0)
    assert summary.scorecard.average_trade_alpha == pytest.approx(0.015)
    assert summary.best_year == 2024
