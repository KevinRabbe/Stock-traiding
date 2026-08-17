from __future__ import annotations

from datetime import date, datetime, timezone

from stock_trading.engine import (
    FeatureSnapshot,
    FixedAllocationPortfolioPolicy,
    Opportunity,
    PassThroughOpportunityRiskPolicy,
    PassThroughPortfolioRiskPolicy,
)
from stock_trading.research import HistoricalCandidate, HistoricalOutcome
from stock_trading.research.execution_realism import (
    ExecutionRealisticHistoricalBacktester,
    HistoricalExecutionLiquidity,
    MarketQualityExclusion,
    target_overlaps_exclusion,
    trailing_adv_supports,
)


class _SingleOpportunityStrategy:
    strategy_id = "test-strategy"

    def evaluate(self, candidates, portfolio):
        del portfolio
        return tuple(
            Opportunity(
                strategy_id=self.strategy_id,
                candidate_id=item.candidate_id,
                event_id=item.event_id,
                company_id=item.company_id,
                security_id=item.security_id,
                execution_date=item.execution_date,
                score=0.99,
                expected_return=0.10,
                expected_alpha=0.08,
                expected_downside=0.02,
                probability_positive=0.75,
                horizon_sessions=20,
            )
            for item in candidates
        )


def _candidate() -> HistoricalCandidate:
    snapshot = FeatureSnapshot(
        candidate_id="candidate-a",
        event_id="candidate-a",
        company_id="company-a",
        security_id="security-a",
        decision_time=datetime(2020, 1, 1, 12, tzinfo=timezone.utc),
        execution_date=date(2020, 1, 2),
        features={"market.avg_dollar_volume_20d": 100_000.0},
    )
    return HistoricalCandidate(
        snapshot=snapshot,
        outcomes={
            20: HistoricalOutcome(
                horizon_sessions=20,
                exit_date=date(2020, 2, 3),
                stock_return=0.10,
                alpha=0.08,
                downside=0.02,
            )
        },
    )


def _run(entry_volume: float):
    candidate = _candidate()
    backtester = ExecutionRealisticHistoricalBacktester(
        starting_capital=10_000.0,
        round_trip_cost_bps=20.0,
    )
    return backtester.run(
        strategy=_SingleOpportunityStrategy(),
        candidates=(candidate,),
        opportunity_risk=PassThroughOpportunityRiskPolicy(),
        portfolio_policy=FixedAllocationPortfolioPolicy(
            allocation_pct=0.02,
            max_open_positions=15,
            max_gross_exposure_pct=1.0,
        ),
        portfolio_risk=PassThroughPortfolioRiskPolicy(),
        entry_liquidity={
            candidate.snapshot.candidate_id: HistoricalExecutionLiquidity(
                candidate_id=candidate.snapshot.candidate_id,
                entry_price=10.0,
                entry_volume=entry_volume,
            )
        },
        max_entry_day_participation_pct=0.01,
    )


def test_trailing_adv_requires_full_reference_order_capacity() -> None:
    assert trailing_adv_supports(
        {"market.avg_dollar_volume_20d": 25_000.0},
        required_capital=200.0,
        max_participation_pct=0.01,
    )
    assert not trailing_adv_supports(
        {"market.avg_dollar_volume_20d": 10_000.0},
        required_capital=200.0,
        max_participation_pct=0.01,
    )
    assert not trailing_adv_supports(
        {},
        required_capital=200.0,
        max_participation_pct=0.01,
    )


def test_market_quality_exclusion_rejects_only_spanning_target() -> None:
    exclusions = (
        MarketQualityExclusion(
            security_id="security-mtst",
            ticker="MTST",
            start_date=date(2015, 10, 8),
            end_date=date(2015, 10, 9),
            reason="verified reverse split discontinuity",
        ),
    )
    assert target_overlaps_exclusion(
        "security-mtst",
        date(2015, 8, 20),
        date(2015, 11, 10),
        exclusions,
    )
    assert not target_overlaps_exclusion(
        "security-mtst",
        date(2015, 10, 12),
        date(2015, 11, 10),
        exclusions,
    )
    assert not target_overlaps_exclusion(
        "other-security",
        date(2015, 8, 20),
        date(2015, 11, 10),
        exclusions,
    )


def test_execution_day_liquidity_rejects_unfillable_order() -> None:
    backtest, diagnostics = _run(entry_volume=100.0)
    assert backtest.trades == ()
    assert diagnostics.rejected_entry_liquidity == 1


def test_execution_day_liquidity_accepts_fillable_order() -> None:
    backtest, diagnostics = _run(entry_volume=5_000.0)
    assert len(backtest.trades) == 1
    assert diagnostics.rejected_entry_liquidity == 0
    assert backtest.trades[0].allocated_capital == 200.0


def test_execution_day_liquidity_is_not_exposed_to_strategy_snapshot() -> None:
    candidate = _candidate()
    assert "entry_volume" not in candidate.snapshot.features
    assert "entry_dollar_volume" not in candidate.snapshot.features
    assert set(candidate.snapshot.features) == {"market.avg_dollar_volume_20d"}
