from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone

import pytest

from stock_trading.engine import (
    AllocationIntent,
    BasicOpportunityRiskPolicy,
    ExecutionReport,
    FeatureSnapshot,
    FixedAllocationPortfolioPolicy,
    HoldPositions,
    Opportunity,
    PortfolioPosition,
    PortfolioSnapshot,
    ProfitabilityGate,
    StrategyRecord,
    StrategyRegistry,
    StrategyScorecard,
    StrategyStage,
    TradingEngine,
)
from stock_trading.engine.policies import PassThroughPortfolioRiskPolicy


_NOW = datetime(2025, 1, 2, 12, 0, tzinfo=timezone.utc)


def _candidate(candidate_id: str, company_id: str, score: float, downside: float) -> FeatureSnapshot:
    return FeatureSnapshot(
        candidate_id=candidate_id,
        event_id=f"evt-{candidate_id}",
        company_id=company_id,
        security_id=f"sec-{company_id}",
        decision_time=_NOW,
        execution_date=date(2025, 1, 3),
        features={"score": score, "downside": downside},
    )


class _Source:
    def __init__(self, candidates):
        self._candidates = tuple(candidates)

    def candidates(self, as_of):
        assert as_of == _NOW
        return self._candidates


class _Strategy:
    def __init__(self, strategy_id: str):
        self._strategy_id = strategy_id

    @property
    def strategy_id(self) -> str:
        return self._strategy_id

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
                score=float(item.features["score"]),
                expected_return=0.03,
                expected_alpha=0.02,
                expected_downside=float(item.features["downside"]),
                probability_positive=0.7,
                horizon_sessions=20,
            )
            for item in candidates
        )


class _State:
    def __init__(self, snapshot):
        self._snapshot = snapshot

    def snapshot(self, as_of):
        assert as_of == self._snapshot.as_of
        return self._snapshot


class _Broker:
    def __init__(self):
        self.orders = ()

    def execute(self, orders):
        self.orders = orders
        return tuple(
            ExecutionReport(order_id=item.order_id, accepted=True, executed_at=_NOW)
            for item in orders
        )


class _Observer:
    def __init__(self):
        self.result = None

    def record(self, result):
        self.result = result


class _UpsizingPortfolioRisk:
    def filter(self, allocations, portfolio):
        del portfolio
        return tuple(
            replace(item, allocation_pct=item.allocation_pct + 0.01)
            for item in allocations
        )


def _registry(strategy: _Strategy) -> StrategyRegistry:
    registry = StrategyRegistry()
    registry.register(
        strategy,
        StrategyRecord(strategy_id=strategy.strategy_id, stage=StrategyStage.PAPER),
    )
    registry.set_champion(strategy.strategy_id)
    return registry


def test_engine_keeps_strategy_risk_portfolio_and_execution_separate() -> None:
    candidates = (
        _candidate("a", "company-a", 0.99, 0.01),
        _candidate("b", "company-b", 0.98, 0.20),  # rejected before allocation
        _candidate("c", "company-c", 0.97, 0.02),
        _candidate("d", "company-d", 0.96, 0.02),
    )
    snapshot = PortfolioSnapshot(
        as_of=_NOW,
        equity=10_000.0,
        cash=9_800.0,
        gross_exposure_pct=0.02,
        positions=(
            PortfolioPosition(
                position_id="pos-a",
                strategy_id="v5",
                company_id="company-a",
                security_id="sec-company-a",
                allocation_pct=0.02,
                opened_on=date(2024, 12, 30),
            ),
        ),
    )
    strategy = _Strategy("v5")
    broker = _Broker()
    observer = _Observer()
    engine = TradingEngine(
        candidate_source=_Source(candidates),
        strategy_provider=_registry(strategy),
        opportunity_risk=BasicOpportunityRiskPolicy(max_expected_downside=0.06),
        portfolio_policy=FixedAllocationPortfolioPolicy(
            allocation_pct=0.02,
            max_open_positions=3,
            max_gross_exposure_pct=0.30,
        ),
        portfolio_risk=PassThroughPortfolioRiskPolicy(),
        position_manager=HoldPositions(),
        state_provider=_State(snapshot),
        broker=broker,
        observer=observer,
    )

    result = engine.run_cycle(_NOW)

    assert result.strategy_id == "v5"
    assert result.candidate_count == 4
    assert result.opportunity_count == 4
    assert result.eligible_opportunity_count == 3
    assert result.allocation_count == 2
    assert [item.company_id for item in result.entry_orders] == ["company-c", "company-d"]
    assert all(item.allocation_pct == pytest.approx(0.02) for item in result.entry_orders)
    assert broker.orders == result.entry_orders
    assert observer.result == result


def test_portfolio_risk_boundary_cannot_upsize_strategy_allocation() -> None:
    candidate = _candidate("a", "company-a", 0.99, 0.01)
    snapshot = PortfolioSnapshot(
        as_of=_NOW,
        equity=10_000.0,
        cash=10_000.0,
        gross_exposure_pct=0.0,
    )
    strategy = _Strategy("v5")
    engine = TradingEngine(
        candidate_source=_Source((candidate,)),
        strategy_provider=_registry(strategy),
        opportunity_risk=BasicOpportunityRiskPolicy(),
        portfolio_policy=FixedAllocationPortfolioPolicy(),
        portfolio_risk=_UpsizingPortfolioRisk(),
        position_manager=HoldPositions(),
        state_provider=_State(snapshot),
        broker=_Broker(),
    )

    with pytest.raises(ValueError, match="not upsize"):
        engine.run_cycle(_NOW)


def test_registry_recommends_profitable_challenger_without_auto_promoting_it() -> None:
    champion = _Strategy("v5")
    challenger = _Strategy("future-model")
    bad = _Strategy("bad-model")
    registry = StrategyRegistry()
    registry.register(
        champion,
        StrategyRecord(
            strategy_id="v5",
            stage=StrategyStage.PAPER,
            selection_score=1.0,
            scorecard=StrategyScorecard(
                compounded_return=0.06,
                profit_factor=1.6,
                worst_realized_drawdown=0.02,
                total_trades=190,
                profitable_year_rate=0.54,
            ),
        ),
    )
    registry.register(
        challenger,
        StrategyRecord(
            strategy_id="future-model",
            stage=StrategyStage.SHADOW,
            selection_score=2.0,
            scorecard=StrategyScorecard(
                compounded_return=0.08,
                profit_factor=1.7,
                worst_realized_drawdown=0.025,
                total_trades=220,
                profitable_year_rate=0.62,
            ),
        ),
    )
    registry.register(
        bad,
        StrategyRecord(
            strategy_id="bad-model",
            stage=StrategyStage.DEVELOPMENT,
            selection_score=99.0,
            scorecard=StrategyScorecard(
                compounded_return=-0.02,
                profit_factor=0.8,
                worst_realized_drawdown=0.05,
                total_trades=500,
                profitable_year_rate=0.30,
            ),
        ),
    )
    registry.set_champion("v5")

    recommendation = registry.recommend_champion(ProfitabilityGate(min_trades=100))

    assert recommendation is not None
    assert recommendation.strategy_id == "future-model"
    assert registry.active().strategy_id == "v5"
