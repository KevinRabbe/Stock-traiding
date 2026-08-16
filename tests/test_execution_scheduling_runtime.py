from __future__ import annotations

from datetime import date, datetime, timezone

from stock_trading.engine import (
    BasicOpportunityRiskPolicy,
    ExecutionReport,
    ExecutionStatus,
    FeatureSnapshot,
    FixedAllocationPortfolioPolicy,
    HoldPositions,
    Opportunity,
    PortfolioSnapshot,
    StrategyRecord,
    StrategyRegistry,
    StrategyStage,
    TradingEngine,
)
from stock_trading.engine.policies import PassThroughPortfolioRiskPolicy


_NOW = datetime(2025, 1, 2, 18, tzinfo=timezone.utc)


class _Broker:
    def __init__(self):
        self.settled = False
        self.orders = ()

    def settle(self, as_of):
        self.settled = True
        return (
            ExecutionReport(
                order_id="old-order",
                accepted=True,
                executed_at=as_of,
                fill_price=100.0,
                status=ExecutionStatus.FILLED,
            ),
        )

    def execute(self, orders):
        self.orders = orders
        return tuple(
            ExecutionReport(
                order_id=order.order_id,
                accepted=True,
                executed_at=_NOW,
                status=(
                    ExecutionStatus.QUEUED
                    if order.execute_on and order.execute_on > _NOW.date()
                    else ExecutionStatus.FILLED
                ),
            )
            for order in orders
        )


class _State:
    def __init__(self, broker):
        self.broker = broker

    def snapshot(self, as_of):
        assert self.broker.settled, "queued settlements must happen before snapshot"
        return PortfolioSnapshot(
            as_of=as_of,
            equity=10_000.0,
            cash=10_000.0,
            gross_exposure_pct=0.0,
        )


class _Source:
    def candidates(self, as_of):
        return (
            FeatureSnapshot(
                candidate_id="a",
                event_id="a",
                company_id="company-a",
                security_id="security-a",
                decision_time=as_of,
                execution_date=date(2025, 1, 3),
                features={},
            ),
        )


class _Strategy:
    strategy_id = "v5"

    def evaluate(self, candidates, portfolio):
        del portfolio
        item = candidates[0]
        return (
            Opportunity(
                strategy_id=self.strategy_id,
                candidate_id=item.candidate_id,
                event_id=item.event_id,
                company_id=item.company_id,
                security_id=item.security_id,
                execution_date=item.execution_date,
                score=0.99,
                expected_return=0.03,
                expected_alpha=0.02,
                expected_downside=0.01,
                probability_positive=0.8,
                horizon_sessions=20,
            ),
        )


def _registry():
    registry = StrategyRegistry()
    registry.register(
        _Strategy(),
        StrategyRecord(strategy_id="v5", stage=StrategyStage.PAPER),
    )
    registry.set_champion("v5")
    return registry


def test_engine_settles_old_orders_before_snapshot_and_schedules_future_entry() -> None:
    broker = _Broker()
    engine = TradingEngine(
        candidate_source=_Source(),
        strategy_provider=_registry(),
        opportunity_risk=BasicOpportunityRiskPolicy(),
        portfolio_policy=FixedAllocationPortfolioPolicy(),
        portfolio_risk=PassThroughPortfolioRiskPolicy(),
        position_manager=HoldPositions(),
        state_provider=_State(broker),
        broker=broker,
    )

    result = engine.run_cycle(_NOW)

    assert result.settlements[0].order_id == "old-order"
    assert len(result.entry_orders) == 1
    assert result.entry_orders[0].execute_on == date(2025, 1, 3)
    assert result.executions[0].status is ExecutionStatus.QUEUED
