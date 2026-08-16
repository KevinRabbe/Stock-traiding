from __future__ import annotations

from datetime import date, datetime, timezone
from hashlib import sha256

import pytest

from stock_trading.engine import (
    BasicOpportunityRiskPolicy,
    ExecutionReport,
    FeatureSnapshot,
    FixedAllocationPortfolioPolicy,
    Opportunity,
    OrderIntent,
    OrderSide,
    PortfolioPosition,
    PortfolioSnapshot,
    StrategyRecord,
    StrategyRegistry,
    StrategyStage,
    TradingEngine,
)
from stock_trading.engine.policies import PassThroughPortfolioRiskPolicy


_NOW = datetime(2025, 1, 2, 12, tzinfo=timezone.utc)


class _Source:
    def candidates(self, as_of):
        assert as_of == _NOW
        return (
            FeatureSnapshot(
                candidate_id="repeat-a",
                event_id="repeat-a",
                company_id="company-a",
                security_id="security-a",
                decision_time=_NOW,
                execution_date=date(2025, 1, 3),
                features={"score": 0.99},
            ),
        )


class _Strategy:
    strategy_id = "v5"

    def evaluate(self, candidates, portfolio):
        del portfolio
        candidate = candidates[0]
        return (
            Opportunity(
                strategy_id=self.strategy_id,
                candidate_id=candidate.candidate_id,
                event_id=candidate.event_id,
                company_id=candidate.company_id,
                security_id=candidate.security_id,
                execution_date=candidate.execution_date,
                score=0.99,
                expected_return=0.03,
                expected_alpha=0.01,
                expected_downside=0.02,
                probability_positive=0.7,
                horizon_sessions=20,
            ),
        )


class _State:
    def snapshot(self, as_of):
        return PortfolioSnapshot(
            as_of=as_of,
            equity=10_000.0,
            cash=9_800.0,
            gross_exposure_pct=0.02,
            positions=(
                PortfolioPosition(
                    position_id="position-a",
                    strategy_id="v5",
                    company_id="company-a",
                    security_id="security-a",
                    allocation_pct=0.02,
                    opened_on=date(2024, 12, 20),
                ),
            ),
        )


class _ReactiveExitManager:
    def orders(self, portfolio, as_of, candidates, opportunities):
        assert candidates[0].company_id == "company-a"
        assert opportunities[0].company_id == "company-a"
        position = portfolio.positions[0]
        digest = sha256(b"position-a-exit").hexdigest()[:20]
        return (
            OrderIntent(
                order_id=f"ord_{digest}",
                strategy_id=position.strategy_id,
                company_id=position.company_id,
                security_id=position.security_id,
                side=OrderSide.SELL,
                allocation_pct=position.allocation_pct,
                created_at=as_of,
                candidate_id=opportunities[0].candidate_id,
                event_id=opportunities[0].event_id,
                reason="repeat_signal_thesis_review_exit",
            ),
        )


class _BadBuyManager:
    def orders(self, portfolio, as_of, candidates, opportunities):
        del candidates, opportunities
        position = portfolio.positions[0]
        return (
            OrderIntent(
                order_id="bad-buy",
                strategy_id=position.strategy_id,
                company_id=position.company_id,
                security_id=position.security_id,
                side=OrderSide.BUY,
                allocation_pct=0.01,
                created_at=as_of,
                reason="hidden_upsize",
            ),
        )


class _Broker:
    def __init__(self):
        self.orders = ()

    def execute(self, orders):
        self.orders = orders
        return tuple(
            ExecutionReport(order_id=order.order_id, accepted=True, executed_at=_NOW)
            for order in orders
        )


def _registry():
    registry = StrategyRegistry()
    registry.register(
        _Strategy(),
        StrategyRecord(strategy_id="v5", stage=StrategyStage.PAPER),
    )
    registry.set_champion("v5")
    return registry


def _engine(manager, broker):
    return TradingEngine(
        candidate_source=_Source(),
        strategy_provider=_registry(),
        opportunity_risk=BasicOpportunityRiskPolicy(),
        portfolio_policy=FixedAllocationPortfolioPolicy(),
        portfolio_risk=PassThroughPortfolioRiskPolicy(),
        position_manager=manager,
        state_provider=_State(),
        broker=broker,
    )


def test_position_manager_can_react_to_repeat_signal_but_only_exit_existing_position() -> None:
    broker = _Broker()
    result = _engine(_ReactiveExitManager(), broker).run_cycle(_NOW)

    assert len(result.position_orders) == 1
    assert result.position_orders[0].side is OrderSide.SELL
    # The repeat opportunity cannot also open a second company-a position.
    assert result.entry_orders == ()
    assert broker.orders == result.position_orders


def test_position_manager_cannot_bypass_portfolio_risk_with_buy_order() -> None:
    with pytest.raises(ValueError, match="only reduce/exit"):
        _engine(_BadBuyManager(), _Broker()).run_cycle(_NOW)
