from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from stock_trading.engine import (
    BasicOpportunityRiskPolicy,
    ExecutionReport,
    FeatureSnapshot,
    FixedAllocationPortfolioPolicy,
    HoldPositions,
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
from stock_trading.execution import FilePaperLedger, SessionClosePaperExecutionBroker


_NOW = datetime(2025, 1, 2, 18, 0, tzinfo=timezone.utc)
_EXECUTION_DATE = date(2025, 1, 3)


class _Source:
    def __init__(self, candidates: tuple[FeatureSnapshot, ...]) -> None:
        self._candidates = candidates

    def candidates(self, as_of):
        assert as_of == _NOW
        return self._candidates


class _Strategy:
    strategy_id = "v5"

    def evaluate(self, candidates, portfolio):
        # Pending reservations must not masquerade as filled positions to strategy logic.
        assert all(
            not bool(position.metadata.get("pending_entry_reservation"))
            for position in portfolio.positions
        )
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
                expected_downside=0.01,
                probability_positive=0.7,
                horizon_sessions=20,
            )
            for item in candidates
        )


class _State:
    def __init__(self, snapshot: PortfolioSnapshot) -> None:
        self._snapshot = snapshot

    def snapshot(self, as_of):
        assert as_of == self._snapshot.as_of
        return self._snapshot


class _ReservationBroker:
    def __init__(self, pending: tuple[OrderIntent, ...]) -> None:
        self._pending = pending
        self.orders: tuple[OrderIntent, ...] = ()

    def pending_entry_orders(self) -> tuple[OrderIntent, ...]:
        return self._pending

    def execute(self, orders):
        self.orders = tuple(orders)
        return tuple(
            ExecutionReport(order_id=item.order_id, accepted=True, executed_at=_NOW)
            for item in orders
        )


class _NoPrices:
    def price(self, security_id, as_of):
        del security_id, as_of
        return None


def _registry() -> StrategyRegistry:
    strategy = _Strategy()
    registry = StrategyRegistry()
    registry.register(
        strategy,
        StrategyRecord(strategy_id=strategy.strategy_id, stage=StrategyStage.PAPER),
    )
    registry.set_champion(strategy.strategy_id)
    return registry


def _candidate(candidate_id: str, company_id: str, score: float) -> FeatureSnapshot:
    return FeatureSnapshot(
        candidate_id=candidate_id,
        event_id=f"event-{candidate_id}",
        company_id=company_id,
        security_id=f"security-{company_id}",
        decision_time=_NOW,
        execution_date=_EXECUTION_DATE,
        features={"score": score},
    )


def _pending_buy(order_id: str, company_id: str, allocation_pct: float = 0.02) -> OrderIntent:
    return OrderIntent(
        order_id=order_id,
        strategy_id="v5",
        candidate_id=f"candidate-{order_id}",
        event_id=f"event-{order_id}",
        company_id=company_id,
        security_id=f"security-{company_id}",
        side=OrderSide.BUY,
        allocation_pct=allocation_pct,
        created_at=_NOW,
        horizon_sessions=20,
        execute_on=_EXECUTION_DATE,
        reason="queued paper entry",
    )


def _engine(
    *,
    candidates: tuple[FeatureSnapshot, ...],
    snapshot: PortfolioSnapshot,
    broker,
    max_open_positions: int = 15,
    max_gross_exposure_pct: float = 0.30,
) -> TradingEngine:
    return TradingEngine(
        candidate_source=_Source(candidates),
        strategy_provider=_registry(),
        opportunity_risk=BasicOpportunityRiskPolicy(),
        portfolio_policy=FixedAllocationPortfolioPolicy(
            allocation_pct=0.02,
            max_open_positions=max_open_positions,
            max_gross_exposure_pct=max_gross_exposure_pct,
        ),
        portfolio_risk=PassThroughPortfolioRiskPolicy(),
        position_manager=HoldPositions(),
        state_provider=_State(snapshot),
        broker=broker,
    )


def test_pending_buy_reserves_company_and_position_slot_for_later_batch() -> None:
    snapshot = PortfolioSnapshot(
        as_of=_NOW,
        equity=10_000.0,
        cash=7_400.0,
        gross_exposure_pct=0.26,
        positions=(
            PortfolioPosition(
                position_id="open-a",
                strategy_id="v5",
                company_id="company-open",
                security_id="security-open",
                allocation_pct=0.26,
                opened_on=date(2024, 12, 20),
            ),
        ),
    )
    broker = _ReservationBroker((_pending_buy("pending-a", "company-a"),))
    engine = _engine(
        candidates=(
            _candidate("a", "company-a", 0.99),
            _candidate("b", "company-b", 0.98),
        ),
        snapshot=snapshot,
        broker=broker,
        max_open_positions=3,
    )

    result = engine.run_cycle(_NOW)

    assert result.allocation_count == 1
    assert [order.company_id for order in result.entry_orders] == ["company-b"]
    assert [order.company_id for order in broker.orders] == ["company-b"]
    # The durable account view itself remains real positions only.
    assert [position.company_id for position in snapshot.positions] == ["company-open"]


def test_pending_buys_reserve_gross_exposure_across_batches() -> None:
    snapshot = PortfolioSnapshot(
        as_of=_NOW,
        equity=10_000.0,
        cash=7_400.0,
        gross_exposure_pct=0.26,
        positions=(
            PortfolioPosition(
                position_id="open-a",
                strategy_id="v5",
                company_id="company-open",
                security_id="security-open",
                allocation_pct=0.26,
                opened_on=date(2024, 12, 20),
            ),
        ),
    )
    broker = _ReservationBroker(
        (
            _pending_buy("pending-a", "company-a"),
            _pending_buy("pending-b", "company-b"),
        )
    )
    engine = _engine(
        candidates=(_candidate("c", "company-c", 0.99),),
        snapshot=snapshot,
        broker=broker,
        max_open_positions=10,
        max_gross_exposure_pct=0.30,
    )

    result = engine.run_cycle(_NOW)

    assert result.allocation_count == 0
    assert result.entry_orders == ()
    assert broker.orders == ()


def test_pending_entry_provider_fails_closed_on_non_buy_order() -> None:
    sell = OrderIntent(
        order_id="bad-sell",
        strategy_id="v5",
        company_id="company-a",
        security_id="security-company-a",
        side=OrderSide.SELL,
        allocation_pct=0.02,
        created_at=_NOW,
        execute_on=_EXECUTION_DATE,
    )
    broker = _ReservationBroker((sell,))
    engine = _engine(
        candidates=(_candidate("b", "company-b", 0.99),),
        snapshot=PortfolioSnapshot(
            as_of=_NOW,
            equity=10_000.0,
            cash=10_000.0,
            gross_exposure_pct=0.0,
        ),
        broker=broker,
    )

    with pytest.raises(ValueError, match="non-BUY"):
        engine.run_cycle(_NOW)


def test_session_paper_broker_exposes_only_durable_queued_buys(tmp_path) -> None:
    ledger = FilePaperLedger(tmp_path / "paper.json")
    broker = SessionClosePaperExecutionBroker(ledger, _NoPrices())
    buy = _pending_buy("pending-buy", "company-a")
    sell = OrderIntent(
        order_id="pending-sell",
        strategy_id="v5",
        company_id="company-b",
        security_id="security-company-b",
        side=OrderSide.SELL,
        allocation_pct=0.02,
        created_at=_NOW,
        execute_on=_EXECUTION_DATE,
    )

    broker.execute((buy, sell))

    assert [order.order_id for order in broker.pending_entry_orders()] == ["pending-buy"]
