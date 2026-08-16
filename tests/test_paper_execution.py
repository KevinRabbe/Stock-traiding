from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from stock_trading.engine import ExecutionStatus, OrderIntent, OrderSide
from stock_trading.execution import (
    FilePaperLedger,
    PaperExecutionBroker,
    PaperPortfolioStateProvider,
)


class _DatedPrices:
    def __init__(self, values):
        self.values = values

    def price(self, security_id, as_of):
        return self.values.get((security_id, as_of.date()))


def _buy(created_at, *, execute_on):
    return OrderIntent(
        order_id="buy-a",
        strategy_id="v5",
        company_id="company-a",
        security_id="security-a",
        side=OrderSide.BUY,
        allocation_pct=0.02,
        created_at=created_at,
        candidate_id="candidate-a",
        event_id="event-a",
        horizon_sessions=20,
        execute_on=execute_on,
        reason="paper entry",
    )


def test_future_paper_order_queues_survives_restart_and_settles_once(tmp_path) -> None:
    day_1 = datetime(2025, 1, 2, 18, tzinfo=timezone.utc)
    day_2 = datetime(2025, 1, 3, 18, tzinfo=timezone.utc)
    prices = _DatedPrices({("security-a", date(2025, 1, 3)): 100.0})
    ledger_path = tmp_path / "paper.json"
    ledger = FilePaperLedger(ledger_path, starting_cash=10_000.0)
    broker = PaperExecutionBroker(ledger, prices, per_side_cost_bps=10.0)
    order = _buy(day_1, execute_on=date(2025, 1, 3))

    queued = broker.execute((order,))
    assert queued[0].status is ExecutionStatus.QUEUED
    assert ledger.load().cash == pytest.approx(10_000.0)
    assert len(ledger.load().pending_orders) == 1

    # A new process/broker instance resumes the durable queue.
    restarted = PaperExecutionBroker(
        FilePaperLedger(ledger_path, starting_cash=1.0),
        prices,
        per_side_cost_bps=10.0,
    )
    settled = restarted.settle(day_2)
    assert len(settled) == 1
    assert settled[0].status is ExecutionStatus.FILLED
    assert settled[0].fill_price == pytest.approx(100.0)

    state = ledger.load()
    assert state.cash == pytest.approx(9_799.8)
    assert len(state.positions) == 1
    assert state.positions[0].shares == pytest.approx(2.0)
    assert state.pending_orders == ()

    # Replaying the same order ID is idempotent and cannot buy twice.
    replay = restarted.execute((order,))
    assert replay[0].status is ExecutionStatus.FILLED
    assert len(ledger.load().positions) == 1
    assert ledger.load().cash == pytest.approx(9_799.8)


def test_paper_snapshot_marks_equity_and_full_sell_returns_to_cash(tmp_path) -> None:
    day = datetime(2025, 1, 3, 18, tzinfo=timezone.utc)
    prices = _DatedPrices({("security-a", day.date()): 100.0})
    ledger = FilePaperLedger(tmp_path / "paper.json", starting_cash=10_000.0)
    broker = PaperExecutionBroker(ledger, prices, per_side_cost_bps=10.0)
    provider = PaperPortfolioStateProvider(ledger, prices)

    buy = _buy(day, execute_on=day.date())
    buy_report = broker.execute((buy,))[0]
    assert buy_report.status is ExecutionStatus.FILLED

    snapshot = provider.snapshot(day)
    assert snapshot.cash == pytest.approx(9_799.8)
    assert snapshot.equity == pytest.approx(9_999.8)
    assert len(snapshot.positions) == 1
    position = snapshot.positions[0]
    assert position.metadata["shares"] == pytest.approx(2.0)

    sell = OrderIntent(
        order_id="sell-a",
        strategy_id="v5",
        company_id="company-a",
        security_id="security-a",
        side=OrderSide.SELL,
        allocation_pct=position.allocation_pct,
        created_at=day,
        reason="paper exit",
    )
    sell_report = broker.execute((sell,))[0]
    assert sell_report.status is ExecutionStatus.FILLED

    final_snapshot = provider.snapshot(day)
    assert final_snapshot.positions == ()
    assert final_snapshot.cash == pytest.approx(9_999.6)
    assert final_snapshot.equity == pytest.approx(9_999.6)


def test_due_paper_order_stays_pending_until_price_is_available(tmp_path) -> None:
    day_1 = datetime(2025, 1, 2, 18, tzinfo=timezone.utc)
    day_2 = datetime(2025, 1, 3, 18, tzinfo=timezone.utc)
    prices = _DatedPrices({})
    ledger = FilePaperLedger(tmp_path / "paper.json")
    broker = PaperExecutionBroker(ledger, prices)
    order = _buy(day_1, execute_on=day_1.date())

    report = broker.execute((order,))[0]
    assert report.status is ExecutionStatus.QUEUED
    assert broker.settle(day_2) == ()
    assert [item.order_id for item in ledger.load().pending_orders] == ["buy-a"]


def test_paper_broker_rejects_duplicate_company_without_mutating_cash(tmp_path) -> None:
    day = datetime(2025, 1, 3, 18, tzinfo=timezone.utc)
    prices = _DatedPrices(
        {
            ("security-a", day.date()): 100.0,
            ("security-b", day.date()): 50.0,
        }
    )
    ledger = FilePaperLedger(tmp_path / "paper.json")
    broker = PaperExecutionBroker(ledger, prices, per_side_cost_bps=0.0)
    first = _buy(day, execute_on=day.date())
    assert broker.execute((first,))[0].status is ExecutionStatus.FILLED
    cash_after_first = ledger.load().cash

    duplicate = OrderIntent(
        order_id="buy-a-again",
        strategy_id="v5",
        company_id="company-a",
        security_id="security-b",
        side=OrderSide.BUY,
        allocation_pct=0.02,
        created_at=day,
        execute_on=day.date(),
    )
    rejected = broker.execute((duplicate,))[0]

    assert rejected.status is ExecutionStatus.REJECTED
    assert not rejected.accepted
    assert ledger.load().cash == pytest.approx(cash_after_first)
    assert len(ledger.load().positions) == 1
