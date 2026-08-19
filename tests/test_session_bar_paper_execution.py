from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from stock_trading.engine import ExecutionStatus, OrderIntent, OrderSide
from stock_trading.execution import FilePaperLedger, SessionBarPaperExecutionBroker
from stock_trading.market import DuckDbMarketStore, MarketBar


UTC = timezone.utc


def _bar(day: date, *, open_price: str, close_price: str) -> MarketBar:
    opening = Decimal(open_price)
    closing = Decimal(close_price)
    high = max(opening, closing) + Decimal("1")
    low = min(opening, closing) - Decimal("1")
    return MarketBar(
        security_id="security-a",
        ticker="AAA",
        date=day,
        open=opening,
        high=high,
        low=low,
        close=closing,
        volume=Decimal("1000000"),
        adj_open=opening,
        adj_high=high,
        adj_low=low,
        adj_close=closing,
        adj_volume=Decimal("1000000"),
    )


def _buy(created_at: datetime, execute_on: date) -> OrderIntent:
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
        horizon_sessions=5,
        execute_on=execute_on,
        reason="next_session_open",
    )


def test_session_bar_broker_buys_at_open_and_sells_at_close(tmp_path) -> None:
    pytest.importorskip("duckdb")
    market_store = DuckDbMarketStore(tmp_path / "market.duckdb")
    market_store.put_many(
        (
            _bar(date(2025, 1, 2), open_price="98", close_price="100"),
            _bar(date(2025, 1, 3), open_price="90", close_price="110"),
            _bar(date(2025, 1, 6), open_price="120", close_price="130"),
        )
    )
    ledger = FilePaperLedger(tmp_path / "paper.json", starting_cash=10_000.0)
    broker = SessionBarPaperExecutionBroker(
        ledger,
        market_store,
        per_side_cost_bps=10.0,
    )

    created = datetime(2025, 1, 2, 20, tzinfo=UTC)
    queued = broker.execute((_buy(created, date(2025, 1, 3)),))[0]
    assert queued.status is ExecutionStatus.QUEUED

    settled = broker.settle(datetime(2025, 1, 3, 21, tzinfo=UTC))[0]
    assert settled.status is ExecutionStatus.FILLED
    assert settled.fill_price == pytest.approx(90.0)
    position = ledger.load().positions[0]
    assert position.average_entry_price == pytest.approx(90.0)

    sell = OrderIntent(
        order_id="sell-a",
        strategy_id="v5",
        company_id="company-a",
        security_id="security-a",
        side=OrderSide.SELL,
        allocation_pct=0.02,
        created_at=datetime(2025, 1, 6, 21, tzinfo=UTC),
        execute_on=date(2025, 1, 6),
        reason="strategy_horizon_complete",
        metadata={"full_exit": True},
    )
    report = broker.execute((sell,))[0]

    assert report.status is ExecutionStatus.FILLED
    assert report.fill_price == pytest.approx(130.0)
    assert ledger.load().positions == ()


def test_session_bar_broker_restart_uses_original_execution_session(tmp_path) -> None:
    pytest.importorskip("duckdb")
    market_store = DuckDbMarketStore(tmp_path / "market.duckdb")
    market_store.put_many(
        (
            _bar(date(2025, 1, 2), open_price="98", close_price="100"),
            _bar(date(2025, 1, 3), open_price="90", close_price="110"),
            _bar(date(2025, 1, 6), open_price="150", close_price="160"),
        )
    )
    ledger_path = tmp_path / "paper.json"
    broker = SessionBarPaperExecutionBroker(
        FilePaperLedger(ledger_path),
        market_store,
        per_side_cost_bps=0.0,
    )
    order = _buy(datetime(2025, 1, 2, 20, tzinfo=UTC), date(2025, 1, 3))
    broker.execute((order,))

    restarted = SessionBarPaperExecutionBroker(
        FilePaperLedger(ledger_path),
        market_store,
        per_side_cost_bps=0.0,
    )
    report = restarted.settle(datetime(2025, 1, 6, 21, tzinfo=UTC))[0]

    assert report.fill_price == pytest.approx(90.0)
    assert report.executed_at.date() == date(2025, 1, 3)
    assert len(restarted.ledger.load().positions) == 1
