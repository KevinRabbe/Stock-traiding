from __future__ import annotations

import json
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from stock_trading.engine import OrderIntent, OrderSide
from stock_trading.execution import FilePaperLedger, PaperLedgerState
from stock_trading.live.paper_lifecycle import service_current_paper_lifecycle
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


def test_paper_lifecycle_settles_due_entry_without_new_signal_batch(tmp_path) -> None:
    pytest.importorskip("duckdb")
    data_root = tmp_path / "data"
    runtime_dir = tmp_path / "runtime"
    market_path = tmp_path / "market.duckdb"
    ledger_path = tmp_path / "paper.json"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "paper_runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "market_db": str(market_path),
                "benchmark_security_id": "security-spy",
                "paper_ledger": str(ledger_path),
            }
        ),
        encoding="utf-8",
    )

    market_store = DuckDbMarketStore(market_path)
    market_store.put_many(
        (
            _bar(date(2025, 1, 2), open_price="98", close_price="100"),
            _bar(date(2025, 1, 3), open_price="90", close_price="110"),
        )
    )
    order = OrderIntent(
        order_id="buy-a",
        strategy_id="v5",
        company_id="company-a",
        security_id="security-a",
        side=OrderSide.BUY,
        allocation_pct=0.02,
        created_at=datetime(2025, 1, 2, 20, tzinfo=UTC),
        candidate_id="candidate-a",
        event_id="event-a",
        horizon_sessions=5,
        execute_on=date(2025, 1, 3),
        reason="next_session_open",
    )
    ledger = FilePaperLedger(ledger_path)
    ledger.save(PaperLedgerState(cash=10_000.0, pending_orders=(order,)))

    first = service_current_paper_lifecycle(
        data_root=data_root,
        runtime_dir=runtime_dir,
        as_of=datetime(2025, 1, 3, 22, tzinfo=UTC),
    )

    assert first["status"] == "completed"
    assert first["completed_session"] == "2025-01-03"
    assert first["market_sync"]["tracked_security_count"] == 1
    assert first["market_sync"]["downloaded_series_count"] == 0
    assert first["settled_order_count"] == 1
    assert first["settled_orders"][0]["fill_price"] == pytest.approx(90.0)
    assert first["open_position_count"] == 1
    assert first["pending_order_count"] == 0

    second = service_current_paper_lifecycle(
        data_root=data_root,
        runtime_dir=runtime_dir,
        as_of=datetime(2025, 1, 3, 22, 5, tzinfo=UTC),
    )

    assert second["settled_order_count"] == 0
    assert second["generated_exit_order_count"] == 0
    assert second["open_position_count"] == 1
    assert second["pending_order_count"] == 0
