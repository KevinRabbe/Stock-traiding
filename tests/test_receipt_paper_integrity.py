from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from stock_trading.engine import ExecutionReport, ExecutionStatus, OrderIntent, OrderSide
from stock_trading.execution import FilePaperLedger, PaperLedgerState
from stock_trading.live.current_cycle_receipt import (
    CurrentCycleReceipt,
    FileCurrentCycleReceiptStore,
    batch_id,
    reconcile_completed_receipts,
    verify_receipt_paper_orders,
)
from stock_trading.live.event_intake import (
    CurrentEventIntakeState,
    FileCurrentEventQueue,
    FilingCursor,
    PendingTrigger,
)


UTC = timezone.utc
_TARGET = date(2026, 8, 20)
_PUBLIC_TIME = datetime(2026, 8, 19, 20, 0, tzinfo=UTC)
_COMPLETED_AT = datetime(2026, 8, 20, 0, 5, tzinfo=UTC)


def _queue(tmp_path, event_id: str) -> FileCurrentEventQueue:
    queue = FileCurrentEventQueue(tmp_path / "current_event_intake.json")
    queue._save(  # noqa: SLF001 - durability boundary test
        CurrentEventIntakeState(
            watermarks={
                "0000000001": FilingCursor(
                    _PUBLIC_TIME,
                    "0000000001-26-000001",
                )
            },
            pending=(
                PendingTrigger(
                    event_id=event_id,
                    company_id="company-a",
                    public_time=_PUBLIC_TIME,
                    cik="0000000001",
                    accession_number="0000000001-26-000001",
                ),
            ),
        )
    )
    return queue


def _receipt(event_id: str, order_ids: tuple[str, ...] = ("ord-a",)) -> CurrentCycleReceipt:
    return CurrentCycleReceipt(
        batch_id=batch_id(_TARGET, (event_id,)),
        completed_at=_COMPLETED_AT,
        target_execution_date=_TARGET,
        selected_event_ids=(event_id,),
        candidate_ids=("opportunity:company-a:2026-08-20",),
        champion_strategy_id="lightgbm-v5-adaptive-horizon",
        champion_entry_order_ids=order_ids,
        shadow_strategy_ids=("shadow-a",),
    )


def _pending_order(order_id: str = "ord-a") -> OrderIntent:
    return OrderIntent(
        order_id=order_id,
        strategy_id="lightgbm-v5-adaptive-horizon",
        candidate_id="opportunity:company-a:2026-08-20",
        event_id="evt-a",
        company_id="company-a",
        security_id="security-a",
        side=OrderSide.BUY,
        allocation_pct=0.02,
        created_at=datetime(2026, 8, 19, 23, 0, tzinfo=UTC),
        horizon_sessions=20,
        execute_on=_TARGET,
        reason="test paper entry",
    )


def test_receipt_reconciliation_fails_closed_when_champion_order_is_missing(tmp_path) -> None:
    event_id = "evt-missing-order"
    queue = _queue(tmp_path, event_id)
    receipt_store = FileCurrentCycleReceiptStore(tmp_path / "receipts")
    receipt_store.write(_receipt(event_id))
    ledger = FilePaperLedger(tmp_path / "paper.json", starting_cash=10_000.0)
    ledger.save(PaperLedgerState(cash=10_000.0))

    with pytest.raises(RuntimeError, match="PAPER order integrity failure"):
        reconcile_completed_receipts(
            queue,
            receipt_store,
            paper_ledger=ledger,
        )

    assert [item.event_id for item in queue.pending()] == [event_id]


def test_receipt_reconciliation_accepts_durable_pending_champion_order(tmp_path) -> None:
    event_id = "evt-pending-order"
    queue = _queue(tmp_path, event_id)
    receipt_store = FileCurrentCycleReceiptStore(tmp_path / "receipts")
    receipt = _receipt(event_id)
    receipt_store.write(receipt)
    ledger = FilePaperLedger(tmp_path / "paper.json", starting_cash=10_000.0)
    ledger.save(
        PaperLedgerState(
            cash=10_000.0,
            pending_orders=(_pending_order(),),
        )
    )

    result = reconcile_completed_receipts(
        queue,
        receipt_store,
        paper_ledger=ledger,
    )

    assert result.matched_receipt_count == 1
    assert result.acknowledged_pending_event_count == 1
    assert result.referenced_champion_order_count == 1
    assert result.pending_champion_order_count == 1
    assert result.completed_champion_order_count == 0
    assert queue.pending() == ()


def test_receipt_integrity_survives_order_transition_to_completed_report(tmp_path) -> None:
    receipt = _receipt("evt-completed-order")
    ledger = FilePaperLedger(tmp_path / "paper.json", starting_cash=10_000.0)
    ledger.save(
        PaperLedgerState(
            cash=10_000.0,
            completed_reports=(
                ExecutionReport(
                    order_id="ord-a",
                    accepted=True,
                    executed_at=datetime(2026, 8, 20, 13, 30, tzinfo=UTC),
                    fill_price=100.0,
                    status=ExecutionStatus.FILLED,
                ),
            ),
        )
    )

    result = verify_receipt_paper_orders((receipt,), ledger)

    assert result.receipt_count == 1
    assert result.referenced_champion_order_count == 1
    assert result.pending_champion_order_count == 0
    assert result.completed_champion_order_count == 1


def test_current_cycle_receipt_rejects_duplicate_champion_order_ids() -> None:
    with pytest.raises(ValueError, match="duplicate champion entry order IDs"):
        _receipt("evt-duplicate-order", ("ord-a", "ord-a"))
