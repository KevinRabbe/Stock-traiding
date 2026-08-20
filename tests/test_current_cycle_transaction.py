from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from stock_trading.engine import OrderIntent, OrderSide
from stock_trading.execution import FilePaperLedger, PaperLedgerState
from stock_trading.live.current_cycle_receipt import (
    FileCurrentCycleReceiptStore,
    batch_id,
    reconcile_completed_receipts,
)
from stock_trading.live.current_cycle_transaction import (
    CurrentCycleTransaction,
    FileCurrentCycleTransactionStore,
    recover_submitted_batch_receipts,
)
from stock_trading.live.event_intake import (
    CurrentEventIntakeState,
    FileCurrentEventQueue,
    FilingCursor,
    PendingTrigger,
)


UTC = timezone.utc
_TARGET = date(2026, 8, 20)
_PUBLIC_TIME = datetime(2026, 8, 19, 18, 0, tzinfo=UTC)


def _transaction(event_id: str, *, prepared_at: datetime | None = None) -> CurrentCycleTransaction:
    resolved_batch = batch_id(_TARGET, (event_id,))
    return CurrentCycleTransaction(
        batch_id=resolved_batch,
        prepared_at=prepared_at or datetime(2026, 8, 19, 19, 0, tzinfo=UTC),
        target_execution_date=_TARGET,
        selected_event_ids=(event_id,),
        candidate_ids=("opportunity:company-a:2026-08-20",),
        champion_strategy_id="v5",
        shadow_strategy_ids=("shadow-a",),
    )


def _submitted_buy(transaction: CurrentCycleTransaction) -> OrderIntent:
    return OrderIntent(
        order_id="buy-a",
        strategy_id=transaction.champion_strategy_id,
        candidate_id=transaction.candidate_ids[0],
        event_id=transaction.candidate_ids[0],
        company_id="company-a",
        security_id="security-a",
        side=OrderSide.BUY,
        allocation_pct=0.02,
        created_at=transaction.prepared_at,
        horizon_sessions=20,
        execute_on=transaction.target_execution_date,
        metadata={"runtime_batch_id": transaction.batch_id},
    )


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


def test_submitted_batch_recovers_receipt_before_later_session_classification(tmp_path) -> None:
    event_id = "evt-cross-session"
    transaction = _transaction(event_id)
    transaction_store = FileCurrentCycleTransactionStore(tmp_path / "transactions")
    transaction_store.write(transaction)
    receipt_store = FileCurrentCycleReceiptStore(tmp_path / "receipts")
    ledger = FilePaperLedger(tmp_path / "paper.json")
    order = _submitted_buy(transaction)
    ledger.save(
        PaperLedgerState(
            cash=10_000.0,
            pending_orders=(order,),
            submitted_orders=(order,),
        )
    )
    queue = _queue(tmp_path, event_id)

    # Simulate a restart well after the original execution session. Recovery does
    # not need to reclassify or reconstruct the event; it uses the prepared batch.
    recovery = recover_submitted_batch_receipts(
        transaction_store=transaction_store,
        receipt_store=receipt_store,
        paper_ledger=ledger,
    )
    assert recovery.recovered_batch_ids == (transaction.batch_id,)
    receipt = receipt_store.load(transaction.batch_id)
    assert receipt is not None
    assert receipt.selected_event_ids == (event_id,)
    assert receipt.champion_entry_order_ids == ("buy-a",)

    reconciliation = reconcile_completed_receipts(
        queue,
        receipt_store,
        paper_ledger=ledger,
    )
    assert reconciliation.acknowledged_pending_event_count == 1
    assert queue.pending() == ()


def test_prepared_transaction_without_submitted_buy_is_not_recovered(tmp_path) -> None:
    transaction = _transaction("evt-not-submitted")
    transaction_store = FileCurrentCycleTransactionStore(tmp_path / "transactions")
    transaction_store.write(transaction)
    receipt_store = FileCurrentCycleReceiptStore(tmp_path / "receipts")
    ledger = FilePaperLedger(tmp_path / "paper.json")
    ledger.save(PaperLedgerState(cash=10_000.0))

    result = recover_submitted_batch_receipts(
        transaction_store=transaction_store,
        receipt_store=receipt_store,
        paper_ledger=ledger,
    )

    assert result.recovered_receipt_count == 0
    assert receipt_store.load(transaction.batch_id) is None


def test_transaction_retry_accepts_new_wall_clock_when_identity_is_unchanged(tmp_path) -> None:
    store = FileCurrentCycleTransactionStore(tmp_path / "transactions")
    original = _transaction("evt-retry")
    path = store.write(original)
    retry = _transaction(
        "evt-retry",
        prepared_at=original.prepared_at + timedelta(minutes=5),
    )

    assert store.write(retry) == path
    assert store.load(original.batch_id) == original
