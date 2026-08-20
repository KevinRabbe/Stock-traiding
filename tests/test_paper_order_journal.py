from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone

import pytest

from stock_trading.engine import ExecutionStatus, OrderIntent, OrderSide
from stock_trading.execution import (
    FilePaperLedger,
    PaperExecutionBroker,
    SessionBarPaperExecutionBroker,
)
from stock_trading.live.paper_order_recovery import receipt_champion_entry_order_ids
from stock_trading.market import DuckDbMarketStore


UTC = timezone.utc


class _DatedPrices:
    def price(self, security_id, as_of):
        if security_id == "security-a" and as_of.date() == date(2025, 1, 3):
            return 100.0
        return None


def _buy(
    *,
    order_id: str = "buy-a",
    created_at: datetime | None = None,
    metadata: dict | None = None,
) -> OrderIntent:
    return OrderIntent(
        order_id=order_id,
        strategy_id="v5",
        candidate_id="candidate-a",
        event_id="event-a",
        company_id="company-a",
        security_id="security-a",
        side=OrderSide.BUY,
        allocation_pct=0.02,
        created_at=created_at or datetime(2025, 1, 2, 18, 0, tzinfo=UTC),
        horizon_sessions=5,
        execute_on=date(2025, 1, 3),
        reason="next_session_open",
        metadata=metadata or {},
    )


def test_submitted_order_journal_survives_pending_to_completed(tmp_path) -> None:
    ledger = FilePaperLedger(tmp_path / "paper.json", starting_cash=10_000.0)
    broker = PaperExecutionBroker(ledger, _DatedPrices(), per_side_cost_bps=0.0)
    order = _buy()

    queued = broker.execute((order,))[0]
    assert queued.status is ExecutionStatus.QUEUED
    state = ledger.load()
    assert [item.order_id for item in state.pending_orders] == ["buy-a"]
    assert state.submitted_orders == (order,)

    settled = broker.settle(datetime(2025, 1, 3, 18, 0, tzinfo=UTC))[0]
    assert settled.status is ExecutionStatus.FILLED
    state = ledger.load()
    assert state.pending_orders == ()
    assert [item.order_id for item in state.completed_reports] == ["buy-a"]
    assert state.submitted_orders == (order,)


def test_idempotent_retry_keeps_original_submitted_order_timestamp(tmp_path) -> None:
    ledger = FilePaperLedger(tmp_path / "paper.json")
    broker = PaperExecutionBroker(ledger, _DatedPrices())
    original = _buy(created_at=datetime(2025, 1, 2, 18, 0, tzinfo=UTC))
    retry = replace(original, created_at=datetime(2025, 1, 2, 19, 0, tzinfo=UTC))

    assert broker.execute((original,))[0].status is ExecutionStatus.QUEUED
    assert broker.execute((retry,))[0].status is ExecutionStatus.QUEUED

    submitted = ledger.load().submitted_orders
    assert len(submitted) == 1
    assert submitted[0].created_at == original.created_at


def test_reused_order_id_with_different_economic_intent_fails_closed(tmp_path) -> None:
    ledger = FilePaperLedger(tmp_path / "paper.json")
    broker = PaperExecutionBroker(ledger, _DatedPrices())
    original = _buy()
    broker.execute((original,))

    changed = replace(original, allocation_pct=0.03)
    with pytest.raises(ValueError, match="different economic intent"):
        broker.execute((changed,))


def test_runtime_batch_tag_recovers_order_after_retry_suppresses_reemission(tmp_path) -> None:
    pytest.importorskip("duckdb")
    ledger = FilePaperLedger(tmp_path / "paper.json")
    broker = SessionBarPaperExecutionBroker(
        ledger,
        DuckDbMarketStore(tmp_path / "market.duckdb"),
        runtime_batch_id="batch_a",
    )

    assert broker.execute((_buy(),))[0].status is ExecutionStatus.QUEUED
    submitted = broker.submitted_entry_orders()
    assert len(submitted) == 1
    assert submitted[0].metadata["runtime_batch_id"] == "batch_a"

    recovered = receipt_champion_entry_order_ids(
        batch_id="batch_a",
        champion_strategy_id="v5",
        emitted_entry_orders=(),
        broker=broker,
    )
    assert recovered == ("buy-a",)

    unrelated = receipt_champion_entry_order_ids(
        batch_id="batch_b",
        champion_strategy_id="v5",
        emitted_entry_orders=(),
        broker=broker,
    )
    assert unrelated == ()


def test_runtime_batch_cannot_claim_existing_order_from_different_batch(tmp_path) -> None:
    pytest.importorskip("duckdb")
    market_store = DuckDbMarketStore(tmp_path / "market.duckdb")
    ledger = FilePaperLedger(tmp_path / "paper.json")
    first = SessionBarPaperExecutionBroker(
        ledger,
        market_store,
        runtime_batch_id="batch_a",
    )
    first.execute((_buy(),))

    second = SessionBarPaperExecutionBroker(
        ledger,
        market_store,
        runtime_batch_id="batch_b",
    )
    with pytest.raises(ValueError, match="different runtime batch"):
        second.execute((_buy(),))


def test_receipt_recovery_fails_if_fresh_emitted_order_is_not_in_journal() -> None:
    class _EmptyJournalBroker:
        def submitted_entry_orders(self):
            return ()

    with pytest.raises(RuntimeError, match="missing from durable submitted PAPER journal"):
        receipt_champion_entry_order_ids(
            batch_id="batch_a",
            champion_strategy_id="v5",
            emitted_entry_orders=(_buy(),),
            broker=_EmptyJournalBroker(),
        )
