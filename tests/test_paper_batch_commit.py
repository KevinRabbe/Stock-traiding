from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from stock_trading.engine import ExecutionStatus, OrderIntent, OrderSide
from stock_trading.execution import FilePaperLedger, SessionBarPaperExecutionBroker
from stock_trading.market import DuckDbMarketStore


UTC = timezone.utc


def _buy() -> OrderIntent:
    return OrderIntent(
        order_id="buy-a",
        strategy_id="v5",
        candidate_id="candidate-a",
        event_id="event-a",
        company_id="company-a",
        security_id="security-a",
        side=OrderSide.BUY,
        allocation_pct=0.02,
        created_at=datetime(2025, 1, 2, 18, 0, tzinfo=UTC),
        horizon_sessions=5,
        execute_on=date(2025, 1, 3),
    )


def test_runtime_state_hook_runs_before_durable_paper_submission(tmp_path) -> None:
    pytest.importorskip("duckdb")
    ledger = FilePaperLedger(tmp_path / "paper.json")
    calls: list[str] = []
    broker = SessionBarPaperExecutionBroker(
        ledger,
        DuckDbMarketStore(tmp_path / "market.duckdb"),
        runtime_batch_id="batch_a",
        before_runtime_batch_commit=lambda: calls.append("state_saved"),
    )

    report = broker.execute((_buy(),))[0]

    assert report.status is ExecutionStatus.QUEUED
    assert calls == ["state_saved"]
    assert [item.order_id for item in ledger.load().submitted_orders] == ["buy-a"]


def test_runtime_state_hook_failure_prevents_paper_submission(tmp_path) -> None:
    pytest.importorskip("duckdb")
    ledger = FilePaperLedger(tmp_path / "paper.json")

    def fail_state_save() -> None:
        raise RuntimeError("state save failed")

    broker = SessionBarPaperExecutionBroker(
        ledger,
        DuckDbMarketStore(tmp_path / "market.duckdb"),
        runtime_batch_id="batch_a",
        before_runtime_batch_commit=fail_state_save,
    )

    with pytest.raises(RuntimeError, match="state save failed"):
        broker.execute((_buy(),))

    state = ledger.load()
    assert state.pending_orders == ()
    assert state.completed_reports == ()
    assert state.submitted_orders == ()


def test_runtime_state_hook_runs_even_when_batch_has_no_orders(tmp_path) -> None:
    pytest.importorskip("duckdb")
    calls: list[str] = []
    broker = SessionBarPaperExecutionBroker(
        FilePaperLedger(tmp_path / "paper.json"),
        DuckDbMarketStore(tmp_path / "market.duckdb"),
        runtime_batch_id="batch_a",
        before_runtime_batch_commit=lambda: calls.append("state_saved"),
    )

    assert broker.execute(()) == ()
    assert calls == ["state_saved"]
