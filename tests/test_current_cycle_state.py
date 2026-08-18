from __future__ import annotations

from datetime import date, datetime, timezone

from stock_trading.engine import OrderIntent, OrderSide
from stock_trading.execution import (
    FilePaperLedger,
    PaperLedgerState,
    SessionClosePaperExecutionBroker,
)
from stock_trading.live.candidates import (
    EventBatchPitCandidateSource,
    PitCandidateAssembly,
)
from stock_trading.live.current_cycle_receipt import (
    CurrentCycleReceipt,
    FileCurrentCycleReceiptStore,
    batch_id,
    reconcile_completed_receipts,
)
from stock_trading.live.event_intake import (
    CurrentEventIntakeState,
    FileCurrentEventQueue,
    FilingCursor,
    PendingBatchSelection,
    PendingTrigger,
)
from stock_trading.live.pending_disposition import (
    FileStaleTriggerDispositionStore,
    dispose_stale_selection,
)
from stock_trading.live.runtime_strategy_state import FileRuntimeStrategyStateStore
from stock_trading.ml.online_calibration import RollingScoreHistory
from stock_trading.strategies.v5_adaptive_horizon import (
    V5AdaptiveHorizonStrategy,
    V5CalibrationState,
    V5StrategyConfig,
)


UTC = timezone.utc


def test_current_candidate_source_uses_cycle_target_when_available() -> None:
    as_of = datetime(2026, 8, 18, 10, 0, tzinfo=UTC)

    class Resolver:
        def execution_date(self, value):
            del value
            return date(2026, 8, 19)

        def cycle_execution_date(self, value):
            del value
            return date(2026, 8, 18)

    class Triggers:
        def events(self, value):
            del value
            return ()

    class EventStore:
        def all_events(self, *, company_ids):
            raise AssertionError(company_ids)

    class Assembler:
        def __init__(self):
            self.execution_date = None

        def assemble(self, trigger_events, *, all_events, as_of, execution_date):
            del trigger_events, all_events
            self.execution_date = execution_date
            return PitCandidateAssembly(
                as_of=as_of,
                execution_date=execution_date,
                trigger_event_count=0,
                affected_company_count=0,
                context_opportunity_count=0,
                candidate_count=0,
                candidates=(),
            )

    assembler = Assembler()
    source = EventBatchPitCandidateSource(
        event_store=EventStore(),  # type: ignore[arg-type]
        assembler=assembler,  # type: ignore[arg-type]
        trigger_provider=Triggers(),  # type: ignore[arg-type]
        session_resolver=Resolver(),  # type: ignore[arg-type]
    )

    assert source.candidates(as_of) == ()
    assert assembler.execution_date == date(2026, 8, 18)
    assert source.last_assembly is not None
    assert source.last_assembly.execution_date == date(2026, 8, 18)


def test_stale_disposition_is_durable_before_queue_removal_and_idempotent(tmp_path) -> None:
    queue = FileCurrentEventQueue(tmp_path / "current_event_intake.json")
    public_time = datetime(2026, 8, 14, 20, 0, tzinfo=UTC)
    pending = PendingTrigger(
        event_id="evt-stale",
        company_id="company-a",
        public_time=public_time,
        cik="0000000001",
        accession_number="0000000001-26-000001",
    )
    queue._save(  # noqa: SLF001 - durability boundary test
        CurrentEventIntakeState(
            watermarks={
                "0000000001": FilingCursor(
                    public_time,
                    "0000000001-26-000001",
                )
            },
            pending=(pending,),
        )
    )
    selection = PendingBatchSelection(
        target_execution_date=date(2026, 8, 18),
        selected_event_ids=(),
        stale_event_ids=(pending.event_id,),
        future_event_ids=(),
    )

    class Resolver:
        def execution_date(self, value):
            del value
            return date(2026, 8, 17)

    store = FileStaleTriggerDispositionStore(tmp_path / "stale.json")
    first = dispose_stale_selection(
        queue=queue,
        store=store,
        selection=selection,
        session_resolver=Resolver(),  # type: ignore[arg-type]
        disposed_at=datetime(2026, 8, 18, 10, 0, tzinfo=UTC),
    )
    assert first.recorded_count == 1
    assert first.removed_from_pending == 1
    assert queue.pending() == ()
    assert len(store.load()) == 1

    second = dispose_stale_selection(
        queue=queue,
        store=store,
        selection=selection,
        session_resolver=Resolver(),  # type: ignore[arg-type]
        disposed_at=datetime(2026, 8, 18, 10, 1, tzinfo=UTC),
    )
    assert second.recorded_count == 0
    assert second.removed_from_pending == 0
    assert len(store.load()) == 1


def test_current_cycle_receipt_is_deterministic_and_reloadable(tmp_path) -> None:
    events = ("evt-b", "evt-a")
    target = date(2026, 8, 18)
    identity = batch_id(target, events)
    store = FileCurrentCycleReceiptStore(tmp_path / "receipts")
    receipt = CurrentCycleReceipt(
        batch_id=identity,
        completed_at=datetime(2026, 8, 18, 10, 0, tzinfo=UTC),
        target_execution_date=target,
        selected_event_ids=events,
        candidate_ids=("opportunity:a:2026-08-18",),
        champion_strategy_id="champion",
        champion_entry_order_ids=("ord-1",),
        shadow_strategy_ids=("shadow-a", "shadow-b"),
    )

    path = store.write(receipt)
    assert path.is_file()
    assert store.load(identity) == receipt
    assert batch_id(target, tuple(reversed(events))) == identity


def test_completed_receipt_reconciles_pending_before_session_classification(tmp_path) -> None:
    target = date(2026, 8, 18)
    public_time = datetime(2026, 8, 17, 20, 0, tzinfo=UTC)
    event_id = "evt-completed"
    queue = FileCurrentEventQueue(tmp_path / "current_event_intake.json")
    queue._save(  # noqa: SLF001 - crash-recovery boundary test
        CurrentEventIntakeState(
            watermarks={
                "0000000001": FilingCursor(
                    public_time,
                    "0000000001-26-000001",
                )
            },
            pending=(
                PendingTrigger(
                    event_id=event_id,
                    company_id="company-a",
                    public_time=public_time,
                    cik="0000000001",
                    accession_number="0000000001-26-000001",
                ),
            ),
        )
    )
    identity = batch_id(target, (event_id,))
    receipt_store = FileCurrentCycleReceiptStore(tmp_path / "receipts")
    receipt_store.write(
        CurrentCycleReceipt(
            batch_id=identity,
            completed_at=datetime(2026, 8, 18, 13, 0, tzinfo=UTC),
            target_execution_date=target,
            selected_event_ids=(event_id,),
            candidate_ids=("opportunity:company-a:2026-08-18",),
            champion_strategy_id="champion",
            champion_entry_order_ids=("ord-a",),
            shadow_strategy_ids=("shadow-a",),
        )
    )

    result = reconcile_completed_receipts(queue, receipt_store)

    assert result.receipt_count == 1
    assert result.matched_receipt_count == 1
    assert result.acknowledged_pending_event_count == 1
    assert result.matched_batch_ids == (identity,)
    assert queue.pending() == ()
    second = reconcile_completed_receipts(queue, receipt_store)
    assert second.matched_receipt_count == 0
    assert second.acknowledged_pending_event_count == 0


def _strategy_with_history(score: float) -> V5AdaptiveHorizonStrategy:
    strategy = object.__new__(V5AdaptiveHorizonStrategy)
    strategy.config = V5StrategyConfig(
        strategy_id="test-strategy",
        horizons=(5,),
        calibration_window_days=365,
    )
    profit = RollingScoreHistory(window_days=365)
    alpha = RollingScoreHistory(window_days=365)
    final = RollingScoreHistory(window_days=365)
    day = date(2026, 8, 18)
    profit.seed(((day, score),))
    alpha.seed(((day, score + 1.0),))
    final.seed(((day, score + 2.0),))
    strategy.calibration = V5CalibrationState(
        profit_histories={5: profit},
        alpha_histories={5: alpha},
        final_history=final,
    )
    strategy.models = {}
    return strategy


def test_runtime_strategy_state_is_bound_to_manifest_and_restores_history(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"artifact":"a"}\n', encoding="utf-8")
    store = FileRuntimeStrategyStateStore(tmp_path / "state")
    original = _strategy_with_history(1.25)
    store.save(original, manifest, completed_batch_id="batch_test")

    restored = _strategy_with_history(99.0)
    assert store.restore(restored, manifest) is True
    assert restored.calibration.profit_histories[5].snapshot() == (
        (date(2026, 8, 18), 1.25),
    )
    assert restored.calibration.alpha_histories[5].snapshot() == (
        (date(2026, 8, 18), 2.25),
    )

    manifest.write_text('{"artifact":"changed"}\n', encoding="utf-8")
    try:
        store.restore(_strategy_with_history(0.0), manifest)
    except ValueError as exc:
        assert "artifact identity mismatch" in str(exc)
    else:
        raise AssertionError("artifact mutation must invalidate runtime calibration overlay")


def test_session_close_broker_settles_on_original_execute_date(tmp_path) -> None:
    class Prices:
        def __init__(self):
            self.requested_dates = []

        def price(self, security_id, as_of):
            assert security_id == "security-a"
            self.requested_dates.append(as_of.date())
            return 100.0 if as_of.date() == date(2026, 8, 18) else None

    prices = Prices()
    ledger = FilePaperLedger(tmp_path / "paper.json", starting_cash=10_000.0)
    ledger.save(PaperLedgerState(cash=10_000.0))
    broker = SessionClosePaperExecutionBroker(ledger, prices)
    created = datetime(2026, 8, 17, 20, 0, tzinfo=UTC)
    order = OrderIntent(
        order_id="ord-a",
        strategy_id="strategy-a",
        candidate_id="candidate-a",
        event_id="event-a",
        company_id="company-a",
        security_id="security-a",
        side=OrderSide.BUY,
        allocation_pct=0.02,
        created_at=created,
        horizon_sessions=20,
        execute_on=date(2026, 8, 18),
        reason="test",
    )
    queued = broker.execute((order,))
    assert queued[0].status.value == "queued"

    settled = broker.settle(datetime(2026, 8, 19, 12, 0, tzinfo=UTC))
    assert len(settled) == 1
    assert settled[0].status.value == "filled"
    assert settled[0].executed_at.date() == date(2026, 8, 18)
    state = ledger.load()
    assert len(state.positions) == 1
    assert state.positions[0].opened_at.date() == date(2026, 8, 18)
    assert date(2026, 8, 18) in prices.requested_dates
    assert date(2026, 8, 19) not in prices.requested_dates
