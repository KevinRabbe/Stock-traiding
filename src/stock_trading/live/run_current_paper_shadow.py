from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from stock_trading.core import as_utc
from stock_trading.engine import (
    BasicOpportunityRiskPolicy,
    FixedAllocationPortfolioPolicy,
    JsonlEngineAuditObserver,
    PassThroughPortfolioRiskPolicy,
    TradingEngine,
)
from stock_trading.execution import (
    DuckDbLatestClosePriceProvider,
    FilePaperLedger,
    PaperPortfolioStateProvider,
    SessionBarPaperExecutionBroker,
)
from stock_trading.local_secrets import load_tiingo_credentials
from stock_trading.market import CandidateSnapshotBuilder, DuckDbMarketStore, TiingoClient
from stock_trading.positions import FixedHorizonPositionManager
from stock_trading.storage import DuckDbEventStore

from .candidates import EventBatchPitCandidateSource, PitCandidateAssembler
from .current_cycle_receipt import (
    CurrentCycleReceipt,
    FileCurrentCycleReceiptStore,
    batch_id,
    reconcile_completed_receipts,
    verify_receipt_paper_orders,
)
from .current_cycle_transaction import (
    CurrentCycleTransaction,
    FileCurrentCycleTransactionStore,
    recover_submitted_batch_receipts,
)
from .current_market import sync_pending_current_market
from .decision_diagnostics import (
    FileStrategyDecisionDiagnosticStore,
    diagnose_registry,
    validate_diagnostic_counts,
)
from .event_intake import DurablePendingTriggerProvider, FileCurrentEventQueue
from .paper_order_recovery import receipt_champion_entry_order_ids
from .pending_disposition import (
    FileStaleTriggerDispositionStore,
    dispose_stale_selection,
)
from .runtime_state import load_persisted_shadow_registry
from .runtime_strategy_state import FileRuntimeStrategyStateStore
from .service import ShadowStrategyEvaluator, TradingService
from .session_calendar import XnysExecutionSessionResolver
from .shadow_persistence import JsonlShadowAuditObserver


@dataclass(frozen=True, slots=True)
class _StaticCandidateSource:
    expected_as_of: datetime
    values: tuple

    def candidates(self, as_of: datetime):
        if as_utc(as_of) != self.expected_as_of:
            raise ValueError("static current candidate source used for a different cycle time")
        return self.values


def _parse_as_of(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("--as-of must include a timezone offset")
    return parsed.astimezone(timezone.utc)


def _load_runtime_config(runtime_dir: Path) -> dict:
    path = runtime_dir / "paper_runtime.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid PAPER runtime config: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported PAPER runtime config schema")
    for name in ("market_db", "benchmark_security_id", "paper_ledger"):
        if not str(payload.get(name) or "").strip():
            raise ValueError(f"PAPER runtime config is missing {name}")
    return payload


def run_current_paper_shadow_cycle(
    *,
    data_root: str | Path = "data",
    runtime_dir: str | Path = "data/runtime",
    as_of: datetime | None = None,
) -> dict:
    """Evaluate one durable current event batch through PAPER champion + SHADOWs.

    The function has PAPER authority only. It never promotes strategies and never
    acknowledges actionable event IDs until all strategy evaluation, PAPER broker
    state, engine/shadow audits, decision diagnostics, mutable calibration overlays
    and the deterministic batch receipt are durable.
    """

    data_root = Path(data_root)
    runtime_dir = Path(runtime_dir)
    cutoff = as_utc(as_of or datetime.now(timezone.utc))
    config = _load_runtime_config(runtime_dir)

    # Preflight all session/runtime dependencies before changing queue or PAPER state.
    resolver = XnysExecutionSessionResolver()
    market_store = DuckDbMarketStore(Path(str(config["market_db"])))
    market_store.enable_read_cache(max_series=16)
    event_store = DuckDbEventStore(data_root / "normalized" / "events.duckdb")
    ledger = FilePaperLedger(Path(str(config["paper_ledger"])), starting_cash=10_000.0)
    queue = FileCurrentEventQueue(runtime_dir / "current_event_intake.json")
    receipt_store = FileCurrentCycleReceiptStore(
        runtime_dir / "current_cycle_receipts"
    )
    transaction_store = FileCurrentCycleTransactionStore(
        runtime_dir / "current_cycle_transactions"
    )
    transaction_recovery = recover_submitted_batch_receipts(
        transaction_store=transaction_store,
        receipt_store=receipt_store,
        paper_ledger=ledger,
    )
    receipt_reconciliation = reconcile_completed_receipts(
        queue,
        receipt_store,
        paper_ledger=ledger,
    )

    provider = DurablePendingTriggerProvider(
        queue=queue,
        event_store=event_store,
        session_resolver=resolver,
    )
    selected_events = provider.events(cutoff)
    selection = provider.last_selection
    if selection is None:
        raise RuntimeError("pending trigger provider did not produce selection diagnostics")

    stale_result = dispose_stale_selection(
        queue=queue,
        store=FileStaleTriggerDispositionStore(
            runtime_dir / "stale_trigger_dispositions.json"
        ),
        selection=selection,
        session_resolver=resolver,
        disposed_at=cutoff,
    )

    base_payload = {
        "as_of": cutoff.isoformat(),
        "target_execution_date": selection.target_execution_date.isoformat(),
        "transaction_receipt_recovery": {
            "transaction_count": transaction_recovery.transaction_count,
            "recovered_receipt_count": transaction_recovery.recovered_receipt_count,
            "recovered_batch_ids": list(transaction_recovery.recovered_batch_ids),
        },
        "receipt_reconciliation": {
            "receipt_count": receipt_reconciliation.receipt_count,
            "matched_receipt_count": receipt_reconciliation.matched_receipt_count,
            "acknowledged_pending_event_count": (
                receipt_reconciliation.acknowledged_pending_event_count
            ),
            "matched_batch_ids": list(receipt_reconciliation.matched_batch_ids),
            "referenced_champion_order_count": (
                receipt_reconciliation.referenced_champion_order_count
            ),
            "pending_champion_order_count": (
                receipt_reconciliation.pending_champion_order_count
            ),
            "completed_champion_order_count": (
                receipt_reconciliation.completed_champion_order_count
            ),
        },
        "stale_disposition": {
            "selected_count": stale_result.selected_count,
            "recorded_count": stale_result.recorded_count,
            "removed_from_pending": stale_result.removed_from_pending,
            "total_disposition_count": stale_result.total_disposition_count,
        },
        "selected_event_count": len(selected_events),
        "selected_event_ids": list(selection.selected_event_ids),
        "future_event_count": len(selection.future_event_ids),
    }
    if not selected_events:
        return {
            **base_payload,
            "status": "no_actionable_batch",
            "acknowledged_selected_count": 0,
            "remaining_pending_count": len(queue.pending()),
        }

    current_batch_id = batch_id(
        selection.target_execution_date,
        tuple(selection.selected_event_ids),
    )
    existing_receipt = receipt_store.load(current_batch_id)
    if existing_receipt is not None:
        if (
            existing_receipt.target_execution_date != selection.target_execution_date
            or existing_receipt.selected_event_ids != tuple(selection.selected_event_ids)
        ):
            raise ValueError("current batch receipt does not match pending event selection")
        acknowledged = queue.acknowledge(selection.selected_event_ids)
        return {
            **base_payload,
            "status": "completed_from_existing_receipt",
            "batch_id": current_batch_id,
            "receipt_path": str(
                runtime_dir / "current_cycle_receipts" / f"{current_batch_id}.json"
            ),
            "candidate_count": len(existing_receipt.candidate_ids),
            "candidate_ids": list(existing_receipt.candidate_ids),
            "champion_strategy_id": existing_receipt.champion_strategy_id,
            "shadow_strategy_ids": list(existing_receipt.shadow_strategy_ids),
            "acknowledged_selected_count": acknowledged,
            "remaining_pending_count": len(queue.pending()),
        }

    selected_id_set = set(selection.selected_event_ids)
    selected_pending = tuple(
        item
        for item in queue.pending(as_of=cutoff)
        if item.event_id in selected_id_set
    )
    if len(selected_pending) != len(selection.selected_event_ids):
        raise RuntimeError(
            "selected pending-event identities changed before current market synchronization"
        )

    sync_end_date = resolver.last_completed_session(cutoff)
    credentials = load_tiingo_credentials(data_root)
    with TiingoClient(credentials.token) as tiingo_client:
        market_sync = sync_pending_current_market(
            selected_pending,
            data_root=data_root,
            market_store=market_store,
            benchmark_security_id=str(config["benchmark_security_id"]),
            tiingo_client=tiingo_client,
            sync_end_date=sync_end_date,
        )
    market_store.clear_read_cache()
    market_sync_payload = {
        "sync_end_date": market_sync.sync_end_date.isoformat(),
        "selected_event_count": market_sync.selected_event_count,
        "accession_count": market_sync.accession_count,
        "company_count": market_sync.company_count,
        "tickers": list(market_sync.tickers),
        "metadata_refreshed": market_sync.metadata_refreshed,
        "resolved_companies": market_sync.resolved_companies,
        "unresolved_companies": market_sync.unresolved_companies,
        "downloaded_price_series": market_sync.downloaded_price_series,
        "failed_price_series": market_sync.failed_price_series,
        "reused_price_responses": market_sync.reused_price_responses,
        "skipped_complete_price_series": market_sync.skipped_complete_price_series,
        "benchmark_downloaded": market_sync.benchmark_downloaded,
        "benchmark_bars_added": market_sync.benchmark_bars_added,
        "failures": [
            {
                "company_id": item.company_id,
                "cik": item.cik,
                "ticker": item.ticker,
                "issuer_name": item.issuer_name,
                "reason": item.reason,
            }
            for item in market_sync.failures
        ],
    }
    if not market_sync.ready or market_sync.resolved_companies != market_sync.company_count:
        reasons = "; ".join(
            f"{item.company_id}/{item.ticker}:{item.reason}"
            for item in market_sync.failures
        ) or "price-series synchronization failure"
        raise RuntimeError(
            "current market synchronization is incomplete: "
            f"resolved={market_sync.resolved_companies}/{market_sync.company_count}, "
            f"failed_price_series={market_sync.failed_price_series}; {reasons}; "
            "selected event IDs remain pending for retry"
        )

    # Build and validate the actionable candidate batch before PAPER settlement or
    # strategy state can change. Every affected modeled company must produce one
    # current company/session candidate; otherwise source/market readiness is not
    # good enough to consume the public event batch.
    snapshot_builder = CandidateSnapshotBuilder(
        market_store,
        benchmark_security_id=str(config["benchmark_security_id"]),
    )
    candidate_source = EventBatchPitCandidateSource(
        event_store=event_store,
        assembler=PitCandidateAssembler(snapshot_builder),
        trigger_provider=provider,
        session_resolver=resolver,
    )
    candidates = candidate_source.candidates(cutoff)
    assembly = candidate_source.last_assembly
    if assembly is None:
        raise RuntimeError("current PIT candidate source produced no assembly diagnostics")
    if assembly.execution_date != selection.target_execution_date:
        raise RuntimeError(
            "candidate execution session differs from pending-event target session"
        )
    if assembly.trigger_event_count != len(selection.selected_event_ids):
        raise RuntimeError("candidate assembly did not consume the complete selected trigger batch")
    if assembly.candidate_count != assembly.affected_company_count:
        raise RuntimeError(
            "current market/security data is not ready for every selected company after sync: "
            f"affected={assembly.affected_company_count} candidates={assembly.candidate_count}; "
            "selected event IDs remain pending for retry"
        )

    loaded = load_persisted_shadow_registry(runtime_dir=runtime_dir)
    state_store = FileRuntimeStrategyStateStore(runtime_dir / "strategy_state")
    restored_state_ids = state_store.restore_registry(loaded.registry)
    transaction_path = transaction_store.write(
        CurrentCycleTransaction(
            batch_id=current_batch_id,
            prepared_at=cutoff,
            target_execution_date=selection.target_execution_date,
            selected_event_ids=tuple(selection.selected_event_ids),
            candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
            champion_strategy_id=loaded.champion_id,
            shadow_strategy_ids=loaded.shadow_strategy_ids,
        )
    )

    # Explain the exact pre-decision model state without updating rolling calibration.
    # The real strategy path below remains authoritative; emitted counts are compared
    # before the diagnostic audit is allowed to become durable.
    decision_diagnostics = diagnose_registry(loaded.registry, candidates)

    checkpoint_paths: tuple[Path, ...] = ()

    def persist_strategy_state_before_broker() -> None:
        nonlocal checkpoint_paths
        if checkpoint_paths:
            return
        checkpoint_paths = state_store.save_registry_checkpoint(
            loaded.registry,
            evaluated_batch_id=current_batch_id,
        )

    opportunity_risk = BasicOpportunityRiskPolicy(max_expected_downside=0.06)
    portfolio_policy = FixedAllocationPortfolioPolicy(
        allocation_pct=0.02,
        max_open_positions=15,
        max_gross_exposure_pct=0.30,
        one_position_per_company=True,
    )
    portfolio_risk = PassThroughPortfolioRiskPolicy()
    price_provider = DuckDbLatestClosePriceProvider(market_store)
    broker = SessionBarPaperExecutionBroker(
        ledger,
        market_store,
        per_side_cost_bps=10.0,
        runtime_batch_id=current_batch_id,
        before_runtime_batch_commit=persist_strategy_state_before_broker,
    )
    engine = TradingEngine(
        candidate_source=_StaticCandidateSource(cutoff, candidates),
        strategy_provider=loaded.registry,
        opportunity_risk=opportunity_risk,
        portfolio_policy=portfolio_policy,
        portfolio_risk=portfolio_risk,
        position_manager=FixedHorizonPositionManager(market_store),
        state_provider=PaperPortfolioStateProvider(ledger, price_provider),
        broker=broker,
        observer=JsonlEngineAuditObserver(runtime_dir / "paper_engine_cycles.jsonl"),
    )
    shadow_evaluator = ShadowStrategyEvaluator(
        loaded.registry,
        opportunity_risk=opportunity_risk,
        portfolio_policy=portfolio_policy,
        portfolio_risk=portfolio_risk,
    )
    result = TradingService(
        engine,
        shadow_evaluator=shadow_evaluator,
    ).run_cycle(cutoff)
    if not checkpoint_paths:
        raise RuntimeError("PAPER broker completed without persisting runtime strategy checkpoint")
    receipt_entry_order_ids = receipt_champion_entry_order_ids(
        batch_id=current_batch_id,
        champion_strategy_id=result.champion.strategy_id,
        emitted_entry_orders=result.champion.entry_orders,
        broker=broker,
    )
    emitted_entry_order_ids = {
        order.order_id for order in result.champion.entry_orders
    }
    recovered_entry_order_count = len(
        set(receipt_entry_order_ids) - emitted_entry_order_ids
    )

    validate_diagnostic_counts(
        decision_diagnostics,
        champion_strategy_id=result.champion.strategy_id,
        champion_opportunity_count=result.champion.opportunity_count,
        shadow_opportunity_counts={
            item.strategy_id: item.opportunity_count for item in result.shadows
        },
    )
    diagnostic_path = FileStrategyDecisionDiagnosticStore(
        runtime_dir / "decision_diagnostics"
    ).write(
        batch_id=current_batch_id,
        as_of=cutoff,
        target_execution_date=selection.target_execution_date,
        diagnostics=decision_diagnostics,
    )

    shadow_observer = JsonlShadowAuditObserver(runtime_dir / "shadow_evaluations.jsonl")
    shadow_observer.record(cutoff, result.shadows)

    state_paths = state_store.save_registry(
        loaded.registry,
        completed_batch_id=current_batch_id,
    )
    receipt = CurrentCycleReceipt(
        batch_id=current_batch_id,
        completed_at=datetime.now(timezone.utc),
        target_execution_date=selection.target_execution_date,
        selected_event_ids=tuple(selection.selected_event_ids),
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        champion_strategy_id=result.champion.strategy_id,
        champion_entry_order_ids=receipt_entry_order_ids,
        shadow_strategy_ids=tuple(item.strategy_id for item in result.shadows),
    )
    # Do not publish a receipt unless every champion entry it names is already
    # durable in the PAPER ledger. The broker persists before returning from execute.
    verify_receipt_paper_orders((receipt,), ledger)
    receipt_path = receipt_store.write(receipt)
    acknowledged = queue.acknowledge(selection.selected_event_ids)

    return {
        **base_payload,
        "status": "completed",
        "batch_id": current_batch_id,
        "transaction_path": str(transaction_path),
        "current_market_sync": market_sync_payload,
        "candidate_assembly": {
            "trigger_event_count": assembly.trigger_event_count,
            "affected_company_count": assembly.affected_company_count,
            "context_opportunity_count": assembly.context_opportunity_count,
            "candidate_count": assembly.candidate_count,
            "candidate_ids": [candidate.candidate_id for candidate in candidates],
        },
        "runtime_strategy_state": {
            "restored_strategy_ids": list(restored_state_ids),
            "checkpoint_paths": [str(path) for path in checkpoint_paths],
            "saved_paths": [str(path) for path in state_paths],
        },
        "decision_diagnostics_path": str(diagnostic_path),
        "champion": {
            "strategy_id": result.champion.strategy_id,
            "candidate_count": result.champion.candidate_count,
            "opportunity_count": result.champion.opportunity_count,
            "eligible_opportunity_count": result.champion.eligible_opportunity_count,
            "allocation_count": result.champion.allocation_count,
            "entry_orders": [
                {
                    "order_id": order.order_id,
                    "candidate_id": order.candidate_id,
                    "company_id": order.company_id,
                    "security_id": order.security_id,
                    "execute_on": (
                        order.execute_on.isoformat() if order.execute_on else None
                    ),
                    "horizon_sessions": order.horizon_sessions,
                    "allocation_pct": order.allocation_pct,
                }
                for order in result.champion.entry_orders
            ],
            "receipt_entry_order_ids": list(receipt_entry_order_ids),
            "recovered_entry_order_count": recovered_entry_order_count,
            "executions": [
                {
                    "order_id": report.order_id,
                    "status": report.status.value,
                    "accepted": report.accepted,
                    "fill_price": report.fill_price,
                    "message": report.message,
                }
                for report in result.champion.executions
            ],
            "settlement_count": len(result.champion.settlements),
        },
        "shadows": [
            {
                "strategy_id": item.strategy_id,
                "candidate_count": item.candidate_count,
                "opportunity_count": item.opportunity_count,
                "eligible_opportunity_count": item.eligible_opportunity_count,
                "allocation_count": item.allocation_count,
                "requested_exposure_pct": item.requested_exposure_pct,
                "top_score": item.top_score,
                "horizon_counts": [list(value) for value in item.horizon_counts],
                "selected_candidate_ids": list(item.selected_candidate_ids),
            }
            for item in result.shadows
        ],
        "receipt_path": str(receipt_path),
        "acknowledged_selected_count": acknowledged,
        "remaining_pending_count": len(queue.pending()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one current PIT batch through the persisted PAPER champion and all "
            "SHADOW challengers, persist forward calibration/decision/audit state, then "
            "acknowledge the actionable event IDs only after a crash-safe batch receipt exists."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--runtime-dir", type=Path, default=Path("data/runtime"))
    parser.add_argument("--as-of")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_current_paper_shadow_cycle(
        data_root=args.data_root,
        runtime_dir=args.runtime_dir,
        as_of=_parse_as_of(args.as_of),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
