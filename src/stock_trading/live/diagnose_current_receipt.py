from __future__ import annotations

import argparse
import json
from pathlib import Path

from stock_trading.market import CandidateSnapshotBuilder, DuckDbMarketStore
from stock_trading.storage import DuckDbEventStore

from .candidates import PitCandidateAssembler
from .current_cycle_receipt import FileCurrentCycleReceiptStore
from .decision_diagnostics import (
    FileStrategyDecisionDiagnosticStore,
    diagnose_registry,
    diagnostics_payload,
    rewind_registry_calibration_before,
)
from .run_current_paper_shadow import _load_runtime_config
from .runtime_state import load_persisted_shadow_registry
from .runtime_strategy_state import FileRuntimeStrategyStateStore


def diagnose_current_receipt(
    *,
    data_root: str | Path = "data",
    runtime_dir: str | Path = "data/runtime",
    batch_id: str | None = None,
    persist: bool = False,
) -> dict:
    """Replay model scoring/gating for one completed current batch.

    The current mutable calibration overlay is loaded only into memory and rewound
    to entries strictly before the receipt's target execution session. By default
    the command is completely read-only. ``persist=True`` is the single permitted
    write: the reconstructed diagnostics are stored through the same immutable-style
    batch diagnostic store used by live cycles. Queue, ledger, receipt, strategy
    state, model artifacts, and calibration remain untouched.
    """

    data_root = Path(data_root)
    runtime_dir = Path(runtime_dir)
    receipt_store = FileCurrentCycleReceiptStore(runtime_dir / "current_cycle_receipts")
    if batch_id is None:
        receipts = receipt_store.load_all()
        if not receipts:
            raise FileNotFoundError("no current cycle receipts exist")
        receipt = max(receipts, key=lambda item: (item.completed_at, item.batch_id))
    else:
        receipt = receipt_store.load(batch_id)
        if receipt is None:
            raise FileNotFoundError(f"current cycle receipt does not exist: {batch_id}")

    company_ids = _company_ids_from_candidate_ids(receipt.candidate_ids)
    event_store = DuckDbEventStore(data_root / "normalized" / "events.duckdb")
    history = event_store.all_events(company_ids=company_ids)
    selected_id_set = set(receipt.selected_event_ids)
    selected_events = tuple(
        item for item in history if item.event_id in selected_id_set
    )
    if {item.event_id for item in selected_events} != selected_id_set:
        missing = sorted(selected_id_set - {item.event_id for item in selected_events})
        raise RuntimeError(
            f"completed receipt events are missing from normalized storage: {missing[:5]}"
        )

    config = _load_runtime_config(runtime_dir)
    market_store = DuckDbMarketStore(Path(str(config["market_db"])))
    market_store.enable_read_cache(max_series=max(8, len(company_ids) + 1))
    assembly = PitCandidateAssembler(
        CandidateSnapshotBuilder(
            market_store,
            benchmark_security_id=str(config["benchmark_security_id"]),
        )
    ).assemble(
        selected_events,
        all_events=history,
        as_of=receipt.completed_at,
        execution_date=receipt.target_execution_date,
    )
    reconstructed_ids = tuple(item.candidate_id for item in assembly.candidates)
    if set(reconstructed_ids) != set(receipt.candidate_ids):
        raise RuntimeError(
            "read-only receipt replay reconstructed different candidates: "
            f"receipt={receipt.candidate_ids} replay={reconstructed_ids}"
        )

    loaded = load_persisted_shadow_registry(runtime_dir=runtime_dir)
    restored_state_ids = FileRuntimeStrategyStateStore(
        runtime_dir / "strategy_state"
    ).restore_registry(loaded.registry)
    rewind_registry_calibration_before(
        loaded.registry,
        receipt.target_execution_date,
    )
    diagnostics = diagnose_registry(loaded.registry, assembly.candidates)

    diagnostic_ids = {item.strategy_id for item in diagnostics}
    expected_ids = {receipt.champion_strategy_id, *receipt.shadow_strategy_ids}
    if diagnostic_ids != expected_ids:
        raise RuntimeError(
            "receipt strategy cohort differs from currently verified runtime cohort: "
            f"receipt={sorted(expected_ids)} current={sorted(diagnostic_ids)}"
        )

    diagnostic_path = None
    if persist:
        diagnostic_path = FileStrategyDecisionDiagnosticStore(
            runtime_dir / "decision_diagnostics"
        ).write(
            batch_id=receipt.batch_id,
            as_of=receipt.completed_at,
            target_execution_date=receipt.target_execution_date,
            diagnostics=diagnostics,
        )

    return {
        "mode": (
            "completed_receipt_diagnosis_with_persisted_audit"
            if persist
            else "read_only_completed_receipt_diagnosis"
        ),
        "writes_performed": persist,
        "diagnostic_path": str(diagnostic_path) if diagnostic_path else None,
        "batch_id": receipt.batch_id,
        "completed_at": receipt.completed_at.isoformat(),
        "target_execution_date": receipt.target_execution_date.isoformat(),
        "selected_event_ids": list(receipt.selected_event_ids),
        "candidate_ids": list(receipt.candidate_ids),
        "reconstructed_candidate_ids": list(reconstructed_ids),
        "restored_runtime_state_ids": list(restored_state_ids),
        "calibration_rewind": "strictly_before_target_execution_date",
        "strategies": diagnostics_payload(diagnostics),
    }


def _company_ids_from_candidate_ids(candidate_ids: tuple[str, ...]) -> tuple[str, ...]:
    companies: set[str] = set()
    for candidate_id in candidate_ids:
        prefix = "opportunity:"
        if not candidate_id.startswith(prefix):
            raise ValueError(f"unsupported current candidate ID: {candidate_id}")
        body = candidate_id[len(prefix):]
        try:
            company_id, execution_date = body.rsplit(":", 1)
        except ValueError as exc:
            raise ValueError(f"invalid current candidate ID: {candidate_id}") from exc
        if not company_id or not execution_date:
            raise ValueError(f"invalid current candidate ID: {candidate_id}")
        companies.add(company_id)
    if not companies:
        raise ValueError("completed receipt contains no candidate companies")
    return tuple(sorted(companies))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct per-horizon predictions, calibration ranks, eligibility gates "
            "and rejection reasons for one completed PAPER/SHADOW receipt."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--runtime-dir", type=Path, default=Path("data/runtime"))
    parser.add_argument("--batch-id")
    parser.add_argument(
        "--persist",
        action="store_true",
        help=(
            "Persist the reconstructed immutable-style decision diagnostic audit. "
            "No trading or calibration state is changed."
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = diagnose_current_receipt(
        data_root=args.data_root,
        runtime_dir=args.runtime_dir,
        batch_id=args.batch_id,
        persist=args.persist,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
