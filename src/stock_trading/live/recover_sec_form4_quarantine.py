from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from stock_trading.sec import SecClient
from stock_trading.storage import DuckDbEventStore, FileRawStore

from .event_intake import DurablePendingTriggerProvider
from .form4_quarantine import FileForm4Quarantine
from .form4_recovery import (
    FileForm4Recovery,
    Form4QuarantineRecovery,
    RecoverableCurrentEventQueue,
)
from .session_calendar import XnysExecutionSessionResolver


def _parse_as_of(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("--as-of must include a timezone offset")
    return parsed.astimezone(timezone.utc)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-discover raw SEC ownership XML for active Form 4 quarantine records, "
            "normalize proven-valid documents, restore their pending event IDs without "
            "rewinding source watermarks, and retain a durable recovery audit."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--runtime-dir", type=Path, default=Path("data/runtime"))
    parser.add_argument("--max-filings", type=int)
    parser.add_argument("--as-of")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.max_filings is not None and args.max_filings <= 0:
        raise ValueError("--max-filings must be > 0")
    as_of = _parse_as_of(args.as_of)
    user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not user_agent:
        raise RuntimeError(
            "SEC_USER_AGENT is required and must identify the application/contact"
        )

    # Calendar construction is an explicit preflight because the final diagnostics
    # classify recovered pending events by the current tradable session. Fail before
    # mutating recovery state if the runtime dependency is unavailable.
    resolver = XnysExecutionSessionResolver()
    raw_store = FileRawStore(args.data_root / "raw")
    event_store = DuckDbEventStore(args.data_root / "normalized" / "events.duckdb")
    queue = RecoverableCurrentEventQueue(args.runtime_dir / "current_event_intake.json")
    quarantine = FileForm4Quarantine(args.runtime_dir / "form4_quarantine.json")
    recovery = FileForm4Recovery(args.runtime_dir / "form4_quarantine_recovery.json")

    with SecClient(user_agent) as client:
        result = Form4QuarantineRecovery(
            client=client,
            raw_store=raw_store,
            event_store=event_store,
            queue=queue,
            quarantine=quarantine,
            recovery=recovery,
        ).recover(as_of=as_of, max_filings=args.max_filings)

    provider = DurablePendingTriggerProvider(
        queue=queue,
        event_store=event_store,
        session_resolver=resolver,
    )
    selected = provider.events(as_of)
    selection = provider.last_selection
    if selection is None:
        raise RuntimeError("pending trigger provider did not produce selection diagnostics")

    print(
        json.dumps(
            {
                "as_of": as_of.isoformat(),
                "recovery": {
                    "attempted": result.attempted,
                    "recovered": result.recovered,
                    "already_recovered": result.already_recovered,
                    "failed": result.failed,
                    "events_normalized": result.events_normalized,
                    "pending_events_added": result.pending_events_added,
                    "unresolved_quarantine_count": result.unresolved_quarantine_count,
                    "recovery_count": result.recovery_count,
                    "recovered_filings": [
                        {
                            "accepted_at": item.accepted_at.isoformat(),
                            "cik": item.cik,
                            "accession_number": item.accession_number,
                            "original_raw_artifact_id": item.original_raw_artifact_id,
                            "recovered_raw_artifact_id": item.recovered_raw_artifact_id,
                            "event_count": len(item.event_ids),
                            "event_ids": list(item.event_ids),
                            "pending_events_added": item.pending_events_added,
                        }
                        for item in result.recovered_filings
                    ],
                    "failures": [
                        {
                            "cik": item.cik,
                            "accession_number": item.accession_number,
                            "error_type": item.error_type,
                            "error_message": item.error_message,
                        }
                        for item in result.failures
                    ],
                },
                "pending_session_selection": {
                    "target_execution_date": str(selection.target_execution_date),
                    "selected_event_count": len(selected),
                    "selected_event_ids": list(selection.selected_event_ids),
                    "stale_event_count": len(selection.stale_event_ids),
                    "stale_event_ids": list(selection.stale_event_ids),
                    "future_event_count": len(selection.future_event_ids),
                    "future_event_ids": list(selection.future_event_ids),
                    "acknowledged": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
