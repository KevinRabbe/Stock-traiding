from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from stock_trading.sec import SecClient
from stock_trading.storage import DuckDbEventStore, FileRawStore

from .event_intake import (
    DurablePendingTriggerProvider,
    FileCurrentEventQueue,
    SecCurrentForm4Poller,
)
from .session_calendar import XnysExecutionSessionResolver


def _modeled_ciks(
    sec_companies_path: Path,
    training_rows_path: Path,
) -> tuple[str, ...]:
    modeled_company_ids: set[str] = set()
    try:
        training_lines = training_rows_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FileNotFoundError(f"missing training rows: {training_rows_path}") from exc
    for line_number, line in enumerate(training_lines, start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            company_id = str(item["company_id"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"invalid training row {line_number}: {training_rows_path}"
            ) from exc
        modeled_company_ids.add(company_id)

    company_to_cik: dict[str, str] = {}
    try:
        company_lines = sec_companies_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FileNotFoundError(f"missing SEC company manifest: {sec_companies_path}") from exc
    for line_number, line in enumerate(company_lines, start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            company_id = str(item["company_id"])
            cik = str(item["sec_cik"]).strip().lstrip("0").zfill(10)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"invalid SEC company manifest row {line_number}: {sec_companies_path}"
            ) from exc
        if company_id in modeled_company_ids:
            if not cik.isdigit():
                raise ValueError(f"invalid SEC CIK for {company_id}: {cik!r}")
            existing = company_to_cik.get(company_id)
            if existing is not None and existing != cik:
                raise ValueError(f"modeled company {company_id} has multiple SEC CIKs")
            company_to_cik[company_id] = cik

    missing = sorted(modeled_company_ids - set(company_to_cik))
    if missing:
        raise RuntimeError(
            f"{len(missing)} modeled companies are absent from SEC company manifest; "
            f"examples={missing[:5]}"
        )
    return tuple(sorted(set(company_to_cik.values())))


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
            "Poll recent SEC Form 4 filings for the modeled universe, persist raw and "
            "normalized events, and enqueue them durably for a future PAPER/SHADOW cycle."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("data/experiments/lightgbm_holdout_250_v2"),
    )
    parser.add_argument("--runtime-dir", type=Path, default=Path("data/runtime"))
    parser.add_argument("--initial-lookback-days", type=int, default=7)
    parser.add_argument("--max-companies", type=int)
    parser.add_argument("--as-of")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.max_companies is not None and args.max_companies <= 0:
        raise ValueError("--max-companies must be > 0")
    as_of = _parse_as_of(args.as_of)
    user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not user_agent:
        raise RuntimeError(
            "SEC_USER_AGENT is required and must identify the application/contact"
        )

    ciks = _modeled_ciks(
        args.data_root / "manifests" / "sec_companies.jsonl",
        args.experiment_dir / "training_rows.jsonl",
    )
    if args.max_companies is not None:
        ciks = ciks[: args.max_companies]

    raw_store = FileRawStore(args.data_root / "raw")
    event_store = DuckDbEventStore(args.data_root / "normalized" / "events.duckdb")
    queue = FileCurrentEventQueue(args.runtime_dir / "current_event_intake.json")
    with SecClient(user_agent) as client:
        poll_result = SecCurrentForm4Poller(
            client=client,
            raw_store=raw_store,
            event_store=event_store,
            queue=queue,
            initial_lookback_days=args.initial_lookback_days,
        ).poll(ciks, as_of=as_of)

    resolver = XnysExecutionSessionResolver()
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
                "modeled_company_count": len(ciks),
                "poll": {
                    "submissions_fetched": poll_result.submissions_fetched,
                    "filings_committed": poll_result.filings_committed,
                    "events_normalized": poll_result.events_normalized,
                    "pending_events_added": poll_result.pending_events_added,
                    "pending_event_count": poll_result.pending_event_count,
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
