from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from stock_trading.core import as_utc

from .event_intake import FileCurrentEventQueue, PendingBatchSelection, PendingTrigger
from .candidates import ExecutionSessionResolver


@dataclass(frozen=True, slots=True)
class StaleTriggerDisposition:
    event_id: str
    company_id: str
    cik: str
    accession_number: str
    public_time: datetime
    intended_execution_date: date
    observed_target_execution_date: date
    disposed_at: datetime
    reason: str = "missed_execution_session"

    def __post_init__(self) -> None:
        object.__setattr__(self, "public_time", as_utc(self.public_time))
        object.__setattr__(self, "disposed_at", as_utc(self.disposed_at))
        if not all(
            str(value).strip()
            for value in (
                self.event_id,
                self.company_id,
                self.cik,
                self.accession_number,
                self.reason,
            )
        ):
            raise ValueError("stale trigger disposition identity fields must not be empty")
        if self.intended_execution_date >= self.observed_target_execution_date:
            raise ValueError("stale trigger disposition does not describe an expired session")


class FileStaleTriggerDispositionStore:
    """Atomic idempotent audit of pending events removed because their open was missed."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> tuple[StaleTriggerDisposition, ...]:
        if not self.path.exists():
            return ()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid stale trigger disposition store: {self.path}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("unsupported stale trigger disposition schema")
        try:
            records = tuple(
                StaleTriggerDisposition(
                    event_id=str(item["event_id"]),
                    company_id=str(item["company_id"]),
                    cik=str(item["cik"]),
                    accession_number=str(item["accession_number"]),
                    public_time=datetime.fromisoformat(str(item["public_time"])),
                    intended_execution_date=date.fromisoformat(
                        str(item["intended_execution_date"])
                    ),
                    observed_target_execution_date=date.fromisoformat(
                        str(item["observed_target_execution_date"])
                    ),
                    disposed_at=datetime.fromisoformat(str(item["disposed_at"])),
                    reason=str(item.get("reason") or "missed_execution_session"),
                )
                for item in payload.get("records", ())
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid stale trigger disposition store: {self.path}") from exc
        ids = [item.event_id for item in records]
        if len(ids) != len(set(ids)):
            raise ValueError("stale trigger disposition store contains duplicate event IDs")
        return tuple(sorted(records, key=lambda item: (item.public_time, item.event_id)))

    def record_many(self, records: tuple[StaleTriggerDisposition, ...]) -> int:
        if not records:
            return 0
        existing = {item.event_id: item for item in self.load()}
        added = 0
        for item in records:
            previous = existing.get(item.event_id)
            if previous is not None:
                stable = (
                    previous.company_id == item.company_id
                    and previous.cik == item.cik
                    and previous.accession_number == item.accession_number
                    and previous.public_time == item.public_time
                    and previous.intended_execution_date == item.intended_execution_date
                    and previous.reason == item.reason
                )
                if not stable:
                    raise ValueError(f"stale disposition identity changed for {item.event_id}")
                continue
            existing[item.event_id] = item
            added += 1
        if added:
            self._save(tuple(existing.values()))
        return added

    def _save(self, records: tuple[StaleTriggerDisposition, ...]) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "records": [
                {
                    "event_id": item.event_id,
                    "company_id": item.company_id,
                    "cik": item.cik,
                    "accession_number": item.accession_number,
                    "public_time": item.public_time.isoformat(),
                    "intended_execution_date": item.intended_execution_date.isoformat(),
                    "observed_target_execution_date": (
                        item.observed_target_execution_date.isoformat()
                    ),
                    "disposed_at": item.disposed_at.isoformat(),
                    "reason": item.reason,
                }
                for item in sorted(records, key=lambda value: (value.public_time, value.event_id))
            ],
        }
        _atomic_json_write(self.path, payload)


@dataclass(frozen=True, slots=True)
class StaleDispositionResult:
    selected_count: int
    recorded_count: int
    removed_from_pending: int
    total_disposition_count: int


def dispose_stale_selection(
    *,
    queue: FileCurrentEventQueue,
    store: FileStaleTriggerDispositionStore,
    selection: PendingBatchSelection,
    session_resolver: ExecutionSessionResolver,
    disposed_at: datetime,
) -> StaleDispositionResult:
    """Durably audit stale pending IDs before removing them from the active queue."""

    stale_ids = tuple(selection.stale_event_ids)
    if not stale_ids:
        return StaleDispositionResult(
            selected_count=0,
            recorded_count=0,
            removed_from_pending=0,
            total_disposition_count=len(store.load()),
        )

    pending_by_id = {item.event_id: item for item in queue.pending()}
    records: list[StaleTriggerDisposition] = []
    missing: list[str] = []
    for event_id in stale_ids:
        pending = pending_by_id.get(event_id)
        if pending is None:
            # A previous crash can leave a durable disposition record after the
            # queue acknowledgement already completed. Treat that as idempotent.
            if any(item.event_id == event_id for item in store.load()):
                continue
            missing.append(event_id)
            continue
        records.append(
            _disposition_from_pending(
                pending,
                target=selection.target_execution_date,
                session_resolver=session_resolver,
                disposed_at=disposed_at,
            )
        )
    if missing:
        raise RuntimeError(f"stale pending triggers disappeared without audit: {missing[:5]}")

    recorded = store.record_many(tuple(records))
    # Audit is durable before queue removal. A crash between these operations is
    # safe because record_many and queue.acknowledge are independently idempotent.
    removed = queue.acknowledge(stale_ids)
    return StaleDispositionResult(
        selected_count=len(stale_ids),
        recorded_count=recorded,
        removed_from_pending=removed,
        total_disposition_count=len(store.load()),
    )


def _disposition_from_pending(
    pending: PendingTrigger,
    *,
    target: date,
    session_resolver: ExecutionSessionResolver,
    disposed_at: datetime,
) -> StaleTriggerDisposition:
    intended = session_resolver.execution_date(pending.public_time)
    return StaleTriggerDisposition(
        event_id=pending.event_id,
        company_id=pending.company_id,
        cik=pending.cik,
        accession_number=pending.accession_number,
        public_time=pending.public_time,
        intended_execution_date=intended,
        observed_target_execution_date=target,
        disposed_at=disposed_at,
    )


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
