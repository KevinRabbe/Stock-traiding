from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

from stock_trading.core import Event, Source, as_utc
from stock_trading.sec import Form4XmlParser, SecClient, SubmissionsParser
from stock_trading.storage import DuckDbEventStore, FileRawStore

from .candidates import ExecutionSessionResolver


@dataclass(frozen=True, order=True, slots=True)
class FilingCursor:
    accepted_at: datetime
    accession_number: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted_at", as_utc(self.accepted_at))
        if not self.accession_number.strip():
            raise ValueError("filing cursor accession_number must not be empty")


@dataclass(frozen=True, slots=True)
class PendingTrigger:
    event_id: str
    company_id: str
    public_time: datetime
    cik: str
    accession_number: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "public_time", as_utc(self.public_time))
        if not all(
            value.strip()
            for value in (
                self.event_id,
                self.company_id,
                self.cik,
                self.accession_number,
            )
        ):
            raise ValueError("pending trigger identity fields must not be empty")


@dataclass(frozen=True, slots=True)
class CurrentEventIntakeState:
    watermarks: dict[str, FilingCursor]
    pending: tuple[PendingTrigger, ...]


class FileCurrentEventQueue:
    """Atomic filing-watermark + pending-event state for current source intake.

    Normalized events are written to DuckDB before ``commit_filing``. The queue
    then atomically advances the CIK watermark and adds all normalized trigger IDs
    in one file replace. A crash between those two steps is safe: source/event
    ingestion is idempotent and the filing is replayed on the next poll.

    Acknowledgement only removes pending event IDs; filing watermarks remain, so a
    successfully processed filing can never be re-enqueued by a later SEC poll.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> CurrentEventIntakeState:
        if not self.path.exists():
            return CurrentEventIntakeState(watermarks={}, pending=())
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid current event queue at {self.path}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("unsupported current event queue schema")
        try:
            watermarks = {
                str(cik): FilingCursor(
                    accepted_at=datetime.fromisoformat(str(item["accepted_at"])),
                    accession_number=str(item["accession_number"]),
                )
                for cik, item in dict(payload.get("watermarks", {})).items()
            }
            pending = tuple(
                PendingTrigger(
                    event_id=str(item["event_id"]),
                    company_id=str(item["company_id"]),
                    public_time=datetime.fromisoformat(str(item["public_time"])),
                    cik=str(item["cik"]),
                    accession_number=str(item["accession_number"]),
                )
                for item in payload.get("pending", ())
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid current event queue at {self.path}") from exc
        event_ids = [item.event_id for item in pending]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("current event queue contains duplicate pending event IDs")
        return CurrentEventIntakeState(watermarks=watermarks, pending=pending)

    def watermark(self, cik: str) -> FilingCursor | None:
        return self.load().watermarks.get(_normalized_cik(cik))

    def commit_filing(
        self,
        *,
        cik: str,
        accession_number: str,
        accepted_at: datetime,
        events: Iterable[Event],
    ) -> int:
        normalized_cik = _normalized_cik(cik)
        cursor = FilingCursor(as_utc(accepted_at), accession_number)
        state = self.load()
        previous = state.watermarks.get(normalized_cik)
        if previous is not None and cursor <= previous:
            return 0

        pending = {item.event_id: item for item in state.pending}
        added = 0
        for event in events:
            if not event.company_id:
                continue
            if as_utc(event.public_time) != cursor.accepted_at:
                raise ValueError("normalized filing event public_time differs from SEC acceptance")
            item = PendingTrigger(
                event_id=event.event_id,
                company_id=event.company_id,
                public_time=event.public_time,
                cik=normalized_cik,
                accession_number=accession_number,
            )
            existing = pending.get(item.event_id)
            if existing is not None and existing != item:
                raise ValueError(f"pending event identity changed for {item.event_id}")
            if existing is None:
                pending[item.event_id] = item
                added += 1

        watermarks = dict(state.watermarks)
        watermarks[normalized_cik] = cursor
        self._save(
            CurrentEventIntakeState(
                watermarks=watermarks,
                pending=tuple(sorted(pending.values(), key=_pending_sort_key)),
            )
        )
        return added

    def pending(self, *, as_of: datetime | None = None) -> tuple[PendingTrigger, ...]:
        items = self.load().pending
        if as_of is None:
            return items
        cutoff = as_utc(as_of)
        return tuple(item for item in items if item.public_time <= cutoff)

    def acknowledge(self, event_ids: Iterable[str]) -> int:
        selected = {str(item) for item in event_ids if str(item)}
        if not selected:
            return 0
        state = self.load()
        kept = tuple(item for item in state.pending if item.event_id not in selected)
        removed = len(state.pending) - len(kept)
        if removed:
            self._save(CurrentEventIntakeState(state.watermarks, kept))
        return removed

    def _save(self, state: CurrentEventIntakeState) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "watermarks": {
                cik: {
                    "accepted_at": cursor.accepted_at.isoformat(),
                    "accession_number": cursor.accession_number,
                }
                for cik, cursor in sorted(state.watermarks.items())
            },
            "pending": [
                {
                    "event_id": item.event_id,
                    "company_id": item.company_id,
                    "public_time": item.public_time.isoformat(),
                    "cik": item.cik,
                    "accession_number": item.accession_number,
                }
                for item in sorted(state.pending, key=_pending_sort_key)
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=self.path.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class SecCurrentPollResult:
    company_count: int
    submissions_fetched: int
    filings_committed: int
    events_normalized: int
    pending_events_added: int
    pending_event_count: int


class SecCurrentForm4Poller:
    """Poll recent Form 4/4-A filings into raw, normalized and pending stores."""

    def __init__(
        self,
        *,
        client: SecClient,
        raw_store: FileRawStore,
        event_store: DuckDbEventStore,
        queue: FileCurrentEventQueue,
        initial_lookback_days: int = 7,
    ) -> None:
        if initial_lookback_days <= 0:
            raise ValueError("initial_lookback_days must be > 0")
        self.client = client
        self.raw_store = raw_store
        self.event_store = event_store
        self.queue = queue
        self.initial_lookback_days = initial_lookback_days
        self.submissions_parser = SubmissionsParser()
        self.form4_parser = Form4XmlParser()

    def poll(self, ciks: Iterable[str], *, as_of: datetime) -> SecCurrentPollResult:
        cutoff = as_utc(as_of)
        normalized_ciks = tuple(sorted({_normalized_cik(cik) for cik in ciks}))
        submissions_fetched = 0
        filings_committed = 0
        events_normalized = 0
        pending_added = 0

        for cik in normalized_ciks:
            submissions_raw = self.client.fetch_submissions_raw(cik)
            submissions_fetched += 1
            self.raw_store.put(submissions_raw)
            payload = _json_payload(submissions_raw.content)
            filings = tuple(
                sorted(
                    self.submissions_parser.recent_form4_filings(payload),
                    key=lambda item: (item.accepted_at, item.accession_number),
                )
            )
            previous = self.queue.watermark(cik)
            initial_floor = cutoff - timedelta(days=self.initial_lookback_days)

            for filing in filings:
                cursor = FilingCursor(filing.accepted_at, filing.accession_number)
                if cursor.accepted_at > cutoff:
                    continue
                if previous is not None:
                    if cursor <= previous:
                        continue
                elif cursor.accepted_at < initial_floor:
                    continue

                filing_raw = self.raw_store.latest(Source.SEC_EDGAR, filing.accession_number)
                if filing_raw is None:
                    filing_raw = self.client.fetch_filing_xml(
                        filing.cik,
                        filing.accession_number,
                        filing.primary_document,
                    )
                    self.raw_store.put(filing_raw)
                if filing_raw.content_type != "application/xml":
                    raise ValueError(
                        f"SEC filing raw artifact is not XML: {filing.accession_number}"
                    )
                events = self.form4_parser.to_events(
                    filing_raw,
                    accepted_at=filing.accepted_at,
                    ingested_at=filing_raw.fetched_at,
                )
                self.event_store.put_many(events)
                pending_added += self.queue.commit_filing(
                    cik=cik,
                    accession_number=filing.accession_number,
                    accepted_at=filing.accepted_at,
                    events=events,
                )
                events_normalized += len(events)
                filings_committed += 1
                previous = cursor

        return SecCurrentPollResult(
            company_count=len(normalized_ciks),
            submissions_fetched=submissions_fetched,
            filings_committed=filings_committed,
            events_normalized=events_normalized,
            pending_events_added=pending_added,
            pending_event_count=len(self.queue.pending()),
        )


@dataclass(frozen=True, slots=True)
class PendingBatchSelection:
    target_execution_date: object
    selected_event_ids: tuple[str, ...]
    stale_event_ids: tuple[str, ...]
    future_event_ids: tuple[str, ...]


class DurablePendingTriggerProvider:
    """Expose only pending events assigned to the current intended session.

    Each event's intended session is derived from its own public timestamp. This
    prevents a restarted process from silently moving yesterday's missed filing to
    a later open. Stale IDs remain pending until an explicit caller disposition.
    """

    def __init__(
        self,
        *,
        queue: FileCurrentEventQueue,
        event_store: DuckDbEventStore,
        session_resolver: ExecutionSessionResolver,
    ) -> None:
        self.queue = queue
        self.event_store = event_store
        self.session_resolver = session_resolver
        self.last_selection: PendingBatchSelection | None = None

    def events(self, as_of: datetime) -> tuple[Event, ...]:
        cutoff = as_utc(as_of)
        target = self.session_resolver.execution_date(cutoff)
        selected: list[PendingTrigger] = []
        stale: list[PendingTrigger] = []
        future: list[PendingTrigger] = []
        for item in self.queue.pending(as_of=cutoff):
            intended = self.session_resolver.execution_date(item.public_time)
            if intended < target:
                stale.append(item)
            elif intended > target:
                future.append(item)
            else:
                selected.append(item)

        selected_ids = tuple(item.event_id for item in selected)
        self.last_selection = PendingBatchSelection(
            target_execution_date=target,
            selected_event_ids=selected_ids,
            stale_event_ids=tuple(item.event_id for item in stale),
            future_event_ids=tuple(item.event_id for item in future),
        )
        if not selected:
            return ()
        company_ids = tuple(sorted({item.company_id for item in selected}))
        events = self.event_store.all_events(company_ids=company_ids)
        by_id = {event.event_id: event for event in events}
        missing = [event_id for event_id in selected_ids if event_id not in by_id]
        if missing:
            raise RuntimeError(f"pending normalized events are missing from event store: {missing[:5]}")
        return tuple(
            sorted(
                (by_id[event_id] for event_id in selected_ids),
                key=lambda event: (event.public_time, event.event_id),
            )
        )

    def acknowledge_selected(self) -> int:
        if self.last_selection is None:
            raise RuntimeError("no pending trigger selection has been made")
        return self.queue.acknowledge(self.last_selection.selected_event_ids)


def load_sec_company_ciks(path: str | Path) -> tuple[str, ...]:
    """Load the generated SEC company manifest without trusting ticker identity."""

    manifest = Path(path)
    ciks: set[str] = set()
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FileNotFoundError(f"missing SEC company manifest: {manifest}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            cik = _normalized_cik(str(item["sec_cik"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid SEC company manifest row {line_number}: {manifest}"
            ) from exc
        ciks.add(cik)
    return tuple(sorted(ciks))


def _normalized_cik(cik: str) -> str:
    value = cik.strip()
    if not value or not value.isdigit():
        raise ValueError(f"invalid SEC CIK: {cik!r}")
    return value.lstrip("0").zfill(10)


def _pending_sort_key(item: PendingTrigger) -> tuple[datetime, str]:
    return item.public_time, item.event_id


def _json_payload(content: bytes | str) -> dict:
    try:
        value = json.loads(content.decode("utf-8") if isinstance(content, bytes) else content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid SEC submissions JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("SEC submissions JSON must be an object")
    return value
