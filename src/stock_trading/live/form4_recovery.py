from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable
from xml.etree import ElementTree

from stock_trading.core import Event, as_utc
from stock_trading.sec import Form4XmlParser, SecClient
from stock_trading.storage import DuckDbEventStore, FileRawStore

from .event_intake import (
    CurrentEventIntakeState,
    FileCurrentEventQueue,
    FilingCursor,
    PendingTrigger,
)
from .form4_quarantine import FileForm4Quarantine, QuarantinedForm4Filing


@dataclass(frozen=True, order=True, slots=True)
class RecoveredForm4Filing:
    accepted_at: datetime
    cik: str
    accession_number: str
    recovered_at: datetime
    original_raw_artifact_id: str
    recovered_raw_artifact_id: str
    event_ids: tuple[str, ...]
    pending_events_added: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted_at", as_utc(self.accepted_at))
        object.__setattr__(self, "recovered_at", as_utc(self.recovered_at))
        for name in (
            "cik",
            "accession_number",
            "original_raw_artifact_id",
            "recovered_raw_artifact_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"recovered filing {name} must not be empty")
            object.__setattr__(self, name, value)
        if self.pending_events_added < 0:
            raise ValueError("pending_events_added must be >= 0")


class FileForm4Recovery:
    """Atomic audit trail for Form 4 accessions recovered from quarantine."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> tuple[RecoveredForm4Filing, ...]:
        if not self.path.exists():
            return ()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid Form 4 recovery audit at {self.path}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("unsupported Form 4 recovery audit schema")
        try:
            records = tuple(
                RecoveredForm4Filing(
                    accepted_at=datetime.fromisoformat(str(item["accepted_at"])),
                    cik=str(item["cik"]),
                    accession_number=str(item["accession_number"]),
                    recovered_at=datetime.fromisoformat(str(item["recovered_at"])),
                    original_raw_artifact_id=str(item["original_raw_artifact_id"]),
                    recovered_raw_artifact_id=str(item["recovered_raw_artifact_id"]),
                    event_ids=tuple(str(value) for value in item.get("event_ids", ())),
                    pending_events_added=int(item.get("pending_events_added", 0)),
                )
                for item in payload.get("records", ())
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid Form 4 recovery audit at {self.path}") from exc
        identities = [(item.cik, item.accession_number) for item in records]
        if len(identities) != len(set(identities)):
            raise ValueError("Form 4 recovery audit contains duplicate accessions")
        return tuple(sorted(records))

    def get(self, cik: str, accession_number: str) -> RecoveredForm4Filing | None:
        key = (_normalized_cik(cik), str(accession_number).strip())
        return next(
            (
                item
                for item in self.load()
                if (item.cik, item.accession_number) == key
            ),
            None,
        )

    def record(self, item: RecoveredForm4Filing) -> bool:
        records = {(value.cik, value.accession_number): value for value in self.load()}
        key = (item.cik, item.accession_number)
        existing = records.get(key)
        if existing is not None:
            stable = (
                existing.accepted_at == item.accepted_at
                and existing.original_raw_artifact_id == item.original_raw_artifact_id
                and existing.recovered_raw_artifact_id == item.recovered_raw_artifact_id
                and existing.event_ids == item.event_ids
            )
            if not stable:
                raise ValueError(
                    f"recovered Form 4 identity changed for {item.accession_number}"
                )
            return False
        records[key] = item
        self._save(tuple(sorted(records.values())))
        return True

    def _save(self, records: tuple[RecoveredForm4Filing, ...]) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "records": [
                {
                    "accepted_at": item.accepted_at.isoformat(),
                    "cik": item.cik,
                    "accession_number": item.accession_number,
                    "recovered_at": item.recovered_at.isoformat(),
                    "original_raw_artifact_id": item.original_raw_artifact_id,
                    "recovered_raw_artifact_id": item.recovered_raw_artifact_id,
                    "event_ids": list(item.event_ids),
                    "pending_events_added": item.pending_events_added,
                }
                for item in records
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


class RecoverableCurrentEventQueue(FileCurrentEventQueue):
    """Add recovered events without ever rewinding an already-advanced watermark."""

    def enqueue_recovered_filing(
        self,
        *,
        cik: str,
        accession_number: str,
        accepted_at: datetime,
        events: Iterable[Event],
    ) -> int:
        normalized_cik = _normalized_cik(cik)
        cursor = FilingCursor(as_utc(accepted_at), accession_number)
        materialized = tuple(events)
        state = self.load()
        watermark = state.watermarks.get(normalized_cik)

        # If the original poll crashed before advancing its cursor, the normal
        # commit path is still authoritative and can advance it together with the
        # recovered pending events.
        if watermark is None or cursor > watermark:
            return self.commit_filing(
                cik=normalized_cik,
                accession_number=accession_number,
                accepted_at=accepted_at,
                events=materialized,
            )

        pending = {item.event_id: item for item in state.pending}
        added = 0
        for event in materialized:
            if not event.company_id:
                continue
            if as_utc(event.public_time) != cursor.accepted_at:
                raise ValueError(
                    "recovered normalized event public_time differs from SEC acceptance"
                )
            item = PendingTrigger(
                event_id=event.event_id,
                company_id=event.company_id,
                public_time=event.public_time,
                cik=normalized_cik,
                accession_number=accession_number,
            )
            existing = pending.get(item.event_id)
            if existing is not None and existing != item:
                raise ValueError(f"recovered pending event identity changed for {item.event_id}")
            if existing is None:
                pending[item.event_id] = item
                added += 1

        if added:
            self._save(
                CurrentEventIntakeState(
                    watermarks=state.watermarks,
                    pending=tuple(
                        sorted(
                            pending.values(),
                            key=lambda item: (item.public_time, item.event_id),
                        )
                    ),
                )
            )
        return added


@dataclass(frozen=True, slots=True)
class Form4RecoveryFailure:
    cik: str
    accession_number: str
    error_type: str
    error_message: str


@dataclass(frozen=True, slots=True)
class Form4RecoveryResult:
    attempted: int
    recovered: int
    already_recovered: int
    failed: int
    events_normalized: int
    pending_events_added: int
    unresolved_quarantine_count: int
    recovery_count: int
    recovered_filings: tuple[RecoveredForm4Filing, ...]
    failures: tuple[Form4RecoveryFailure, ...]


class Form4QuarantineRecovery:
    """Replay active quarantine records through SEC raw-XML discovery.

    A record leaves active quarantine only after the replacement raw XML, normalized
    events, pending-queue state, and recovery audit are all durable. The original
    bad raw artifact remains content-addressed in the raw store.
    """

    def __init__(
        self,
        *,
        client: SecClient,
        raw_store: FileRawStore,
        event_store: DuckDbEventStore,
        queue: RecoverableCurrentEventQueue,
        quarantine: FileForm4Quarantine,
        recovery: FileForm4Recovery,
    ) -> None:
        self.client = client
        self.raw_store = raw_store
        self.event_store = event_store
        self.queue = queue
        self.quarantine = quarantine
        self.recovery = recovery
        self.parser = Form4XmlParser()

    def recover(
        self,
        *,
        as_of: datetime | None = None,
        max_filings: int | None = None,
    ) -> Form4RecoveryResult:
        if max_filings is not None and max_filings <= 0:
            raise ValueError("max_filings must be > 0")
        cutoff = as_utc(as_of or datetime.now(timezone.utc))
        active = tuple(
            item for item in self.quarantine.load() if item.accepted_at <= cutoff
        )
        if max_filings is not None:
            active = active[:max_filings]

        recovered_this_run: list[RecoveredForm4Filing] = []
        failures: list[Form4RecoveryFailure] = []
        already_recovered = 0
        events_normalized = 0
        pending_added = 0

        for quarantined in active:
            existing = self.recovery.get(
                quarantined.cik,
                quarantined.accession_number,
            )
            if existing is not None:
                self.quarantine.resolve(
                    quarantined.cik,
                    quarantined.accession_number,
                )
                already_recovered += 1
                continue

            try:
                raw = self.client.fetch_filing_xml(
                    quarantined.cik,
                    quarantined.accession_number,
                    None,
                )
                if raw.content_type != "application/xml":
                    raise ValueError(
                        "recovered SEC filing artifact is not verified ownership XML"
                    )
                events = self.parser.to_events(
                    raw,
                    accepted_at=quarantined.accepted_at,
                    ingested_at=raw.fetched_at,
                )
                self.raw_store.put(raw)
                self.event_store.put_many(events)
                added = self.queue.enqueue_recovered_filing(
                    cik=quarantined.cik,
                    accession_number=quarantined.accession_number,
                    accepted_at=quarantined.accepted_at,
                    events=events,
                )
                recovered = RecoveredForm4Filing(
                    accepted_at=quarantined.accepted_at,
                    cik=_normalized_cik(quarantined.cik),
                    accession_number=quarantined.accession_number,
                    recovered_at=datetime.now(timezone.utc),
                    original_raw_artifact_id=quarantined.raw_artifact_id,
                    recovered_raw_artifact_id=raw.artifact_id,
                    event_ids=tuple(event.event_id for event in events),
                    pending_events_added=added,
                )
                self.recovery.record(recovered)
                self.quarantine.resolve(
                    quarantined.cik,
                    quarantined.accession_number,
                )
            except (ElementTree.ParseError, ValueError, OSError, RuntimeError) as exc:
                failures.append(
                    Form4RecoveryFailure(
                        cik=quarantined.cik,
                        accession_number=quarantined.accession_number,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                continue
            except Exception as exc:  # network/client errors remain visible per accession
                failures.append(
                    Form4RecoveryFailure(
                        cik=quarantined.cik,
                        accession_number=quarantined.accession_number,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                continue

            recovered_this_run.append(recovered)
            events_normalized += len(events)
            pending_added += added

        unresolved = self.quarantine.load()
        all_recovered = self.recovery.load()
        return Form4RecoveryResult(
            attempted=len(active),
            recovered=len(recovered_this_run),
            already_recovered=already_recovered,
            failed=len(failures),
            events_normalized=events_normalized,
            pending_events_added=pending_added,
            unresolved_quarantine_count=len(unresolved),
            recovery_count=len(all_recovered),
            recovered_filings=tuple(recovered_this_run),
            failures=tuple(failures),
        )


def _normalized_cik(cik: str) -> str:
    value = str(cik).strip()
    if not value or not value.isdigit():
        raise ValueError(f"invalid SEC CIK: {cik!r}")
    return value.lstrip("0").zfill(10)
