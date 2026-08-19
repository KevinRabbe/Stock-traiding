from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stock_trading.sec import OwnershipFiling, SubmissionsParser
from stock_trading.storage import DuckDbEventStore

from .current_cycle_receipt import FileCurrentCycleReceiptStore
from .pending_disposition import FileStaleTriggerDispositionStore


@dataclass(frozen=True, slots=True)
class FinalizedForm4Index:
    accessions: frozenset[str]
    receipt_event_count: int
    receipt_accession_count: int
    stale_accession_count: int


class FinalizedAwareSubmissionsParser:
    """Filter already-finalized Form 4 accessions from mutable SEC submissions.

    The SEC submissions document is mutable and filing cursor metadata is useful as
    a fast incremental boundary, but accession identity is the stronger source
    identity. Once every normalized event from an accession has either a completed
    cycle receipt or an explicit stale disposition, that accession must never be
    normalized/enqueued again even if a later submissions response repeats it or
    presents cursor metadata that would otherwise pass the watermark.

    The raw submissions response is still stored unchanged by the poller; filtering
    occurs only after strict parsing of that raw response.
    """

    def __init__(
        self,
        finalized_accessions: frozenset[str] | set[str],
        *,
        parser: SubmissionsParser | None = None,
    ) -> None:
        self.finalized_accessions = frozenset(
            str(item).strip() for item in finalized_accessions if str(item).strip()
        )
        self.parser = parser or SubmissionsParser()
        self.suppressed_replays = 0

    def recent_form4_filings(self, payload: dict) -> tuple[OwnershipFiling, ...]:
        filings = self.parser.recent_form4_filings(payload)
        kept: list[OwnershipFiling] = []
        for filing in filings:
            if filing.accession_number in self.finalized_accessions:
                self.suppressed_replays += 1
                continue
            kept.append(filing)
        return tuple(kept)


def load_finalized_form4_index(
    *,
    runtime_dir: str | Path,
    event_store: DuckDbEventStore,
) -> FinalizedForm4Index:
    """Build the durable finalized-accession set from receipts + stale audits.

    Receipts contain event IDs, so source accession identity is resolved directly
    from the normalized event store with a targeted DuckDB query. Stale dispositions
    already persist accession identity explicitly. Missing receipt events fail closed:
    a completed receipt whose source event disappeared must never silently weaken the
    idempotency boundary.
    """

    runtime_dir = Path(runtime_dir)
    receipts = FileCurrentCycleReceiptStore(
        runtime_dir / "current_cycle_receipts"
    ).load_all()
    receipt_event_ids = tuple(
        sorted(
            {
                event_id
                for receipt in receipts
                for event_id in receipt.selected_event_ids
            }
        )
    )
    receipt_accessions = _accessions_for_event_ids(event_store, receipt_event_ids)

    stale_records = FileStaleTriggerDispositionStore(
        runtime_dir / "stale_trigger_dispositions.json"
    ).load()
    stale_accessions = {
        item.accession_number.strip()
        for item in stale_records
        if item.accession_number.strip()
    }

    return FinalizedForm4Index(
        accessions=frozenset(receipt_accessions | stale_accessions),
        receipt_event_count=len(receipt_event_ids),
        receipt_accession_count=len(receipt_accessions),
        stale_accession_count=len(stale_accessions),
    )


def _accessions_for_event_ids(
    event_store: DuckDbEventStore,
    event_ids: tuple[str, ...],
) -> set[str]:
    if not event_ids:
        return set()

    accessions: set[str] = set()
    found_ids: set[str] = set()
    chunk_size = 500
    with event_store._connect() as connection:  # noqa: SLF001 - same storage boundary
        for offset in range(0, len(event_ids), chunk_size):
            chunk = event_ids[offset : offset + chunk_size]
            placeholders = ", ".join("?" for _ in chunk)
            rows = connection.execute(
                f"SELECT event_id, source_record_id, source FROM events "
                f"WHERE event_id IN ({placeholders})",
                list(chunk),
            ).fetchall()
            for event_id, source_record_id, source in rows:
                found_ids.add(str(event_id))
                if str(source) != "sec_edgar":
                    raise RuntimeError(
                        f"completed current receipt references non-SEC event {event_id}"
                    )
                accession = _accession_from_source_record_id(str(source_record_id))
                accessions.add(accession)

    missing = sorted(set(event_ids) - found_ids)
    if missing:
        raise RuntimeError(
            "completed current receipt events are missing from normalized storage: "
            f"{missing[:5]}"
        )
    return accessions


def _accession_from_source_record_id(source_record_id: str) -> str:
    marker = ":NONDERIV_TRANS:"
    accession, separator, suffix = source_record_id.partition(marker)
    if not separator or not accession.strip() or not suffix.isdigit():
        raise ValueError(
            f"unsupported current Form 4 source_record_id: {source_record_id}"
        )
    return accession.strip()
