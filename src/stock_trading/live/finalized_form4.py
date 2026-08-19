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
    partial_accession_count: int = 0


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

    Receipts and stale dispositions are event-level authorities. An accession is
    terminal only when every normalized SEC event belonging to it is covered by one
    of those durable finalization records. This prevents one completed transaction
    from suppressing unfinished sibling transactions from the same Form 4 filing.

    Stale-only accessions with no normalized rows retain the previous behavior. That
    state is not produced by the live pipeline (stale records originate from the
    normalized pending queue), but preserving it keeps historical audit fixtures
    readable while completeness is enforced whenever normalized rows exist.
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
    finalized_event_ids = frozenset(receipt_event_ids) | frozenset(
        item.event_id for item in stale_records
    )
    accessions, partial_accession_count = _fully_finalized_accessions(
        event_store,
        candidate_accessions=receipt_accessions | stale_accessions,
        finalized_event_ids=finalized_event_ids,
        stale_accessions=stale_accessions,
    )

    return FinalizedForm4Index(
        accessions=frozenset(accessions),
        receipt_event_count=len(receipt_event_ids),
        receipt_accession_count=len(receipt_accessions),
        stale_accession_count=len(stale_accessions),
        partial_accession_count=partial_accession_count,
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


def _fully_finalized_accessions(
    event_store: DuckDbEventStore,
    *,
    candidate_accessions: set[str],
    finalized_event_ids: frozenset[str],
    stale_accessions: set[str],
) -> tuple[set[str], int]:
    if not candidate_accessions:
        return set(), 0

    finalized: set[str] = set()
    partial_count = 0
    with event_store._connect() as connection:  # noqa: SLF001 - same storage boundary
        for accession in sorted(candidate_accessions):
            rows = connection.execute(
                "SELECT event_id, source_record_id FROM events "
                "WHERE source = ? AND source_record_id LIKE ?",
                ["sec_edgar", f"{accession}:NONDERIV_TRANS:%"],
            ).fetchall()
            normalized_event_ids = {str(event_id) for event_id, _ in rows}

            if not normalized_event_ids:
                if accession in stale_accessions:
                    finalized.add(accession)
                    continue
                raise RuntimeError(
                    "completed current receipt accession is missing from normalized "
                    f"storage: {accession}"
                )

            for _, source_record_id in rows:
                actual_accession = _accession_from_source_record_id(
                    str(source_record_id)
                )
                if actual_accession != accession:
                    raise RuntimeError(
                        "normalized Form 4 accession query returned foreign event: "
                        f"expected={accession} actual={actual_accession}"
                    )

            if normalized_event_ids.issubset(finalized_event_ids):
                finalized.add(accession)
            else:
                partial_count += 1

    return finalized, partial_count


def _accession_from_source_record_id(source_record_id: str) -> str:
    marker = ":NONDERIV_TRANS:"
    accession, separator, suffix = source_record_id.partition(marker)
    if not separator or not accession.strip() or not suffix.isdigit():
        raise ValueError(
            f"unsupported current Form 4 source_record_id: {source_record_id}"
        )
    return accession.strip()
