from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from stock_trading.core import RawRecord, Source, content_sha256
from stock_trading.live.current_cycle_receipt import (
    CurrentCycleReceipt,
    FileCurrentCycleReceiptStore,
    batch_id,
)
from stock_trading.live.finalized_form4 import (
    FinalizedAwareSubmissionsParser,
    load_finalized_form4_index,
)
from stock_trading.live.pending_disposition import (
    FileStaleTriggerDispositionStore,
    StaleTriggerDisposition,
)
from stock_trading.sec import Form4XmlParser
from stock_trading.storage import DuckDbEventStore


UTC = timezone.utc
_CIK = "0000012345"
_RECEIPT_ACCESSION = "0000012345-26-000011"
_STALE_ACCESSION = "0000012345-26-000012"
_FRESH_ACCESSION = "0000012345-26-000013"


def _form4_xml() -> bytes:
    return b"""<?xml version="1.0"?>
<ownershipDocument>
  <documentType>4</documentType>
  <issuer><issuerCik>0000012345</issuerCik></issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0000054321</rptOwnerCik>
      <rptOwnerName>Example Owner</rptOwnerName>
    </reportingOwnerId>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-08-18</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>100</value></transactionShares>
        <transactionPricePerShare><value>10</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>1000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
      <ownershipNature>
        <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
      </ownershipNature>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""


def _receipt_event(accession: str):
    content = _form4_xml()
    accepted = datetime(2026, 8, 18, 20, 0, tzinfo=UTC)
    raw = RawRecord(
        source=Source.SEC_EDGAR,
        source_record_id=accession,
        fetched_at=accepted,
        content_type="application/xml",
        content=content,
        sha256=content_sha256(content),
    )
    return Form4XmlParser().to_events(
        raw,
        accepted_at=accepted,
        ingested_at=accepted,
    )[0]


def _submissions_payload() -> dict:
    # The receipt accession is intentionally presented with a later acceptance
    # timestamp than the normalized event. Accession identity must still suppress it.
    return {
        "cik": "12345",
        "filings": {
            "recent": {
                "form": ["4", "4/A", "4"],
                "accessionNumber": [
                    _RECEIPT_ACCESSION,
                    _STALE_ACCESSION,
                    _FRESH_ACCESSION,
                ],
                "filingDate": ["2026-08-19", "2026-08-19", "2026-08-19"],
                "reportDate": ["2026-08-18", "2026-08-18", "2026-08-18"],
                "acceptanceDateTime": [
                    "2026-08-19T14:00:00Z",
                    "2026-08-19T14:01:00Z",
                    "2026-08-19T14:02:00Z",
                ],
                "primaryDocument": ["a.xml", "b.xml", "c.xml"],
            }
        },
    }


def test_finalized_accessions_come_from_receipts_and_stale_audits(tmp_path) -> None:
    pytest.importorskip("duckdb")
    runtime_dir = tmp_path / "runtime"
    event_store = DuckDbEventStore(tmp_path / "events.duckdb")
    event = _receipt_event(_RECEIPT_ACCESSION)
    event_store.put(event)

    target = date(2026, 8, 19)
    selected_ids = (event.event_id,)
    receipt_id = batch_id(target, selected_ids)
    FileCurrentCycleReceiptStore(runtime_dir / "current_cycle_receipts").write(
        CurrentCycleReceipt(
            batch_id=receipt_id,
            completed_at=datetime(2026, 8, 18, 20, 5, tzinfo=UTC),
            target_execution_date=target,
            selected_event_ids=selected_ids,
            candidate_ids=(f"opportunity:{event.company_id}:2026-08-19",),
            champion_strategy_id="champion",
            champion_entry_order_ids=(),
            shadow_strategy_ids=("shadow",),
        )
    )
    FileStaleTriggerDispositionStore(
        runtime_dir / "stale_trigger_dispositions.json"
    ).record_many(
        (
            StaleTriggerDisposition(
                event_id="evt_stale",
                company_id=event.company_id or "company",
                cik=_CIK,
                accession_number=_STALE_ACCESSION,
                public_time=datetime(2026, 8, 18, 18, 0, tzinfo=UTC),
                intended_execution_date=date(2026, 8, 19),
                observed_target_execution_date=date(2026, 8, 20),
                disposed_at=datetime(2026, 8, 19, 14, 0, tzinfo=UTC),
            ),
        )
    )

    index = load_finalized_form4_index(
        runtime_dir=runtime_dir,
        event_store=event_store,
    )

    assert index.receipt_event_count == 1
    assert index.receipt_accession_count == 1
    assert index.stale_accession_count == 1
    assert index.accessions == frozenset({_RECEIPT_ACCESSION, _STALE_ACCESSION})


def test_finalized_parser_suppresses_exact_accession_even_if_cursor_changes() -> None:
    parser = FinalizedAwareSubmissionsParser(
        {_RECEIPT_ACCESSION, _STALE_ACCESSION}
    )

    filings = parser.recent_form4_filings(_submissions_payload())

    assert [item.accession_number for item in filings] == [_FRESH_ACCESSION]
    assert parser.suppressed_replays == 2
