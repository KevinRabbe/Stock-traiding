from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from stock_trading.core import RawRecord, Source, content_sha256
from stock_trading.live.current_cycle_receipt import (
    CurrentCycleReceipt,
    FileCurrentCycleReceiptStore,
    batch_id,
)
from stock_trading.live.finalized_form4 import load_finalized_form4_index
from stock_trading.live.pending_disposition import (
    FileStaleTriggerDispositionStore,
    StaleTriggerDisposition,
)
from stock_trading.sec import Form4XmlParser
from stock_trading.storage import DuckDbEventStore


UTC = timezone.utc
_CIK = "0000012345"
_ACCESSION = "0000012345-26-000099"


def _two_transaction_events():
    transaction = """
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
"""
    content = f"""<?xml version="1.0"?>
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
{transaction}{transaction}  </nonDerivativeTable>
</ownershipDocument>
""".encode("utf-8")
    accepted = datetime(2026, 8, 18, 20, 0, tzinfo=UTC)
    raw = RawRecord(
        source=Source.SEC_EDGAR,
        source_record_id=_ACCESSION,
        fetched_at=accepted,
        content_type="application/xml",
        content=content,
        sha256=content_sha256(content),
    )
    return Form4XmlParser().to_events(
        raw,
        accepted_at=accepted,
        ingested_at=accepted,
    )


def test_accession_requires_every_normalized_event_to_be_finalized(tmp_path) -> None:
    pytest.importorskip("duckdb")
    runtime_dir = tmp_path / "runtime"
    event_store = DuckDbEventStore(tmp_path / "events.duckdb")
    events = _two_transaction_events()
    assert len(events) == 2
    event_store.put_many(events)

    target = date(2026, 8, 19)
    selected_ids = (events[0].event_id,)
    receipt_id = batch_id(target, selected_ids)
    FileCurrentCycleReceiptStore(runtime_dir / "current_cycle_receipts").write(
        CurrentCycleReceipt(
            batch_id=receipt_id,
            completed_at=datetime(2026, 8, 18, 20, 5, tzinfo=UTC),
            target_execution_date=target,
            selected_event_ids=selected_ids,
            candidate_ids=(f"opportunity:{events[0].company_id}:2026-08-19",),
            champion_strategy_id="champion",
            champion_entry_order_ids=(),
            shadow_strategy_ids=("shadow",),
        )
    )

    partial = load_finalized_form4_index(
        runtime_dir=runtime_dir,
        event_store=event_store,
    )

    assert partial.receipt_accession_count == 1
    assert partial.partial_accession_count == 1
    assert _ACCESSION not in partial.accessions

    FileStaleTriggerDispositionStore(
        runtime_dir / "stale_trigger_dispositions.json"
    ).record_many(
        (
            StaleTriggerDisposition(
                event_id=events[1].event_id,
                company_id=events[1].company_id or "company",
                cik=_CIK,
                accession_number=_ACCESSION,
                public_time=events[1].public_time,
                intended_execution_date=date(2026, 8, 19),
                observed_target_execution_date=date(2026, 8, 20),
                disposed_at=datetime(2026, 8, 19, 14, 0, tzinfo=UTC),
            ),
        )
    )

    completed = load_finalized_form4_index(
        runtime_dir=runtime_dir,
        event_store=event_store,
    )

    assert completed.partial_accession_count == 0
    assert completed.accessions == frozenset({_ACCESSION})
