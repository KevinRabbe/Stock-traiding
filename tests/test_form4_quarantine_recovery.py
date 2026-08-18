from __future__ import annotations

from datetime import datetime, timezone

import pytest

from stock_trading.core import RawRecord, Source, content_sha256
from stock_trading.live.event_intake import FileCurrentEventQueue
from stock_trading.live.form4_quarantine import (
    FileForm4Quarantine,
    QuarantinedForm4Filing,
)
from stock_trading.live.form4_recovery import (
    FileForm4Recovery,
    Form4QuarantineRecovery,
    RecoverableCurrentEventQueue,
)
from stock_trading.storage import DuckDbEventStore, FileRawStore


_CIK = "0000012345"
_ACCESSION = "0000012345-26-000011"
_LATER_ACCESSION = "0000012345-26-000012"
_ACCEPTED = datetime(2026, 8, 17, 20, 30, 9, tzinfo=timezone.utc)
_LATER = datetime(2026, 8, 17, 21, 30, 9, tzinfo=timezone.utc)


def _ownership_xml() -> bytes:
    return b"""<?xml version=\"1.0\"?>
<ownershipDocument>
  <documentType>4</documentType>
  <issuer><issuerCik>0000012345</issuerCik></issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0000054321</rptOwnerCik>
      <rptOwnerName>Example Owner</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isOfficer>1</isOfficer>
      <officerTitle>Chief Executive Officer</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-08-15</value></transactionDate>
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


def _raw(content: bytes, content_type: str, fetched_at: datetime) -> RawRecord:
    return RawRecord(
        source=Source.SEC_EDGAR,
        source_record_id=_ACCESSION,
        fetched_at=fetched_at,
        content_type=content_type,
        content=content,
        sha256=content_sha256(content),
    )


class _RecoveryClient:
    def __init__(self, raw: RawRecord | None = None, error: Exception | None = None) -> None:
        self.raw = raw
        self.error = error
        self.calls = 0

    def fetch_filing_xml(self, cik: str, accession_number: str, primary_document=None):
        assert cik == _CIK
        assert accession_number == _ACCESSION
        assert primary_document is None
        self.calls += 1
        if self.error is not None:
            raise self.error
        assert self.raw is not None
        return self.raw


def _stores(tmp_path):
    pytest.importorskip("duckdb")
    raw_store = FileRawStore(tmp_path / "raw")
    event_store = DuckDbEventStore(tmp_path / "events.duckdb")
    queue = RecoverableCurrentEventQueue(tmp_path / "runtime" / "current_event_intake.json")
    quarantine = FileForm4Quarantine(tmp_path / "runtime" / "form4_quarantine.json")
    recovery = FileForm4Recovery(tmp_path / "runtime" / "form4_quarantine_recovery.json")
    return raw_store, event_store, queue, quarantine, recovery


def _seed_quarantine(raw_store, queue, quarantine) -> str:
    bad = _raw(
        b"<html><body><broken></body></html>",
        "text/html",
        datetime(2026, 8, 17, 20, 31, tzinfo=timezone.utc),
    )
    raw_store.put(bad)
    quarantine.record(
        QuarantinedForm4Filing(
            accepted_at=_ACCEPTED,
            cik=_CIK,
            accession_number=_ACCESSION,
            raw_artifact_id=bad.artifact_id,
            error_type="ParseError",
            error_message="mismatched tag",
        )
    )
    # The original quarantine path advanced this filing's watermark, and a later
    # filing may have advanced it again. Recovery must not rewind it.
    queue.commit_filing(
        cik=_CIK,
        accession_number=_ACCESSION,
        accepted_at=_ACCEPTED,
        events=(),
    )
    queue.commit_filing(
        cik=_CIK,
        accession_number=_LATER_ACCESSION,
        accepted_at=_LATER,
        events=(),
    )
    return bad.artifact_id


def test_recovery_restores_events_without_rewinding_watermark(tmp_path) -> None:
    raw_store, event_store, queue, quarantine, recovery = _stores(tmp_path)
    original_artifact_id = _seed_quarantine(raw_store, queue, quarantine)
    good = _raw(
        _ownership_xml(),
        "application/xml",
        datetime(2026, 8, 18, 8, 0, tzinfo=timezone.utc),
    )
    client = _RecoveryClient(good)

    result = Form4QuarantineRecovery(
        client=client,  # type: ignore[arg-type]
        raw_store=raw_store,
        event_store=event_store,
        queue=queue,
        quarantine=quarantine,
        recovery=recovery,
    ).recover(as_of=datetime(2026, 8, 18, 8, 30, tzinfo=timezone.utc))

    assert result.attempted == 1
    assert result.recovered == 1
    assert result.failed == 0
    assert result.events_normalized == 1
    assert result.pending_events_added == 1
    assert result.unresolved_quarantine_count == 0
    assert client.calls == 1
    assert quarantine.load() == ()

    watermark = queue.watermark(_CIK)
    assert watermark is not None
    assert watermark.accession_number == _LATER_ACCESSION
    assert watermark.accepted_at == _LATER
    pending = queue.pending()
    assert len(pending) == 1
    assert pending[0].accession_number == _ACCESSION
    assert len(event_store.all_events()) == 1

    audit = recovery.load()
    assert len(audit) == 1
    assert audit[0].original_raw_artifact_id == original_artifact_id
    assert audit[0].recovered_raw_artifact_id == good.artifact_id
    assert len(audit[0].event_ids) == 1

    # With the active quarantine cleared, a rerun is a no-op rather than a duplicate.
    second = Form4QuarantineRecovery(
        client=client,  # type: ignore[arg-type]
        raw_store=raw_store,
        event_store=event_store,
        queue=queue,
        quarantine=quarantine,
        recovery=recovery,
    ).recover(as_of=datetime(2026, 8, 18, 8, 31, tzinfo=timezone.utc))
    assert second.attempted == 0
    assert second.recovery_count == 1
    assert len(queue.pending()) == 1


def test_recovery_failure_keeps_accession_quarantined(tmp_path) -> None:
    raw_store, event_store, queue, quarantine, recovery = _stores(tmp_path)
    _seed_quarantine(raw_store, queue, quarantine)
    client = _RecoveryClient(error=ValueError("no ownership XML found"))

    result = Form4QuarantineRecovery(
        client=client,  # type: ignore[arg-type]
        raw_store=raw_store,
        event_store=event_store,
        queue=queue,
        quarantine=quarantine,
        recovery=recovery,
    ).recover(as_of=datetime(2026, 8, 18, 8, 30, tzinfo=timezone.utc))

    assert result.recovered == 0
    assert result.failed == 1
    assert result.unresolved_quarantine_count == 1
    assert len(quarantine.load()) == 1
    assert recovery.load() == ()
    assert event_store.all_events() == ()
    assert queue.pending() == ()
