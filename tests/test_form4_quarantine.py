from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from stock_trading.core import RawRecord, Source, content_sha256
from stock_trading.live.event_intake import FileCurrentEventQueue, SecCurrentForm4Poller
from stock_trading.live.form4_quarantine import FileForm4Quarantine
from stock_trading.storage import DuckDbEventStore, FileRawStore


_CIK = "0000012345"
_BAD_ACCESSION = "0000012345-26-000010"
_GOOD_ACCESSION = "0000012345-26-000011"


def _raw(record_id: str, content_type: str, content: str, at: datetime) -> RawRecord:
    return RawRecord(
        source=Source.SEC_EDGAR,
        source_record_id=record_id,
        fetched_at=at,
        content_type=content_type,
        content=content,
        sha256=content_sha256(content),
    )


def _submissions() -> dict:
    return {
        "cik": "12345",
        "filings": {
            "recent": {
                "form": ["4", "4"],
                "accessionNumber": [_BAD_ACCESSION, _GOOD_ACCESSION],
                "filingDate": ["2026-08-17", "2026-08-17"],
                "reportDate": ["2026-08-15", "2026-08-15"],
                "acceptanceDateTime": [
                    "2026-08-17T20:20:00Z",
                    "2026-08-17T20:30:00Z",
                ],
                "primaryDocument": ["bad.xml", "good.xml"],
            }
        },
    }


def _valid_form4() -> str:
    return """<?xml version="1.0"?>
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


class _MixedSecClient:
    def __init__(self) -> None:
        self.filing_calls = 0

    def fetch_submissions_raw(self, cik: str) -> RawRecord:
        assert cik == _CIK
        content = json.dumps(_submissions())
        return _raw(
            f"submissions:CIK{_CIK}",
            "application/json",
            content,
            datetime(2026, 8, 17, 20, 31, tzinfo=timezone.utc),
        )

    def fetch_filing_xml(
        self,
        cik: str,
        accession_number: str,
        primary_document: str,
    ) -> RawRecord:
        assert cik == _CIK
        self.filing_calls += 1
        if accession_number == _BAD_ACCESSION:
            assert primary_document == "bad.xml"
            content = "<ownershipDocument><issuer></ownershipDocument>"
        else:
            assert accession_number == _GOOD_ACCESSION
            assert primary_document == "good.xml"
            content = _valid_form4()
        return _raw(
            accession_number,
            "application/xml",
            content,
            datetime(2026, 8, 17, 20, 31, 1, tzinfo=timezone.utc),
        )


def test_malformed_form4_is_quarantined_and_later_filing_continues(tmp_path) -> None:
    pytest.importorskip("duckdb")
    raw_store = FileRawStore(tmp_path / "raw")
    event_store = DuckDbEventStore(tmp_path / "events.duckdb")
    queue = FileCurrentEventQueue(tmp_path / "runtime" / "current_event_intake.json")
    quarantine = FileForm4Quarantine(tmp_path / "runtime" / "form4_quarantine.json")
    client = _MixedSecClient()
    poller = SecCurrentForm4Poller(
        client=client,  # type: ignore[arg-type]
        raw_store=raw_store,
        event_store=event_store,
        queue=queue,
        quarantine=quarantine,
    )
    as_of = datetime(2026, 8, 17, 21, 0, tzinfo=timezone.utc)

    result = poller.poll((_CIK,), as_of=as_of)

    assert result.filings_quarantined == 1
    assert result.filings_committed == 1
    assert result.events_normalized == 1
    assert result.pending_events_added == 1
    assert result.pending_event_count == 1
    assert result.quarantine_count == 1
    assert len(result.quarantined_filings) == 1
    quarantined = result.quarantined_filings[0]
    assert quarantined.accession_number == _BAD_ACCESSION
    assert quarantined.error_type == "ParseError"
    assert "mismatched tag" in quarantined.error_message
    assert raw_store.latest(Source.SEC_EDGAR, _BAD_ACCESSION) is not None
    assert event_store.count() == 1
    assert queue.watermark(_CIK) is not None
    assert queue.watermark(_CIK).accession_number == _GOOD_ACCESSION
    assert client.filing_calls == 2

    # The latest watermark covers both accessions. A restart must neither retry
    # the poison document nor duplicate the valid normalized event.
    second = poller.poll((_CIK,), as_of=as_of)
    assert second.filings_quarantined == 0
    assert second.filings_committed == 0
    assert second.pending_events_added == 0
    assert second.quarantine_count == 1
    assert client.filing_calls == 2


def test_quarantine_store_is_idempotent_for_same_raw_artifact(tmp_path) -> None:
    store = FileForm4Quarantine(tmp_path / "form4_quarantine.json")
    from stock_trading.live.form4_quarantine import QuarantinedForm4Filing

    item = QuarantinedForm4Filing(
        accepted_at=datetime(2026, 8, 17, 20, 20, tzinfo=timezone.utc),
        cik=_CIK,
        accession_number=_BAD_ACCESSION,
        raw_artifact_id="raw_0123456789abcdef0123456789abcdef",
        error_type="ParseError",
        error_message="mismatched tag",
    )
    assert store.record(item) is True
    assert store.record(item) is False
    assert store.load() == (item,)
