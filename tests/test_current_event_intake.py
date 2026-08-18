from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from stock_trading.core import RawRecord, Source, content_sha256
from stock_trading.live.event_intake import (
    DurablePendingTriggerProvider,
    FileCurrentEventQueue,
    SecCurrentForm4Poller,
)
from stock_trading.live.session_calendar import XnysExecutionSessionResolver
from stock_trading.storage import DuckDbEventStore, FileRawStore


_CIK = "0000012345"
_ACCESSION = "0000012345-26-000011"


def _raw(source_record_id: str, content_type: str, content: str, at: datetime) -> RawRecord:
    return RawRecord(
        source=Source.SEC_EDGAR,
        source_record_id=source_record_id,
        fetched_at=at,
        content_type=content_type,
        content=content,
        sha256=content_sha256(content),
    )


def _submissions(accepted: str) -> dict:
    return {
        "cik": "12345",
        "filings": {
            "recent": {
                "form": ["4"],
                "accessionNumber": [_ACCESSION],
                "filingDate": ["2026-08-17"],
                "reportDate": ["2026-08-15"],
                "acceptanceDateTime": [accepted],
                "primaryDocument": ["ownership.xml"],
            }
        },
    }


def _form4_xml() -> str:
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
      <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""


class _SecClient:
    def __init__(self, accepted: str) -> None:
        self.accepted = accepted
        self.submission_calls = 0
        self.filing_calls = 0

    def fetch_submissions_raw(self, cik: str) -> RawRecord:
        assert cik == _CIK
        self.submission_calls += 1
        at = datetime(2026, 8, 17, 20, 31, tzinfo=timezone.utc)
        content = json.dumps(_submissions(self.accepted))
        return _raw(f"submissions:CIK{_CIK}", "application/json", content, at)

    def fetch_filing_xml(self, cik: str, accession_number: str, primary_document: str) -> RawRecord:
        assert cik == _CIK
        assert accession_number == _ACCESSION
        assert primary_document == "ownership.xml"
        self.filing_calls += 1
        return _raw(
            _ACCESSION,
            "application/xml",
            _form4_xml(),
            datetime(2026, 8, 17, 20, 31, 1, tzinfo=timezone.utc),
        )


def _poll_one(tmp_path):
    raw_store = FileRawStore(tmp_path / "raw")
    event_store = DuckDbEventStore(tmp_path / "events.duckdb")
    queue = FileCurrentEventQueue(tmp_path / "runtime" / "current_event_intake.json")
    client = _SecClient("2026-08-17T20:30:09Z")
    poller = SecCurrentForm4Poller(
        client=client,  # type: ignore[arg-type]
        raw_store=raw_store,
        event_store=event_store,
        queue=queue,
        initial_lookback_days=7,
    )
    poller.poll((_CIK,), as_of=datetime(2026, 8, 17, 21, 0, tzinfo=timezone.utc))
    return raw_store, event_store, queue, client, poller


def test_current_form4_poll_is_durable_and_idempotent(tmp_path) -> None:
    pytest.importorskip("duckdb")
    _, _, queue, client, poller = _poll_one(tmp_path)
    as_of = datetime(2026, 8, 17, 21, 0, tzinfo=timezone.utc)

    assert len(queue.pending()) == 1
    assert client.filing_calls == 1
    pending_id = queue.pending()[0].event_id
    assert queue.acknowledge((pending_id,)) == 1
    assert queue.pending() == ()

    # The filing watermark survives acknowledgement. Polling the same mutable
    # submissions document again must not requeue the already-consumed event.
    second = poller.poll((_CIK,), as_of=as_of)
    assert second.filings_committed == 0
    assert second.pending_events_added == 0
    assert second.pending_event_count == 0
    assert client.filing_calls == 1


def test_pending_provider_never_moves_stale_filing_to_later_session(tmp_path) -> None:
    pytest.importorskip("duckdb")
    _, event_store, queue, _, _ = _poll_one(tmp_path)
    provider = DurablePendingTriggerProvider(
        queue=queue,
        event_store=event_store,
        session_resolver=XnysExecutionSessionResolver(),
    )

    selected = provider.events(datetime(2026, 8, 17, 21, 0, tzinfo=timezone.utc))
    assert len(selected) == 1
    assert provider.last_selection is not None
    assert str(provider.last_selection.target_execution_date) == "2026-08-18"
    assert provider.last_selection.stale_event_ids == ()

    # After the Aug 18 open the same unacknowledged filing is stale. It stays
    # pending for explicit disposition and is NOT silently assigned Aug 19.
    selected_late = provider.events(datetime(2026, 8, 18, 21, 0, tzinfo=timezone.utc))
    assert selected_late == ()
    assert provider.last_selection is not None
    assert str(provider.last_selection.target_execution_date) == "2026-08-19"
    assert len(provider.last_selection.stale_event_ids) == 1
    assert len(queue.pending()) == 1


def test_preopen_restart_keeps_previous_day_filing_actionable(tmp_path) -> None:
    pytest.importorskip("duckdb")
    _, event_store, queue, _, _ = _poll_one(tmp_path)
    provider = DurablePendingTriggerProvider(
        queue=queue,
        event_store=event_store,
        session_resolver=XnysExecutionSessionResolver(),
    )

    # 11:00 UTC is 07:00 New York on Aug 18, before the 09:30 regular open.
    selected = provider.events(datetime(2026, 8, 18, 11, 0, tzinfo=timezone.utc))
    assert len(selected) == 1
    assert provider.last_selection is not None
    assert provider.last_selection.target_execution_date.isoformat() == "2026-08-18"
    assert provider.last_selection.stale_event_ids == ()

    # Once the open has passed, today's execution opportunity is no longer valid.
    selected_after_open = provider.events(
        datetime(2026, 8, 18, 14, 0, tzinfo=timezone.utc)
    )
    assert selected_after_open == ()
    assert provider.last_selection is not None
    assert provider.last_selection.target_execution_date.isoformat() == "2026-08-19"
    assert len(provider.last_selection.stale_event_ids) == 1


def test_xnys_resolver_handles_observed_independence_day_closure() -> None:
    resolver = XnysExecutionSessionResolver()
    # July 3, 2026 was the Friday observation of July 4 and XNYS was closed.
    publication = datetime(2026, 7, 2, 21, 0, tzinfo=timezone.utc)
    assert resolver.execution_date(publication).isoformat() == "2026-07-06"
