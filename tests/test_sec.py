import io
import zipfile
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from stock_trading.core import RawRecord, Source, TradeDirection, content_sha256
from stock_trading.entities import company_id_from_sec_cik
from stock_trading.sec import Form4XmlParser, QuarterlyArchiveParser, SecClient


def _quarterly_zip() -> bytes:
    submission = (
        "ACCESSION_NUMBER\tFILING_DATE\tPERIOD_OF_REPORT\tDOCUMENT_TYPE\tISSUERCIK\t"
        "ISSUERNAME\tISSUERTRADINGSYMBOL\tAFF10B5ONE\n"
        "0000000001-26-000001\t10-AUG-2026\t08-AUG-2026\t4\t12345\tExample Corp\tEXM\t1\n"
        "0000000001-26-000002\t10-AUG-2026\t08-AUG-2026\t3\t99999\tIgnore Corp\tIGN\t0\n"
    )
    owners = (
        "ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNERNAME\tRPTOWNER_RELATIONSHIP\tRPTOWNER_TITLE\n"
        "0000000001-26-000001\t54321\tJane Doe\tOFFICER\tChief Executive Officer\n"
    )
    transactions = (
        "ACCESSION_NUMBER\tNONDERIV_TRANS_SK\tSECURITY_TITLE\tTRANS_DATE\tTRANS_CODE\t"
        "TRANS_SHARES\tTRANS_PRICEPERSHARE\tTRANS_ACQUIRED_DISP_CD\t"
        "SHRS_OWND_FOLWNG_TRANS\tDIRECT_INDIRECT_OWNERSHIP\tNATURE_OF_OWNERSHIP\n"
        "0000000001-26-000001\t101\tCommon Stock\t08-AUG-2026\tP\t100\t10.50\tA\t1100\tD\t\n"
        "0000000001-26-000002\t102\tCommon Stock\t08-AUG-2026\tP\t50\t5.00\tA\t50\tD\t\n"
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("SUBMISSION.tsv", submission)
        archive.writestr("REPORTINGOWNER.tsv", owners)
        archive.writestr("NONDERIV_TRANS.tsv", transactions)
    return buffer.getvalue()


def _raw(source: Source, record_id: str, content: bytes | str, content_type: str) -> RawRecord:
    return RawRecord(
        source=source,
        source_record_id=record_id,
        fetched_at=datetime(2026, 8, 11, 15, 0, tzinfo=timezone.utc),
        content_type=content_type,
        content=content,
        sha256=content_sha256(content),
    )


def test_quarterly_parser_filters_to_form4_and_builds_safe_event() -> None:
    archive = _raw(
        Source.SEC_QUARTERLY,
        "2026Q3",
        _quarterly_zip(),
        "application/zip",
    )
    parser = QuarterlyArchiveParser()

    rows = parser.parse(archive.content)
    assert len(rows) == 1
    assert rows[0].issuer_cik == "0000012345"
    assert rows[0].reporting_owners[0].cik == "0000054321"

    events = parser.to_events(
        archive,
        ingested_at=datetime(2026, 8, 11, 15, 30, tzinfo=timezone.utc),
    )
    assert len(events) == 1

    event = events[0]
    assert event.company_id == company_id_from_sec_cik("12345")
    assert event.actor_id == "sec_owner_cik_0000054321"
    assert event.first_tradable_time is None
    assert event.payload.direction is TradeDirection.BUY
    assert event.payload.intent_class == "DISCRETIONARY_BUY"
    assert event.payload.value == Decimal("1050.00")
    assert event.payload.is_10b5_1 is True
    assert event.public_time.date().isoformat() == "2026-08-11"  # UTC conversion of ET day-end


def test_quarterly_parser_rejects_naive_ingestion_timestamp() -> None:
    archive = _raw(
        Source.SEC_QUARTERLY,
        "2026Q3",
        _quarterly_zip(),
        "application/zip",
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        QuarterlyArchiveParser().to_events(
            archive,
            ingested_at=datetime(2026, 8, 11, 15, 30),
        )


def test_live_form4_xml_uses_exact_acceptance_time() -> None:
    xml = """<?xml version="1.0"?>
<ownershipDocument>
  <documentType>4</documentType>
  <aff10b5One>1</aff10b5One>
  <issuer>
    <issuerCik>12345</issuerCik>
    <issuerName>Example Corp</issuerName>
    <issuerTradingSymbol>EXM</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>54321</rptOwnerCik>
      <rptOwnerName>Jane Doe</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>0</isDirector>
      <isOfficer>1</isOfficer>
      <isTenPercentOwner>0</isTenPercentOwner>
      <isOther>0</isOther>
      <officerTitle>Chief Executive Officer</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-08-08</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>50</value></transactionShares>
        <transactionPricePerShare><value>24.60</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>500</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
      <ownershipNature>
        <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
      </ownershipNature>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""
    raw = _raw(Source.SEC_EDGAR, "0000000001-26-000001", xml, "application/xml")
    accepted = datetime(2026, 8, 10, 20, 30, 9, tzinfo=timezone.utc)

    events = Form4XmlParser().to_events(
        raw,
        accepted_at=accepted,
        ingested_at=accepted.replace(second=10),
    )

    assert len(events) == 1
    event = events[0]
    assert event.company_id == company_id_from_sec_cik("12345")
    assert event.public_time == accepted
    assert event.payload.direction is TradeDirection.BUY
    assert event.payload.intent_class == "DISCRETIONARY_BUY"
    assert event.payload.value == Decimal("1230.00")
    assert event.payload.is_10b5_1 is True
    assert event.actor_id == "sec_owner_cik_0000054321"
    assert "Chief Executive Officer" in event.payload.insider_role


def test_non_form4_xml_is_ignored() -> None:
    xml = "<ownershipDocument><documentType>3</documentType></ownershipDocument>"
    raw = _raw(Source.SEC_EDGAR, "filing", xml, "application/xml")
    now = datetime(2026, 8, 11, 15, 0, tzinfo=timezone.utc)
    assert Form4XmlParser().to_events(raw, accepted_at=now, ingested_at=now) == ()


def test_sec_client_builds_current_official_paths_and_caps_rate() -> None:
    assert SecClient.quarterly_archive_url(2026, 2).endswith("2026q2_form345.zip")
    assert SecClient.submissions_url("12345").endswith("CIK0000012345.json")
    assert SecClient.filing_document_url(
        "12345",
        "0000000001-26-000001",
        "ownership.xml",
    ).endswith("/12345/000000000126000001/ownership.xml")

    with pytest.raises(ValueError, match="10"):
        SecClient("test contact@example.com", max_requests_per_second=11)
