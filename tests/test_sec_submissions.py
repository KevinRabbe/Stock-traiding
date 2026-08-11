from datetime import datetime, timezone

import pytest

from stock_trading.sec import SubmissionsParser, parse_sec_acceptance_time


def test_submissions_parser_filters_form4_and_preserves_acceptance_time() -> None:
    payload = {
        "cik": "12345",
        "filings": {
            "recent": {
                "form": ["10-K", "4", "4/A"],
                "accessionNumber": [
                    "0000000001-26-000010",
                    "0000000001-26-000011",
                    "0000000001-26-000012",
                ],
                "filingDate": ["2026-08-10", "2026-08-10", "2026-08-11"],
                "reportDate": ["2026-06-30", "2026-08-08", "2026-08-08"],
                "acceptanceDateTime": [
                    "2026-08-10T12:00:00.000Z",
                    "2026-08-10T20:30:09.000Z",
                    "2026-08-11T14:15:01+00:00",
                ],
                "primaryDocument": ["annual.htm", "ownership.xml", "amended.xml"],
            }
        },
    }

    filings = SubmissionsParser().recent_form4_filings(payload)

    assert len(filings) == 2
    assert filings[0].cik == "0000012345"
    assert filings[0].form == "4"
    assert filings[0].accepted_at == datetime(
        2026, 8, 10, 20, 30, 9, tzinfo=timezone.utc
    )
    assert filings[0].primary_document == "ownership.xml"
    assert filings[1].is_amendment is True


def test_submissions_parser_rejects_column_length_mismatch() -> None:
    payload = {
        "cik": "12345",
        "filings": {
            "recent": {
                "form": ["4"],
                "accessionNumber": [],
                "filingDate": ["2026-08-10"],
                "reportDate": ["2026-08-08"],
                "acceptanceDateTime": ["2026-08-10T20:30:09.000Z"],
                "primaryDocument": ["ownership.xml"],
            }
        },
    }

    with pytest.raises(ValueError, match="inconsistent lengths"):
        SubmissionsParser().recent_form4_filings(payload)


def test_acceptance_time_requires_explicit_timezone() -> None:
    with pytest.raises(ValueError, match="include a timezone"):
        parse_sec_acceptance_time("2026-08-10T20:30:09.000")
