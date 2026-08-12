import json
from datetime import datetime, timezone
from decimal import Decimal

from stock_trading.core import RawRecord, Source, content_sha256
from stock_trading.contracts import UsaSpendingNormalizer
from stock_trading.lobbying import LdaFilingNormalizer


def _raw(source: Source, record_id: str, payload) -> RawRecord:
    content = json.dumps(payload)
    return RawRecord(
        source=source,
        source_record_id=record_id,
        fetched_at=datetime(2026, 8, 12, 20, 0, tzinfo=timezone.utc),
        content_type="application/json",
        content=content,
        sha256=content_sha256(content),
    )


def test_usaspending_transaction_keeps_award_context_and_safe_observed_time() -> None:
    normalizer = UsaSpendingNormalizer()
    award = normalizer.parse_award(
        _raw(
            Source.USASPENDING,
            "award:test",
            {
                "generated_unique_award_id": "CONT_AWD_TEST_9700",
                "type": "C",
                "type_description": "DELIVERY ORDER",
                "description": "MISSILE DEFENSE INTERCEPTOR SUPPORT",
                "total_obligation": 900000000,
                "base_and_all_options": 1200000000,
                "recipient": {
                    "recipient_name": "Example Defense Subsidiary LLC",
                    "recipient_uei": "SUBSIDIARYUEI",
                    "parent_recipient_name": "Example Defense Corp",
                    "parent_recipient_uei": "PARENTUEI",
                },
                "awarding_agency": {
                    "toptier_agency": {"name": "Department of Defense"},
                    "subtier_agency": {"name": "Missile Defense Agency"},
                },
                "latest_transaction_contract_data": {
                    "naics": "336414",
                    "product_or_service_code": "1410",
                },
            },
        )
    )
    transactions = _raw(
        Source.USASPENDING,
        "transactions:test",
        {
            "results": [
                {
                    "id": "tx-1",
                    "action_date": "2026-08-10",
                    "action_type": "A",
                    "action_type_description": "NEW AWARD",
                    "description": "Incremental interceptor procurement",
                    "federal_action_obligation": 250000000,
                    "modification_number": "P00001",
                    "type": "C",
                    "type_description": "DELIVERY ORDER",
                }
            ]
        },
    )
    observed_at = datetime(2026, 8, 12, 20, 5, tzinfo=timezone.utc)

    events = normalizer.to_events(
        transactions,
        award=award,
        observed_at=observed_at,
        company_ids_by_uei={"PARENTUEI": "cmp_parent"},
    )

    assert len(events) == 1
    event = events[0]
    assert event.company_id == "cmp_parent"
    assert event.public_time == observed_at
    assert event.payload.obligation_amount == Decimal("250000000")
    assert event.payload.total_obligation == Decimal("900000000")
    assert event.payload.agency == "Department of Defense"
    assert event.payload.subagency == "Missile Defense Agency"
    assert event.payload.naics_code == "336414"
    assert event.payload.psc_code == "1410"
    assert "interceptor" in normalizer.semantic_text(event).lower()


def test_usaspending_unresolved_recipient_does_not_guess_company() -> None:
    normalizer = UsaSpendingNormalizer()
    award = normalizer.parse_award(
        _raw(
            Source.USASPENDING,
            "award:test",
            {
                "generated_unique_award_id": "CONT_AWD_TEST_9700",
                "recipient": {
                    "recipient_name": "Unknown Vendor",
                    "recipient_uei": "UNKNOWNUEI",
                },
            },
        )
    )
    events = normalizer.to_events(
        _raw(
            Source.USASPENDING,
            "transactions:test",
            {
                "results": [
                    {
                        "id": "tx-1",
                        "action_date": "2026-08-10",
                        "federal_action_obligation": 100,
                    }
                ]
            },
        ),
        award=award,
        observed_at=datetime(2026, 8, 12, 20, 5, tzinfo=timezone.utc),
    )

    assert events[0].company_id is None


def test_lda_filing_uses_dt_posted_and_preserves_issue_details() -> None:
    raw = _raw(
        Source.LDA,
        "filings:2026:page=1",
        {
            "count": 1,
            "next": None,
            "previous": None,
            "results": [
                {
                    "filing_uuid": "62b1778e-e2e3-443d-a795-ca3813b6cee5",
                    "filing_type": "Q2",
                    "filing_year": 2026,
                    "filing_period": "second_quarter",
                    "dt_posted": "2026-07-20T18:15:22Z",
                    "income": "150000.00",
                    "expenses": None,
                    "client": {"id": 123, "name": "Example Semiconductor Corp"},
                    "registrant": {"id": 456, "name": "Example Lobbying LLC"},
                    "lobbying_activities": [
                        {
                            "general_issue_code": "TRD",
                            "description": "Export controls for advanced semiconductor equipment.",
                            "government_entities": [
                                {"id": 1, "name": "Department of Commerce"}
                            ],
                        }
                    ],
                }
            ],
        },
    )

    events = LdaFilingNormalizer().to_events(
        raw,
        company_ids_by_client_id={123: "cmp_chip"},
    )

    assert len(events) == 1
    event = events[0]
    assert event.company_id == "cmp_chip"
    assert event.public_time == datetime(2026, 7, 20, 18, 15, 22, tzinfo=timezone.utc)
    assert event.event_time == event.public_time
    assert event.payload.amount == Decimal("150000.00")
    assert event.payload.issue_codes == ("TRD",)
    assert event.payload.government_entities == ("Department of Commerce",)
    assert "Export controls" in event.payload.specific_issues[0]
    assert "Department of Commerce" in LdaFilingNormalizer.semantic_text(event)
