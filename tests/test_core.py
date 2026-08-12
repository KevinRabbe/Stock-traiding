from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from stock_trading.core import (
    Event,
    EventType,
    InsiderTransactionPayload,
    MarketBarPayload,
    RawRecord,
    SemanticAnnotation,
    SemanticDirection,
    Source,
    TradeDirection,
    as_utc,
    content_sha256,
    deterministic_event_id,
)


def _insider_event(**overrides) -> Event:
    source = Source.SEC_EDGAR
    source_record_id = "0000000000-26-000001"
    event_type = EventType.INSIDER_TRANSACTION
    event_index = 0

    values = {
        "event_id": deterministic_event_id(
            source, source_record_id, event_type, event_index
        ),
        "event_type": event_type,
        "event_index": event_index,
        "company_id": "cmp_test",
        "actor_id": "insider_test",
        "event_time": datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
        "public_time": datetime(2026, 8, 10, 22, 15, tzinfo=timezone.utc),
        "first_tradable_time": datetime(2026, 8, 11, 13, 30, tzinfo=timezone.utc),
        "source": source,
        "source_record_id": source_record_id,
        "payload": InsiderTransactionPayload(
            source_transaction_code="P",
            direction=TradeDirection.BUY,
            shares=Decimal("1000"),
            price=Decimal("25.50"),
            value=Decimal("25500"),
            insider_role="CEO",
            is_10b5_1=False,
        ),
        "semantic": None,
        "raw_artifact_id": "raw_0123456789abcdef0123456789abcdef",
        "ingested_at": datetime(2026, 8, 10, 22, 16, tzinfo=timezone.utc),
    }
    values.update(overrides)
    return Event(**values)


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        as_utc(datetime(2026, 8, 11, 12, 0))


def test_datetime_is_normalized_to_utc() -> None:
    cet = timezone(timedelta(hours=2))
    value = datetime(2026, 8, 11, 17, 30, tzinfo=cet)
    assert as_utc(value) == datetime(2026, 8, 11, 15, 30, tzinfo=timezone.utc)


def test_raw_record_verifies_content_hash_and_is_stable() -> None:
    content = "<ownershipDocument />"
    digest = content_sha256(content)
    raw = RawRecord(
        source=Source.SEC_EDGAR,
        source_record_id="filing-1",
        fetched_at=datetime.now(timezone.utc),
        content_type="application/xml",
        content=content,
        sha256=digest,
    )

    assert raw.sha256 == digest
    assert raw.artifact_id.startswith("raw_")

    with pytest.raises(ValidationError, match="sha256 does not match raw content"):
        RawRecord(
            source=Source.SEC_EDGAR,
            source_record_id="filing-1",
            fetched_at=datetime.now(timezone.utc),
            content_type="application/xml",
            content=content,
            sha256="0" * 64,
        )


def test_event_id_is_deterministic() -> None:
    first = deterministic_event_id(
        Source.SEC_EDGAR,
        "filing-1",
        EventType.INSIDER_TRANSACTION,
        3,
    )
    second = deterministic_event_id(
        Source.SEC_EDGAR,
        "filing-1",
        EventType.INSIDER_TRANSACTION,
        3,
    )
    different = deterministic_event_id(
        Source.SEC_EDGAR,
        "filing-1",
        EventType.INSIDER_TRANSACTION,
        4,
    )

    assert first == second
    assert first != different


def test_event_rejects_non_deterministic_id() -> None:
    with pytest.raises(ValidationError, match="event_id is not deterministic"):
        _insider_event(event_id="evt_wrong")


def test_event_rejects_payload_type_mismatch() -> None:
    market_payload = MarketBarPayload(
        open=Decimal("10"),
        high=Decimal("11"),
        low=Decimal("9"),
        close=Decimal("10.50"),
        volume=Decimal("100000"),
    )

    with pytest.raises(ValidationError, match="requires payload"):
        _insider_event(payload=market_payload)


def test_event_rejects_tradable_time_before_publication() -> None:
    public_time = datetime(2026, 8, 10, 22, 15, tzinfo=timezone.utc)
    with pytest.raises(ValidationError, match="cannot precede public_time"):
        _insider_event(
            public_time=public_time,
            first_tradable_time=public_time - timedelta(minutes=1),
        )


def test_public_visibility_and_tradability_are_separate() -> None:
    event = _insider_event()

    before_public = event.public_time - timedelta(seconds=1)
    after_public = event.public_time + timedelta(seconds=1)
    after_tradable = event.first_tradable_time + timedelta(seconds=1)

    assert not event.is_public_at(before_public)
    assert event.is_public_at(after_public)
    assert not event.is_tradable_at(after_public)
    assert event.is_tradable_at(after_tradable)


def test_semantic_annotation_cannot_replace_authoritative_payload() -> None:
    annotation = SemanticAnnotation(
        topics=("TECH.SEMICONDUCTORS",),
        direction=SemanticDirection.POSITIVE,
        novelty=0.8,
        importance=0.9,
        company_relevance=1.0,
        confidence=0.7,
        model="Qwen3.5-4B",
        extractor_version="v1",
        schema_version="semantic-v1",
    )
    event = _insider_event(semantic=annotation)

    assert event.payload.price == Decimal("25.50")
    assert event.semantic.importance == 0.9
    assert not hasattr(event.semantic, "price")


def test_event_is_immutable() -> None:
    event = _insider_event()
    with pytest.raises(ValidationError):
        event.company_id = "cmp_changed"
