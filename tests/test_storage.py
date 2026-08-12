from datetime import datetime, timezone
from decimal import Decimal

import pytest

from stock_trading.core import (
    Event,
    EventType,
    InsiderTransactionPayload,
    RawRecord,
    Source,
    TradeDirection,
    content_sha256,
    deterministic_event_id,
)
from stock_trading.entities import company_id_from_sec_cik
from stock_trading.storage import DuckDbEventStore, FileRawStore


def _raw() -> RawRecord:
    content = "<ownershipDocument />"
    return RawRecord(
        source=Source.SEC_EDGAR,
        source_record_id="filing-1",
        fetched_at=datetime(2026, 8, 11, 15, 0, tzinfo=timezone.utc),
        content_type="application/xml",
        content=content,
        sha256=content_sha256(content),
    )


def _event(raw: RawRecord) -> Event:
    source_record_id = "filing-1:NONDERIV_TRANS:0"
    return Event(
        event_id=deterministic_event_id(
            Source.SEC_EDGAR,
            source_record_id,
            EventType.INSIDER_TRANSACTION,
        ),
        event_type=EventType.INSIDER_TRANSACTION,
        company_id=company_id_from_sec_cik("12345"),
        actor_id="sec_owner_cik_0000054321",
        event_time=datetime(2026, 8, 8, 4, 0, tzinfo=timezone.utc),
        public_time=datetime(2026, 8, 10, 20, 30, tzinfo=timezone.utc),
        first_tradable_time=None,
        source=Source.SEC_EDGAR,
        source_record_id=source_record_id,
        payload=InsiderTransactionPayload(
            source_transaction_code="P",
            direction=TradeDirection.BUY,
            shares=Decimal("100"),
            price=Decimal("10.50"),
            value=Decimal("1050"),
            intent_class="DISCRETIONARY_BUY",
        ),
        semantic=None,
        raw_artifact_id=raw.artifact_id,
        ingested_at=datetime(2026, 8, 10, 20, 31, tzinfo=timezone.utc),
    )


def test_raw_store_is_content_addressed_and_idempotent(tmp_path) -> None:
    raw = _raw()
    store = FileRawStore(tmp_path)

    first = store.put(raw)
    second = store.put(raw)

    assert first == second
    assert first.read_text() == raw.content
    assert first.with_name(f"{raw.artifact_id}.metadata.json").exists()


def test_duckdb_event_store_is_idempotent_and_point_in_time_safe(tmp_path) -> None:
    pytest.importorskip("duckdb")

    raw = _raw()
    event = _event(raw)
    store = DuckDbEventStore(tmp_path / "events.duckdb")

    store.put(event)
    store.put(event)
    assert store.count() == 1

    before = store.public_rows(
        event.company_id,
        datetime(2026, 8, 10, 20, 29, tzinfo=timezone.utc),
    )
    after = store.public_rows(
        event.company_id,
        datetime(2026, 8, 10, 20, 31, tzinfo=timezone.utc),
    )
    assert before == []
    assert len(after) == 1
    assert after[0]["event_id"] == event.event_id

    restored = store.all_events()
    assert restored == (event,)
    assert isinstance(restored[0].payload, InsiderTransactionPayload)
    assert store.all_events(event_types=(EventType.GOVERNMENT_CONTRACT,)) == ()
    assert store.all_events(event_types=(EventType.INSIDER_TRANSACTION,)) == (event,)

    parquet = store.export_parquet(tmp_path / "normalized" / "events.parquet")
    assert parquet.exists()
