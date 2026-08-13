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


def _raw(
    *,
    fetched_at: datetime | None = None,
    source_record_id: str = "filing-1",
) -> RawRecord:
    content = "<ownershipDocument />"
    return RawRecord(
        source=Source.SEC_EDGAR,
        source_record_id=source_record_id,
        fetched_at=fetched_at or datetime(2026, 8, 11, 15, 0, tzinfo=timezone.utc),
        content_type="application/xml",
        content=content,
        sha256=content_sha256(content),
    )


def _event(raw: RawRecord, *, index: int = 0) -> Event:
    source_record_id = f"filing-1:NONDERIV_TRANS:{index}"
    return Event(
        event_id=deterministic_event_id(
            Source.SEC_EDGAR,
            source_record_id,
            EventType.INSIDER_TRANSACTION,
            index,
        ),
        event_type=EventType.INSIDER_TRANSACTION,
        event_index=index,
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


def test_raw_store_accepts_same_artifact_refetched_later_and_can_resume(tmp_path) -> None:
    first = _raw(fetched_at=datetime(2026, 8, 11, 15, 0, tzinfo=timezone.utc))
    later_same_month = _raw(fetched_at=datetime(2026, 8, 13, 2, 0, tzinfo=timezone.utc))
    store = FileRawStore(tmp_path)

    first_path = store.put(first)
    second_path = store.put(later_same_month)

    assert first_path == second_path
    restored = store.latest(Source.SEC_EDGAR, "filing-1")
    assert restored is not None
    assert restored.sha256 == first.sha256
    assert restored.content == first.content.encode("utf-8")
    # Metadata remains immutable: a harmless later retrieval does not rewrite it.
    assert restored.fetched_at == first.fetched_at


def test_raw_store_keeps_source_identity_strict_for_same_content_hash(tmp_path) -> None:
    store = FileRawStore(tmp_path)
    store.put(_raw(source_record_id="filing-1"))

    with pytest.raises(ValueError, match="immutable raw artifact collision"):
        store.put(_raw(source_record_id="different-filing"))


def test_raw_store_latest_prefers_newest_stored_snapshot(tmp_path) -> None:
    august = _raw(fetched_at=datetime(2026, 8, 31, 23, 0, tzinfo=timezone.utc))
    september = _raw(fetched_at=datetime(2026, 9, 1, 1, 0, tzinfo=timezone.utc))
    store = FileRawStore(tmp_path)

    store.put(august)
    store.put(september)

    restored = store.latest(Source.SEC_EDGAR, "filing-1")
    assert restored is not None
    assert restored.fetched_at == september.fetched_at


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


def test_duckdb_event_store_bulk_insert_is_idempotent(tmp_path) -> None:
    pytest.importorskip("duckdb")

    raw = _raw()
    events = [_event(raw, index=index) for index in range(500)]
    store = DuckDbEventStore(tmp_path / "events.duckdb")

    store.put_many(events)
    assert store.count() == 500

    # A resumed historical import may replay an already committed quarter.
    store.put_many(events)
    assert store.count() == 500

    restored = store.all_events()
    assert len(restored) == 500
    assert restored[0].first_tradable_time is None
    assert restored[0].semantic is None
