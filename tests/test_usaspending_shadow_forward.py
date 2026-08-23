from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import pytest

from stock_trading.contracts import UsaSpendingClient
from stock_trading.core import RawRecord, SemanticAnnotation, SemanticDirection, Source, content_sha256
from stock_trading.live.run_current_usaspending_shadow import (
    FileUsaSpendingShadowIntake,
    _matching_transaction_ids,
    _transaction_search_rows_for_award,
    poll_current_usaspending_shadow,
)
from stock_trading.storage import DuckDbEventStore


UTC = timezone.utc


def _raw(record_id: str, payload) -> RawRecord:
    content = json.dumps(payload)
    return RawRecord(
        source=Source.USASPENDING,
        source_record_id=record_id,
        fetched_at=datetime(2026, 8, 23, 19, 0, tzinfo=UTC),
        content_type="application/json",
        content=content,
        sha256=content_sha256(content),
    )


def _seed_modeled_identity(data_root: Path, experiment_dir: Path) -> None:
    (data_root / "manifests").mkdir(parents=True)
    (data_root / "manifests" / "sec_companies.jsonl").write_text(
        json.dumps(
            {
                "company_id": "cmp_microsoft",
                "issuer_names": ["MICROSOFT CORPORATION"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "training_rows.jsonl").write_text(
        json.dumps({"company_id": "cmp_microsoft"}) + "\n",
        encoding="utf-8",
    )


class _FakeExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def extract(self, source_text: str, *, context: str = "") -> SemanticAnnotation:
        self.calls += 1
        assert "Microsoft Federal" in source_text
        assert "procurement" in source_text.lower()
        assert context == "US federal government contract modification"
        return SemanticAnnotation(
            topics=("GOVERNMENT.PROCUREMENT",),
            direction=SemanticDirection.POSITIVE,
            novelty=0.6,
            importance=0.8,
            company_relevance=0.95,
            policy_relevance=0.7,
            confidence=0.95,
            model="test-qwen",
            extractor_version="semantic-v1",
            schema_version="semantic-v1",
        )


class _FakeUsaSpendingClient:
    def __init__(self, *, award_has_next: bool = False) -> None:
        self.award_has_next = award_has_next
        self.award_search_calls = 0
        self.award_detail_calls = 0
        self.transaction_search_calls = 0
        self.transaction_detail_calls = 0

    def search_contract_awards_page(self, **kwargs) -> RawRecord:
        self.award_search_calls += 1
        return _raw(
            f"award-search-{self.award_search_calls}",
            {
                "results": [
                    {
                        "Award ID": "W91TEST-26-C-0001",
                        "Recipient Name": "Microsoft Federal LLC",
                        "Recipient UEI": "CHILDUEI1234",
                        "Last Modified Date": "2026-08-23",
                        "generated_internal_id": "CONT_AWD_TEST_9700_-NONE-_-NONE-",
                    }
                ],
                "page_metadata": {"page": 1, "hasNext": self.award_has_next},
            },
        )

    def fetch_award(self, award_id: str) -> RawRecord:
        self.award_detail_calls += 1
        assert award_id == "CONT_AWD_TEST_9700_-NONE-_-NONE-"
        return _raw(
            "award-detail",
            {
                "generated_unique_award_id": award_id,
                "type": "D",
                "type_description": "DEFINITIVE CONTRACT",
                "description": "Cloud procurement support",
                "total_obligation": 5000000,
                "base_and_all_options": 10000000,
                "recipient": {
                    "recipient_name": "Microsoft Federal LLC",
                    "recipient_uei": "CHILDUEI1234",
                    "parent_recipient_name": "Microsoft Corporation",
                    "parent_recipient_uei": "PARENTUEI123",
                },
                "awarding_agency": {
                    "toptier_agency": {"name": "Department of Defense"},
                    "subtier_agency": {"name": "Defense Information Systems Agency"},
                },
                "latest_transaction_contract_data": {
                    "naics": "541512",
                    "product_or_service_code": "D399",
                },
            },
        )

    def search_contract_transactions_page(self, award_search_id: str, **kwargs) -> RawRecord:
        self.transaction_search_calls += 1
        assert award_search_id == "W91TEST-26-C-0001"
        return _raw(
            "transaction-search",
            {
                "results": [
                    {
                        "Award ID": award_search_id,
                        "Mod": "P00001",
                        "Recipient Name": "Microsoft Federal LLC",
                        "Recipient UEI": "CHILDUEI1234",
                        "Action Date": "2026-08-22",
                        "Action Type": "FUNDING ONLY ACTION",
                        "Transaction Amount": 250000,
                        "Transaction Description": "Incremental cloud procurement funding",
                        "generated_internal_id": "CONT_AWD_TEST_9700_-NONE-_-NONE-",
                    }
                ],
                "page_metadata": {"page": 1, "hasNext": False},
            },
        )

    def fetch_transactions(self, award_id: str, **kwargs) -> RawRecord:
        self.transaction_detail_calls += 1
        assert award_id == "CONT_AWD_TEST_9700_-NONE-_-NONE-"
        return _raw(
            "transaction-detail",
            {
                "results": [
                    {
                        "id": "CONT_TX_TEST_1",
                        "type": "D",
                        "type_description": "DEFINITIVE CONTRACT",
                        "action_date": "2026-08-22",
                        "action_type": "C",
                        "action_type_description": "FUNDING ONLY ACTION",
                        "modification_number": "P00001",
                        "description": "Incremental cloud procurement funding",
                        "federal_action_obligation": 250000,
                    },
                    {
                        "id": "CONT_TX_OLD",
                        "type": "D",
                        "action_date": "2026-07-01",
                        "action_type": "A",
                        "action_type_description": "NEW AWARD",
                        "modification_number": "0",
                        "description": "Original award",
                        "federal_action_obligation": 1000000,
                    },
                ],
                "page_metadata": {"page": 1, "hasNext": False},
            },
        )


def test_transaction_search_rows_require_exact_generated_award_identity() -> None:
    matching = {
        "Award ID": "75F40125F19008",
        "Mod": "P00002",
        "generated_internal_id": "CONT_AWD_TARGET",
    }
    foreign = {
        "Award ID": "75F40125F19008",
        "Mod": "P00001",
        "generated_internal_id": "CONT_AWD_OTHER_PARENT",
    }
    missing = {
        "Award ID": "75F40125F19008",
        "Mod": "P00003",
    }

    accepted, filtered = _transaction_search_rows_for_award(
        [matching, foreign, missing],
        "CONT_AWD_TARGET",
    )

    assert accepted == [matching]
    assert filtered == [foreign, missing]


def test_contract_search_client_uses_last_modified_and_documented_award_filter() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"results": [], "page_metadata": {"page": 1, "hasNext": False}},
        )

    client = UsaSpendingClient(client=httpx.Client(transport=httpx.MockTransport(handler)))
    client.search_contract_awards_page(
        modified_after=date(2026, 8, 22),
        modified_before=date(2026, 8, 23),
    )
    client.search_contract_transactions_page(
        "W91TEST-26-C-0001",
        modified_after=date(2026, 8, 22),
        modified_before=date(2026, 8, 23),
    )

    assert len(requests) == 2
    awards = json.loads(requests[0].content)
    assert awards["filters"]["award_type_codes"] == ["A", "B", "C", "D"]
    assert awards["filters"]["time_period"] == [
        {
            "start_date": "2026-08-22",
            "end_date": "2026-08-23",
            "date_type": "last_modified_date",
        }
    ]
    assert awards["sort"] == "Last Modified Date"

    transactions = json.loads(requests[1].content)
    assert transactions["filters"]["award_ids"] == ["W91TEST-26-C-0001"]
    assert "award_unique_id" not in transactions["filters"]
    assert transactions["filters"]["time_period"][0]["date_type"] == "last_modified_date"


def test_contract_search_client_rejects_invalid_window() -> None:
    client = UsaSpendingClient(client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200))))
    with pytest.raises(ValueError, match="start date"):
        client.search_contract_awards_page(
            modified_after=date(2026, 8, 24),
            modified_before=date(2026, 8, 23),
        )


def test_usaspending_shadow_maps_explicit_parent_and_isolated_qwen(tmp_path) -> None:
    pytest.importorskip("duckdb")
    data_root = tmp_path / "data"
    runtime_dir = data_root / "runtime"
    experiment_dir = data_root / "experiments" / "model"
    _seed_modeled_identity(data_root, experiment_dir)
    authoritative = DuckDbEventStore(data_root / "normalized" / "events.duckdb")
    extractor = _FakeExtractor()
    client = _FakeUsaSpendingClient()

    result = poll_current_usaspending_shadow(
        data_root=data_root,
        experiment_dir=experiment_dir,
        runtime_dir=runtime_dir,
        usaspending_client=client,  # type: ignore[arg-type]
        extractor=extractor,  # type: ignore[arg-type]
        as_of=datetime(2026, 8, 23, 19, 0, tzinfo=UTC),
    )

    assert result.awards_seen == 1
    assert result.candidate_award_count == 1
    assert result.mapped_award_count == 1
    assert result.transaction_search_row_count == 1
    assert result.transaction_identity_filtered_row_count == 0
    assert result.matched_transaction_count == 1
    assert result.unmatched_transaction_count == 0
    assert result.mapped_event_count == 1
    assert result.semantic_enriched_event_count == 1
    assert result.pending_events_added == 1
    assert extractor.calls == 1
    assert authoritative.count() == 0

    shadow = DuckDbEventStore(runtime_dir / "usaspending_shadow" / "events.duckdb")
    events = shadow.all_events()
    assert len(events) == 1
    event = events[0]
    assert event.company_id == "cmp_microsoft"
    assert event.source_record_id.endswith(":CONT_TX_TEST_1")
    assert event.public_time == datetime(2026, 8, 23, 19, 0, tzinfo=UTC)
    assert event.event_time == datetime(2026, 8, 22, 0, 0, tzinfo=UTC)
    assert event.semantic is not None
    assert event.semantic.model == "test-qwen"
    assert event.payload.obligation_amount == 250000

    # The explicit parent relationship is only persisted in the isolated alias DB.
    assert not (data_root / "normalized" / "aliases.duckdb").exists()
    assert (runtime_dir / "usaspending_shadow" / "aliases.duckdb").exists()

    second = poll_current_usaspending_shadow(
        data_root=data_root,
        experiment_dir=experiment_dir,
        runtime_dir=runtime_dir,
        usaspending_client=client,  # type: ignore[arg-type]
        extractor=extractor,  # type: ignore[arg-type]
        as_of=datetime(2026, 8, 23, 19, 5, tzinfo=UTC),
    )
    assert second.pending_events_added == 0
    assert second.pending_event_count == 1
    assert second.semantic_enriched_event_count == 0
    assert extractor.calls == 1
    assert authoritative.count() == 0


def test_usaspending_shadow_recovers_stored_event_without_repeating_qwen(tmp_path) -> None:
    pytest.importorskip("duckdb")
    data_root = tmp_path / "data"
    runtime_dir = data_root / "runtime"
    experiment_dir = data_root / "experiments" / "model"
    _seed_modeled_identity(data_root, experiment_dir)
    extractor = _FakeExtractor()
    client = _FakeUsaSpendingClient()

    first = poll_current_usaspending_shadow(
        data_root=data_root,
        experiment_dir=experiment_dir,
        runtime_dir=runtime_dir,
        usaspending_client=client,  # type: ignore[arg-type]
        extractor=extractor,  # type: ignore[arg-type]
        as_of=datetime(2026, 8, 23, 19, 0, tzinfo=UTC),
    )
    assert first.semantic_enriched_event_count == 1
    assert extractor.calls == 1

    # Simulate a crash after event durability but before intake durability.
    (runtime_dir / "usaspending_shadow" / "intake.json").unlink()
    replay = poll_current_usaspending_shadow(
        data_root=data_root,
        experiment_dir=experiment_dir,
        runtime_dir=runtime_dir,
        usaspending_client=client,  # type: ignore[arg-type]
        extractor=extractor,  # type: ignore[arg-type]
        as_of=datetime(2026, 8, 23, 19, 10, tzinfo=UTC),
    )
    assert replay.recovered_event_count == 1
    assert replay.semantic_enriched_event_count == 0
    assert replay.pending_events_added == 1
    assert extractor.calls == 1


def test_usaspending_handled_tombstone_prevents_overlap_requeue(tmp_path) -> None:
    pytest.importorskip("duckdb")
    data_root = tmp_path / "data"
    runtime_dir = data_root / "runtime"
    experiment_dir = data_root / "experiments" / "model"
    _seed_modeled_identity(data_root, experiment_dir)
    extractor = _FakeExtractor()
    client = _FakeUsaSpendingClient()

    first = poll_current_usaspending_shadow(
        data_root=data_root,
        experiment_dir=experiment_dir,
        runtime_dir=runtime_dir,
        usaspending_client=client,  # type: ignore[arg-type]
        extractor=extractor,  # type: ignore[arg-type]
        as_of=datetime(2026, 8, 23, 19, 0, tzinfo=UTC),
    )
    intake = FileUsaSpendingShadowIntake(runtime_dir / "usaspending_shadow" / "intake.json")
    event_id = intake.pending()[0].event_id
    assert intake.acknowledge((event_id,)) == 1
    assert intake.pending() == ()
    assert event_id in intake.load().handled_event_ids

    replay = poll_current_usaspending_shadow(
        data_root=data_root,
        experiment_dir=experiment_dir,
        runtime_dir=runtime_dir,
        usaspending_client=client,  # type: ignore[arg-type]
        extractor=extractor,  # type: ignore[arg-type]
        as_of=datetime(2026, 8, 23, 19, 30, tzinfo=UTC),
    )
    assert replay.pending_events_added == 0
    assert replay.pending_event_count == 0
    assert replay.handled_event_count == 1
    assert extractor.calls == 1
    assert first.watermark is not None
    assert replay.watermark is not None
    assert replay.watermark > first.watermark


def test_usaspending_truncated_award_pagination_does_not_advance_watermark(tmp_path) -> None:
    pytest.importorskip("duckdb")
    data_root = tmp_path / "data"
    runtime_dir = data_root / "runtime"
    experiment_dir = data_root / "experiments" / "model"
    _seed_modeled_identity(data_root, experiment_dir)
    intake_path = runtime_dir / "usaspending_shadow" / "intake.json"

    with pytest.raises(RuntimeError, match="max_pages reached"):
        poll_current_usaspending_shadow(
            data_root=data_root,
            experiment_dir=experiment_dir,
            runtime_dir=runtime_dir,
            usaspending_client=_FakeUsaSpendingClient(award_has_next=True),  # type: ignore[arg-type]
            extractor=_FakeExtractor(),  # type: ignore[arg-type]
            as_of=datetime(2026, 8, 23, 19, 0, tzinfo=UTC),
            max_pages=1,
        )

    assert not intake_path.exists()
    assert DuckDbEventStore(runtime_dir / "usaspending_shadow" / "events.duckdb").count() == 0


def test_transaction_search_match_requires_unique_detail_identity() -> None:
    changed = {
        "Action Date": "2026-08-22",
        "Mod": "P00001",
        "Transaction Amount": 250000,
        "Transaction Description": "Incremental cloud procurement funding",
        "Action Type": "FUNDING ONLY ACTION",
    }
    detail = {
        "id": "tx-1",
        "action_date": "2026-08-22",
        "modification_number": "P00001",
        "federal_action_obligation": 250000,
        "description": "Incremental cloud procurement funding",
        "action_type": "C",
        "action_type_description": "FUNDING ONLY ACTION",
    }
    assert _matching_transaction_ids(changed, [detail]) == ["tx-1"]
    assert _matching_transaction_ids(changed, [detail, {**detail, "id": "tx-2"}]) == [
        "tx-1",
        "tx-2",
    ]
