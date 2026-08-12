import json
from datetime import datetime, timezone

import pytest

from stock_trading.core import RawRecord, SemanticAnnotation, SemanticDirection, Source, content_sha256
from stock_trading.entities import DuckDbExternalEntityAliases
from stock_trading.experiments.enrich import (
    LdaEnrichmentConfig,
    enrich_lda_and_qwen,
    load_unique_company_name_index,
)
from stock_trading.storage import DuckDbEventStore


def _raw(record_id: str, payload) -> RawRecord:
    content = json.dumps(payload)
    return RawRecord(
        source=Source.LDA,
        source_record_id=record_id,
        fetched_at=datetime(2026, 8, 12, 20, 0, tzinfo=timezone.utc),
        content_type="application/json",
        content=content,
        sha256=content_sha256(content),
    )


class _FakeLdaClient:
    def fetch_filings_page(self, *, filing_year: int, page: int, page_size: int) -> RawRecord:
        assert filing_year == 2026
        assert page_size == 25
        if page > 1:
            return _raw(f"filings:{page}", {"results": [], "next": None})
        return _raw(
            "filings:1",
            {
                "count": 2,
                "next": None,
                "previous": None,
                "results": [
                    {
                        "filing_uuid": "mapped-filing",
                        "filing_year": 2026,
                        "filing_period": "second_quarter",
                        "dt_posted": "2026-07-20T18:15:22Z",
                        "income": "100000",
                        "client": {"id": 10, "name": "Example Semiconductor Corp"},
                        "registrant": {"id": 100, "name": "Lobby LLC"},
                        "lobbying_activities": [
                            {
                                "general_issue_code": "TRD",
                                "description": "Advanced semiconductor export controls.",
                                "government_entities": [
                                    {"name": "Department of Commerce"}
                                ],
                            }
                        ],
                    },
                    {
                        "filing_uuid": "unresolved-filing",
                        "filing_year": 2026,
                        "filing_period": "second_quarter",
                        "dt_posted": "2026-07-21T18:15:22Z",
                        "income": "50000",
                        "client": {"id": 11, "name": "Unknown Subsidiary LLC"},
                        "registrant": {"id": 100, "name": "Lobby LLC"},
                        "lobbying_activities": [],
                    },
                ],
            },
        )


class _FakeExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def extract(self, source_text: str, *, context: str = "") -> SemanticAnnotation:
        self.calls += 1
        assert "semiconductor" in source_text.lower()
        assert context == "US federal lobbying disclosure"
        return SemanticAnnotation(
            topics=("TECH.SEMICONDUCTORS", "TRADE.EXPORT_CONTROLS"),
            direction=SemanticDirection.NEUTRAL,
            novelty=0.5,
            importance=0.8,
            company_relevance=0.9,
            policy_relevance=0.9,
            confidence=0.95,
            model="Qwen/Qwen3.5-4B",
            extractor_version="semantic-v1",
            schema_version="semantic-v1",
        )


def _write_company_manifest(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "company_id": "cmp_chip",
                        "sec_cik": "0000000001",
                        "issuer_names": ["Example Semiconductor Corporation"],
                        "tickers": ["CHIP"],
                    }
                ),
                json.dumps(
                    {
                        "company_id": "cmp_one",
                        "sec_cik": "0000000002",
                        "issuer_names": ["Shared Name Inc"],
                        "tickers": ["ONE"],
                    }
                ),
                json.dumps(
                    {
                        "company_id": "cmp_two",
                        "sec_cik": "0000000003",
                        "issuer_names": ["Shared Name Corp"],
                        "tickers": ["TWO"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_company_name_index_preserves_ambiguity(tmp_path) -> None:
    path = tmp_path / "manifests" / "sec_companies.jsonl"
    _write_company_manifest(path)
    index = load_unique_company_name_index(path)

    assert index["EXAMPLE SEMICONDUCTOR"] == ("cmp_chip",)
    assert index["SHARED NAME"] == ("cmp_one", "cmp_two")


def test_lda_enrichment_maps_only_unique_names_and_qwen_only_mapped_events(tmp_path) -> None:
    pytest.importorskip("duckdb")
    companies_path = tmp_path / "manifests" / "sec_companies.jsonl"
    _write_company_manifest(companies_path)
    extractor = _FakeExtractor()

    result = enrich_lda_and_qwen(
        LdaEnrichmentConfig(
            data_root=tmp_path,
            start_year=2026,
            end_year=2026,
        ),
        lda_client=_FakeLdaClient(),
        extractor=extractor,
    )

    assert result.pages_downloaded == 1
    assert result.filings_seen == 2
    assert result.events_stored == 2
    assert result.mapped_events == 1
    assert result.qwen_enriched_events == 1
    assert result.unresolved_clients == 1
    assert extractor.calls == 1

    events = DuckDbEventStore(tmp_path / "normalized" / "events.duckdb").all_events()
    by_id = {event.source_record_id: event for event in events}
    assert by_id["mapped-filing"].company_id == "cmp_chip"
    assert by_id["mapped-filing"].semantic is not None
    assert by_id["unresolved-filing"].company_id is None
    assert by_id["unresolved-filing"].semantic is None

    aliases = DuckDbExternalEntityAliases(tmp_path / "normalized" / "aliases.duckdb")
    assert aliases.resolve(Source.LDA, "10") == "cmp_chip"
    assert aliases.resolve(Source.LDA, "11") is None

    unresolved = (tmp_path / "manifests" / "unresolved_lda_clients.jsonl").read_text(
        encoding="utf-8"
    )
    assert "Unknown Subsidiary LLC" in unresolved
