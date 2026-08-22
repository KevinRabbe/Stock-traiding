import json
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest

from stock_trading.core import RawRecord, Source, content_sha256
from stock_trading.lobbying import LdaApiError, LdaClient, LdaRelayError
import stock_trading.lobbying.relay_snapshot as relay_snapshot
from stock_trading.lobbying.relay_snapshot import build_relay_snapshot


def _filing(filing_uuid: str, posted: str, *, year: int = 2026) -> dict:
    return {
        "filing_uuid": filing_uuid,
        "dt_posted": posted,
        "filing_year": year,
        "client": {"id": 1, "name": "Example Corp"},
        "registrant": {"id": 2, "name": "Example Lobbying"},
        "lobbying_activities": [],
    }


def _snapshot(*, generated_at: datetime | None = None) -> dict:
    filings = [
        _filing("b", "2026-08-21T12:00:00+00:00"),
        _filing("a", "2026-08-20T12:00:00+00:00"),
    ]
    return {
        "schema_version": 1,
        "generated_at": (generated_at or datetime.now(timezone.utc)).isoformat(),
        "source_base_url": "https://lda.gov/api/v1",
        "posted_after": "2026-08-08",
        "posted_before": "2026-08-22",
        "pages_fetched": 1,
        "filing_count": len(filings),
        "filings": filings,
    }


def _edge_client(call_count: list[int]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        return httpx.Response(
            403,
            request=request,
            text="<HTML><HEAD><TITLE>Access Denied</TITLE></HEAD><BODY>Access Denied</BODY></HTML>",
            headers={"Content-Type": "text/html"},
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def _relay_client(payload: dict, *, assert_no_auth: bool = False) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if assert_no_auth:
            assert "authorization" not in request.headers
        return httpx.Response(200, request=request, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_edge_denial_falls_back_to_fresh_relay_without_leaking_api_key(monkeypatch) -> None:
    monkeypatch.setenv("LDA_API_KEY", "registered-secret")
    direct_calls = [0]
    direct = _edge_client(direct_calls)
    relay = _relay_client(_snapshot(), assert_no_auth=True)
    client = LdaClient(client=direct, relay_client=relay)

    raw = client.fetch_filings_page(
        posted_after=date(2026, 8, 20),
        posted_before=date(2026, 8, 22),
        page=1,
        page_size=25,
    )
    payload = json.loads(raw.content)

    assert [item["filing_uuid"] for item in payload["results"]] == ["a", "b"]
    assert client.transport_mode == "relay"
    assert client.relay_snapshot_generated_at is not None
    assert direct_calls[0] == 1

    # Once the edge denial is known for this process, later pages stay on the relay.
    second = client.fetch_filings_page(
        posted_after=date(2026, 8, 20),
        posted_before=date(2026, 8, 22),
        page=2,
        page_size=25,
    )
    assert json.loads(second.content)["results"] == []
    assert direct_calls[0] == 1
    client.close()
    direct.close()
    relay.close()


def test_relay_rejects_stale_snapshot() -> None:
    direct_calls = [0]
    direct = _edge_client(direct_calls)
    relay = _relay_client(_snapshot(generated_at=datetime.now(timezone.utc) - timedelta(days=2)))
    client = LdaClient(client=direct, relay_client=relay, relay_max_age_hours=18)

    with pytest.raises(LdaRelayError, match="stale"):
        client.fetch_filings_page(
            posted_after=date(2026, 8, 20),
            posted_before=date(2026, 8, 22),
        )

    client.close()
    direct.close()
    relay.close()


def test_relay_rejects_request_outside_snapshot_coverage() -> None:
    direct_calls = [0]
    direct = _edge_client(direct_calls)
    relay = _relay_client(_snapshot())
    client = LdaClient(client=direct, relay_client=relay)

    with pytest.raises(LdaRelayError, match="outside relay coverage"):
        client.fetch_filings_page(
            posted_after=date(2026, 8, 1),
            posted_before=date(2026, 8, 22),
        )

    client.close()
    direct.close()
    relay.close()


class _SnapshotClient:
    base_url = "https://lda.gov/api/v1"

    def __init__(self) -> None:
        self.pages = {
            1: {
                "count": 2,
                "next": "page-2",
                "previous": None,
                "results": [_filing("b", "2026-08-21T12:00:00+00:00")],
            },
            2: {
                "count": 2,
                "next": None,
                "previous": "page-1",
                "results": [_filing("a", "2026-08-20T12:00:00+00:00")],
            },
        }

    def fetch_filings_page(self, *, page: int, **kwargs) -> RawRecord:
        content = json.dumps(self.pages[page]).encode("utf-8")
        return RawRecord(
            source=Source.LDA,
            source_record_id=f"relay-test:{page}",
            fetched_at=datetime(2026, 8, 22, 12, tzinfo=timezone.utc),
            content_type="application/json",
            content=content,
            sha256=content_sha256(content),
        )


def test_build_relay_snapshot_aggregates_pages_deterministically() -> None:
    payload = build_relay_snapshot(
        client=_SnapshotClient(),
        as_of=datetime(2026, 8, 22, 12, tzinfo=timezone.utc),
        lookback_days=14,
    )

    assert payload["schema_version"] == 1
    assert payload["source_base_url"] == "https://lda.gov/api/v1"
    assert payload["posted_after"] == "2026-08-08"
    assert payload["posted_before"] == "2026-08-22"
    assert payload["pages_fetched"] == 2
    assert payload["filing_count"] == 2
    assert [item["filing_uuid"] for item in payload["filings"]] == ["a", "b"]


class _RateLimitedSnapshotClient(_SnapshotClient):
    def __init__(self, *, always_rate_limited: bool = False) -> None:
        super().__init__()
        self.attempts = 0
        self.always_rate_limited = always_rate_limited

    def fetch_filings_page(self, *, page: int, **kwargs) -> RawRecord:
        self.attempts += 1
        if self.always_rate_limited or self.attempts == 1:
            raise LdaApiError(
                "rate limited",
                status_code=429,
                authenticated=False,
                retry_after="1",
            )
        return super().fetch_filings_page(page=page, **kwargs)


def test_build_relay_snapshot_retries_same_page_after_429(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(relay_snapshot.time, "sleep", sleeps.append)
    client = _RateLimitedSnapshotClient()

    payload = build_relay_snapshot(
        client=client,
        as_of=datetime(2026, 8, 22, 12, tzinfo=timezone.utc),
        lookback_days=14,
        rate_limit_retries=2,
    )

    assert payload["pages_fetched"] == 2
    assert payload["filing_count"] == 2
    assert client.attempts == 3  # page 1 throttle + page 1 retry + page 2
    assert sleeps == [1.0]


def test_build_relay_snapshot_stops_after_bounded_429_retries(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(relay_snapshot.time, "sleep", sleeps.append)
    client = _RateLimitedSnapshotClient(always_rate_limited=True)

    with pytest.raises(LdaApiError) as raised:
        build_relay_snapshot(
            client=client,
            as_of=datetime(2026, 8, 22, 12, tzinfo=timezone.utc),
            lookback_days=14,
            rate_limit_retries=2,
        )

    assert raised.value.status_code == 429
    assert client.attempts == 3
    assert sleeps == [1.0, 1.0]
