import json
from collections import Counter
from datetime import date, datetime, timezone

import httpx
import pytest

from stock_trading.core import RawRecord, Source, content_sha256
from stock_trading.market import DuckDbMarketStore, IssuerObservation, MarketBackfillService
from stock_trading.storage import FileRawStore


def _raw(record_id: str, payload) -> RawRecord:
    content = json.dumps(payload)
    return RawRecord(
        source=Source.TIINGO,
        source_record_id=record_id,
        fetched_at=datetime(2026, 8, 12, 20, 0, tzinfo=timezone.utc),
        content_type="application/json",
        content=content,
        sha256=content_sha256(content),
    )


class _FakeTiingoClient:
    def __init__(self) -> None:
        self.metadata_calls: Counter[str] = Counter()
        self.price_calls: Counter[str] = Counter()

    def fetch_metadata(self, ticker: str) -> RawRecord:
        self.metadata_calls[ticker] += 1
        if ticker == "DEAD":
            request = httpx.Request("GET", "https://api.tiingo.com/tiingo/daily/DEAD")
            response = httpx.Response(404, request=request)
            raise httpx.HTTPStatusError("not found", request=request, response=response)
        if ticker == "NOSTART":
            return _raw(
                f"metadata:{ticker}",
                {
                    "ticker": ticker,
                    "name": "No Start Corp",
                    "exchangeCode": "NASDAQ",
                    "startDate": None,
                    "endDate": None,
                },
            )
        return _raw(
            f"metadata:{ticker}",
            {
                "ticker": ticker,
                "name": "Good Corp",
                "exchangeCode": "NASDAQ",
                "startDate": "2010-01-01",
                "endDate": None,
            },
        )

    def fetch_prices(self, ticker: str, start: date, end: date) -> RawRecord:
        self.price_calls[ticker] += 1
        assert ticker == "GOOD"
        return _raw(
            f"prices:{ticker}:{start}:{end}",
            [
                {
                    "date": "2020-01-02T00:00:00.000Z",
                    "open": 10,
                    "high": 11,
                    "low": 9,
                    "close": 10.5,
                    "volume": 1000,
                    "adjOpen": 10,
                    "adjHigh": 11,
                    "adjLow": 9,
                    "adjClose": 10.5,
                    "adjVolume": 1000,
                    "divCash": 0,
                    "splitFactor": 1,
                }
            ],
        )


def test_market_backfill_continues_after_unavailable_or_incomplete_metadata(tmp_path) -> None:
    pytest.importorskip("duckdb")
    client = _FakeTiingoClient()
    service = MarketBackfillService(
        client=client,
        raw_store=FileRawStore(tmp_path / "raw"),
        market_store=DuckDbMarketStore(tmp_path / "market.duckdb"),
    )
    observations = [
        IssuerObservation(
            sec_cik="11111",
            issuer_name="Dead Corp",
            ticker="DEAD",
            observed_date=date(2020, 1, 2),
        ),
        IssuerObservation(
            sec_cik="33333",
            issuer_name="No Start Corp",
            ticker="NOSTART",
            observed_date=date(2020, 1, 2),
        ),
        IssuerObservation(
            sec_cik="22222",
            issuer_name="Good Corp",
            ticker="GOOD",
            observed_date=date(2020, 1, 2),
        ),
    ]

    result = service.backfill(
        observations,
        start=date(2020, 1, 1),
        end=date(2020, 1, 31),
    )

    assert result.failed_metadata_requests == 1
    assert result.unresolved_observations == 2
    assert result.downloaded_price_series == 1
    assert result.failed_price_series == 0
    assert result.normalized_bars == 1
    reasons = [resolution.reason for resolution in result.resolutions]
    assert "tiingo_metadata_http_404" in reasons
    assert "tiingo_metadata_missing_start_date" in reasons


def test_market_backfill_reuses_cached_metadata_and_price_series(tmp_path) -> None:
    pytest.importorskip("duckdb")
    raw_store = FileRawStore(tmp_path / "raw")
    market_store = DuckDbMarketStore(tmp_path / "market.duckdb")
    client = _FakeTiingoClient()
    observations = [
        IssuerObservation(
            sec_cik="22222",
            issuer_name="Good Corp",
            ticker="GOOD",
            observed_date=date(2020, 1, 2),
        )
    ]

    first = MarketBackfillService(
        client=client,
        raw_store=raw_store,
        market_store=market_store,
    ).backfill(
        observations,
        start=date(2020, 1, 1),
        end=date(2020, 1, 31),
    )
    second = MarketBackfillService(
        client=client,
        raw_store=raw_store,
        market_store=market_store,
    ).backfill(
        observations,
        start=date(2020, 1, 1),
        end=date(2020, 1, 31),
    )

    assert first.downloaded_price_series == 1
    assert second.downloaded_price_series == 0
    assert second.normalized_bars == 1
    assert client.metadata_calls["GOOD"] == 1
    assert client.price_calls["GOOD"] == 1
