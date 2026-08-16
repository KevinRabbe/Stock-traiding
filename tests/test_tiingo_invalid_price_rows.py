import json
from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from stock_trading.core import RawRecord, Source, content_sha256
from stock_trading.market import (
    DuckDbMarketStore,
    IssuerObservation,
    MarketBackfillService,
    TiingoNormalizer,
)
from stock_trading.storage import FileRawStore


def _raw(record_id: str, payload) -> RawRecord:
    content = json.dumps(payload)
    return RawRecord(
        source=Source.TIINGO,
        source_record_id=record_id,
        fetched_at=datetime(2026, 8, 16, 9, 0, tzinfo=timezone.utc),
        content_type="application/json",
        content=content,
        sha256=content_sha256(content),
    )


def _price_row(day: str, *, invalid_range: bool = False) -> dict[str, object]:
    return {
        "date": f"{day}T00:00:00.000Z",
        "open": 10,
        "high": 9 if invalid_range else 11,
        "low": 8,
        "close": 10.5,
        "volume": 1000,
        "adjOpen": 10,
        "adjHigh": 11,
        "adjLow": 8,
        "adjClose": 10.5,
        "adjVolume": 1000,
        "divCash": 0,
        "splitFactor": 1,
    }


def test_tiingo_price_normalization_stays_strict_by_default() -> None:
    raw = _raw("prices:BAD", [_price_row("2020-01-02", invalid_range=True)])

    with pytest.raises(ValidationError, match="raw OHLC range is inconsistent"):
        TiingoNormalizer().parse_prices(
            raw,
            security_id="security_bad",
            ticker="BAD",
        )


def test_tiingo_price_normalization_can_quarantine_only_invalid_rows() -> None:
    raw = _raw(
        "prices:MIX",
        [
            _price_row("2020-01-02"),
            _price_row("2020-01-03", invalid_range=True),
        ],
    )
    invalid_rows = []

    bars = TiingoNormalizer().parse_prices(
        raw,
        security_id="security_mix",
        ticker="MIX",
        invalid_rows=invalid_rows,
    )

    assert [bar.date for bar in bars] == [date(2020, 1, 2)]
    assert len(invalid_rows) == 1
    assert invalid_rows[0].row_index == 1
    assert invalid_rows[0].date == "2020-01-03"
    assert "raw OHLC range is inconsistent" in invalid_rows[0].reason


class _MixedQualityClient:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def fetch_metadata(self, ticker: str) -> RawRecord:
        return _raw(
            f"metadata:{ticker}",
            {
                "ticker": ticker,
                "name": "Mixed Corp",
                "exchangeCode": "NASDAQ",
                "startDate": "2010-01-01",
                "endDate": None,
            },
        )

    def fetch_prices(self, ticker: str, start: date, end: date) -> RawRecord:
        return _raw(f"prices:{ticker}:{start}:{end}", self.rows)


def _observation() -> IssuerObservation:
    return IssuerObservation(
        sec_cik="12345",
        issuer_name="Mixed Corp",
        ticker="MIX",
        observed_date=date(2020, 1, 2),
    )


def test_market_backfill_preserves_good_rows_when_one_provider_row_is_invalid(tmp_path) -> None:
    pytest.importorskip("duckdb")
    service = MarketBackfillService(
        client=_MixedQualityClient(
            [
                _price_row("2020-01-02"),
                _price_row("2020-01-03", invalid_range=True),
            ]
        ),
        raw_store=FileRawStore(tmp_path / "raw"),
        market_store=DuckDbMarketStore(tmp_path / "market.duckdb"),
    )

    with pytest.warns(RuntimeWarning, match="Skipped 1 invalid Tiingo price row"):
        result = service.backfill(
            [_observation()],
            start=date(2020, 1, 1),
            end=date(2020, 1, 31),
        )

    assert result.downloaded_price_series == 1
    assert result.failed_price_series == 0
    assert result.skipped_invalid_price_rows == 1
    assert result.normalized_bars == 1


def test_market_backfill_marks_series_failed_when_every_provider_row_is_invalid(tmp_path) -> None:
    pytest.importorskip("duckdb")
    service = MarketBackfillService(
        client=_MixedQualityClient([_price_row("2020-01-03", invalid_range=True)]),
        raw_store=FileRawStore(tmp_path / "raw"),
        market_store=DuckDbMarketStore(tmp_path / "market.duckdb"),
    )

    with pytest.warns(RuntimeWarning, match="Skipped 1 invalid Tiingo price row"):
        result = service.backfill(
            [_observation()],
            start=date(2020, 1, 1),
            end=date(2020, 1, 31),
        )

    assert result.downloaded_price_series == 1
    assert result.failed_price_series == 1
    assert result.skipped_invalid_price_rows == 1
    assert result.normalized_bars == 0
