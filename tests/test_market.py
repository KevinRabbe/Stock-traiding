import json
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from stock_trading.core import RawRecord, Source, content_sha256
from stock_trading.entities import company_id_from_sec_cik
from stock_trading.market import (
    DuckDbMarketStore,
    MarketBar,
    SecurityMapping,
    SecurityRegistry,
    TiingoNormalizer,
    build_forward_label,
    conservative_first_tradable_time,
    normalize_tiingo_ticker,
)


def _raw_json(record_id: str, payload) -> RawRecord:
    content = json.dumps(payload)
    return RawRecord(
        source=Source.TIINGO,
        source_record_id=record_id,
        fetched_at=datetime(2026, 8, 11, 20, 0, tzinfo=timezone.utc),
        content_type="application/json",
        content=content,
        sha256=content_sha256(content),
    )


def _bar(company_id: str, ticker: str, day: date, open_: str, close: str, high: str, low: str) -> MarketBar:
    return MarketBar(
        company_id=company_id,
        ticker=ticker,
        date=day,
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal("1000000"),
        adj_open=Decimal(open_),
        adj_high=Decimal(high),
        adj_low=Decimal(low),
        adj_close=Decimal(close),
        adj_volume=Decimal("1000000"),
    )


def test_tiingo_normalizer_preserves_raw_adjusted_and_corporate_actions() -> None:
    company_id = company_id_from_sec_cik("12345")
    payload = [
        {
            "date": "2026-08-10T00:00:00.000Z",
            "open": 100,
            "high": 110,
            "low": 99,
            "close": 108,
            "volume": 1000000,
            "adjOpen": 50,
            "adjHigh": 55,
            "adjLow": 49.5,
            "adjClose": 54,
            "adjVolume": 2000000,
            "divCash": 0.25,
            "splitFactor": 2.0,
        }
    ]
    bars = TiingoNormalizer().parse_prices(
        _raw_json("prices:EXM", payload),
        company_id=company_id,
        ticker="EXM",
    )

    assert len(bars) == 1
    assert bars[0].close == Decimal("108")
    assert bars[0].adj_close == Decimal("54")
    assert bars[0].dividend_cash == Decimal("0.25")
    assert bars[0].split_factor == Decimal("2.0")


def test_tiingo_metadata_becomes_point_in_time_mapping() -> None:
    company_id = company_id_from_sec_cik("12345")
    metadata = {
        "ticker": "EXM",
        "name": "Example Corp",
        "exchangeCode": "NASDAQ",
        "startDate": "2012-01-03",
        "endDate": "2026-08-10",
    }
    mapping = TiingoNormalizer().parse_metadata(
        _raw_json("metadata:EXM", metadata),
        company_id=company_id,
    )

    assert mapping.contains(date(2020, 1, 2))
    assert not mapping.contains(date(2030, 1, 2))


def test_security_registry_blocks_overlapping_ticker_reuse() -> None:
    registry = SecurityRegistry()
    registry.add(
        SecurityMapping(
            company_id="cmp_a",
            ticker="ABC",
            valid_from=date(2010, 1, 1),
            valid_to=date(2020, 12, 31),
        )
    )
    registry.add(
        SecurityMapping(
            company_id="cmp_b",
            ticker="ABC",
            valid_from=date(2021, 1, 1),
            valid_to=None,
        )
    )

    assert registry.company_for_ticker("ABC", date(2015, 1, 1)) == "cmp_a"
    assert registry.company_for_ticker("ABC", date(2025, 1, 1)) == "cmp_b"

    with pytest.raises(ValueError, match="overlaps"):
        registry.add(
            SecurityMapping(
                company_id="cmp_c",
                ticker="ABC",
                valid_from=date(2020, 6, 1),
                valid_to=date(2021, 6, 1),
            )
        )


def test_forward_label_uses_adjusted_open_close_and_excursions() -> None:
    stock = [
        _bar("cmp_stock", "AAA", date(2026, 8, 10), "100", "105", "107", "98"),
        _bar("cmp_stock", "AAA", date(2026, 8, 11), "105", "110", "112", "103"),
    ]
    benchmark = [
        _bar("cmp_spy", "SPY", date(2026, 8, 10), "100", "101", "102", "99"),
        _bar("cmp_spy", "SPY", date(2026, 8, 11), "101", "102", "103", "100"),
    ]

    label = build_forward_label(stock, benchmark, horizon=2)

    assert label.stock_return == pytest.approx(0.10)
    assert label.benchmark_return == pytest.approx(0.02)
    assert label.alpha == pytest.approx(0.08)
    assert label.max_favorable_excursion == pytest.approx(0.12)
    assert label.max_adverse_excursion == pytest.approx(-0.02)


def test_conservative_execution_uses_next_actual_market_date() -> None:
    public_time = datetime(2026, 8, 7, 20, 15, tzinfo=timezone.utc)  # Friday afternoon ET
    result = conservative_first_tradable_time(public_time, date(2026, 8, 10))
    assert result == datetime(2026, 8, 10, 13, 30, tzinfo=timezone.utc)


def test_tiingo_symbol_normalization_matches_documented_dash_style() -> None:
    assert normalize_tiingo_ticker("brk.b") == "BRK-B"


def test_market_store_uses_actual_next_bar_and_is_idempotent(tmp_path) -> None:
    pytest.importorskip("duckdb")
    company_id = company_id_from_sec_cik("12345")
    bars = [
        _bar(company_id, "EXM", date(2026, 8, 7), "10", "10.5", "10.8", "9.9"),
        _bar(company_id, "EXM", date(2026, 8, 10), "10.6", "11", "11.2", "10.5"),
    ]
    store = DuckDbMarketStore(tmp_path / "market.duckdb")
    store.put_many(bars)
    store.put_many(bars)

    next_bar = store.next_bar_after(company_id, date(2026, 8, 7))
    assert next_bar is not None
    assert next_bar["date"] == date(2026, 8, 10)
    assert len(store.bars_from(company_id, date(2026, 8, 7), 10)) == 2
