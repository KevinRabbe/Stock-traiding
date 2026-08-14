from datetime import date
from decimal import Decimal

import pytest

from stock_trading.market import DuckDbMarketStore, MarketBar, SecurityMapping


def _bar(security_id: str, ticker: str, day: date, close: str) -> MarketBar:
    value = Decimal(close)
    return MarketBar(
        security_id=security_id,
        ticker=ticker,
        date=day,
        open=value,
        high=value,
        low=value,
        close=value,
        volume=Decimal("1000000"),
        adj_open=value,
        adj_high=value,
        adj_low=value,
        adj_close=value,
        adj_volume=Decimal("1000000"),
    )


def test_market_read_cache_reuses_series_and_preserves_slice_semantics(tmp_path) -> None:
    pytest.importorskip("duckdb")
    security_id = "security_cached"
    store = DuckDbMarketStore(tmp_path / "market.duckdb")
    store.put_many(
        [
            _bar(security_id, "CACHE", date(2026, 8, 6), "10"),
            _bar(security_id, "CACHE", date(2026, 8, 7), "11"),
            _bar(security_id, "CACHE", date(2026, 8, 10), "12"),
            _bar(security_id, "CACHE", date(2026, 8, 11), "13"),
        ]
    )
    store.enable_read_cache(max_series=2)

    assert store.next_bar_after(security_id, date(2026, 8, 7)).date == date(2026, 8, 10)
    assert store.bar_on(security_id, date(2026, 8, 7)).close == Decimal("11.0")
    assert [bar.date for bar in store.bars_before(security_id, date(2026, 8, 10), 2)] == [
        date(2026, 8, 6),
        date(2026, 8, 7),
    ]
    assert [bar.date for bar in store.bars_from(security_id, date(2026, 8, 7), 2)] == [
        date(2026, 8, 7),
        date(2026, 8, 10),
    ]

    stats = store.read_cache_stats()
    assert stats["series_misses"] == 1
    assert stats["series_hits"] == 3
    assert stats["cached_series"] == 1


def test_market_read_cache_invalidates_bars_and_company_mapping_after_writes(tmp_path) -> None:
    pytest.importorskip("duckdb")
    security_id = "security_cached"
    company_id = "company_cached"
    store = DuckDbMarketStore(tmp_path / "market.duckdb")
    store.register_mapping(
        SecurityMapping(
            company_id=company_id,
            security_id=security_id,
            ticker="CACHE",
            valid_from=date(2020, 1, 1),
            valid_to=None,
        )
    )
    store.put_many([_bar(security_id, "CACHE", date(2026, 8, 10), "12")])
    store.enable_read_cache(max_series=2)

    assert store.security_for_company(company_id, date(2026, 8, 10)) == security_id
    assert store.security_for_company(company_id, date(2026, 8, 11)) == security_id
    assert store.bar_on(security_id, date(2026, 8, 10)) is not None
    assert store.read_cache_stats()["mapping_misses"] == 1
    assert store.read_cache_stats()["mapping_hits"] == 1
    assert store.read_cache_stats()["series_misses"] == 1

    store.put_many([_bar(security_id, "CACHE", date(2026, 8, 11), "13")])
    assert store.bar_on(security_id, date(2026, 8, 11)) is not None
    assert store.read_cache_stats()["series_misses"] == 2

    store.register_mapping(
        SecurityMapping(
            company_id=company_id,
            security_id=security_id,
            ticker="CACHE",
            valid_from=date(2020, 1, 1),
            valid_to=date(2026, 8, 10),
        )
    )
    assert store.security_for_company(company_id, date(2026, 8, 11)) is None
    assert store.read_cache_stats()["mapping_misses"] == 2
