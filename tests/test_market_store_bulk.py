from datetime import date, timedelta
from decimal import Decimal

import pytest

from stock_trading.market import DuckDbMarketStore, MarketBar


def _bar(day: date, *, close: str = "10.5") -> MarketBar:
    value = Decimal(close)
    return MarketBar(
        security_id="security_bulk",
        ticker="BULK",
        date=day,
        open=Decimal("10"),
        high=max(value, Decimal("11")),
        low=Decimal("9"),
        close=value,
        volume=Decimal("1000"),
        adj_open=Decimal("10"),
        adj_high=max(value, Decimal("11")),
        adj_low=Decimal("9"),
        adj_close=value,
        adj_volume=Decimal("1000"),
        dividend_cash=Decimal("0"),
        split_factor=Decimal("1"),
    )


def test_market_store_bulk_upsert_is_idempotent_and_updates_conflicts(tmp_path) -> None:
    pytest.importorskip("duckdb")
    store = DuckDbMarketStore(tmp_path / "market.duckdb")
    start = date(2020, 1, 1)
    bars = [_bar(start + timedelta(days=index)) for index in range(1000)]

    store.put_many(bars)
    store.put_many(bars)

    end = start + timedelta(days=999)
    assert store.count_bars("security_bulk", "BULK", start, end) == 1000
    assert store.date_bounds("security_bulk", "BULK") == (start, end)

    replacement = _bar(start + timedelta(days=500), close="99")
    store.put_many([replacement])

    restored = store.bar_on("security_bulk", replacement.date)
    assert restored is not None
    assert restored.close == Decimal("99.0")
