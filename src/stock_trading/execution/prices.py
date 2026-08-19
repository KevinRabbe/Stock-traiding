from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from stock_trading.market import DuckDbMarketStore


class PriceProvider(Protocol):
    """Return a usable execution/mark price known at `as_of`, or None."""

    def price(self, security_id: str, as_of: datetime) -> float | None: ...


@dataclass(frozen=True, slots=True)
class FixedPriceProvider:
    prices: dict[str, float]

    def price(self, security_id: str, as_of: datetime) -> float | None:
        del as_of
        value = self.prices.get(security_id)
        return float(value) if value is not None else None


class DuckDbClosePriceProvider:
    """Use only the bar for the requested calendar date as a paper mark/fill.

    It deliberately does not fall back to yesterday for execution: if today's bar
    is not stored yet, a queued paper order remains pending rather than receiving a
    fabricated stale fill.
    """

    def __init__(self, market_store: DuckDbMarketStore) -> None:
        self.market_store = market_store

    def price(self, security_id: str, as_of: datetime) -> float | None:
        bar = self.market_store.bar_on(security_id, as_of.date())
        return float(bar.adj_close) if bar is not None else None


class DuckDbLatestClosePriceProvider:
    """Mark from the latest adjusted close on or before the requested date.

    This is a valuation provider, not an execution provider. Before today's EOD bar
    exists it naturally carries yesterday's completed close, while an after-close
    lifecycle run marks from today's completed close.
    """

    def __init__(self, market_store: DuckDbMarketStore) -> None:
        self.market_store = market_store

    def price(self, security_id: str, as_of: datetime) -> float | None:
        bars = self.market_store.bars_before(
            security_id,
            as_of.date() + timedelta(days=1),
            1,
        )
        return float(bars[-1].adj_close) if bars else None


class DuckDbPreviousClosePriceProvider:
    """Return the last adjusted close strictly before the requested date.

    PAPER entries execute at the next session open. Existing portfolio equity used
    to size those opening orders therefore must not observe that session's close.
    """

    def __init__(self, market_store: DuckDbMarketStore) -> None:
        self.market_store = market_store

    def price(self, security_id: str, as_of: datetime) -> float | None:
        bars = self.market_store.bars_before(security_id, as_of.date(), 1)
        return float(bars[-1].adj_close) if bars else None
