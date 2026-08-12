from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

from stock_trading.core import as_utc


_EASTERN = ZoneInfo("America/New_York")
_MARKET_OPEN = time(hour=9, minute=30)


def decision_market_date(public_time: datetime) -> date:
    """Return the U.S. Eastern calendar date on which information became public."""

    return as_utc(public_time).astimezone(_EASTERN).date()


def next_open_timestamp(next_trading_date: date) -> datetime:
    """Return the regular-session open for a known actual trading date."""

    local = datetime.combine(next_trading_date, _MARKET_OPEN, tzinfo=_EASTERN)
    return local.astimezone(timezone.utc)


def conservative_first_tradable_time(public_time: datetime, next_trading_date: date) -> datetime:
    """Conservatively execute at the next actual regular-session open.

    `next_trading_date` must come from actual market data and must be strictly
    after the Eastern calendar date of publication. This deliberately avoids
    same-day assumptions in the first daily/EOD backtester.
    """

    publication_date = decision_market_date(public_time)
    if next_trading_date <= publication_date:
        raise ValueError("next_trading_date must be after publication date")
    return next_open_timestamp(next_trading_date)
