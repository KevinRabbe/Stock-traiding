from __future__ import annotations

from datetime import date, datetime, timedelta

from stock_trading.core import as_utc
from stock_trading.market.execution_time import decision_market_date


class XnysExecutionSessionResolver:
    """Resolve the conservative next NYSE regular session from publication time.

    The strategy's established execution contract is strictly next-session-open:
    information published on an Eastern calendar date is never assigned to that
    same date's session, even when published before the open. Exchange holidays
    and special closures come from ``exchange_calendars`` rather than weekday
    heuristics or future market-price rows.
    """

    def __init__(self) -> None:
        try:
            import exchange_calendars as xcals
        except ImportError as exc:  # pragma: no cover - package dependency guard
            raise RuntimeError("exchange_calendars is required for XNYS sessions") from exc
        self._calendar = xcals.get_calendar("XNYS")

    def execution_date(self, publication_time: datetime) -> date:
        publication_day = decision_market_date(as_utc(publication_time))
        # Search strictly after publication_day. Fourteen calendar days is much
        # wider than any ordinary NYSE closure window while remaining fail-closed
        # if the installed calendar unexpectedly lacks coverage.
        start = publication_day + timedelta(days=1)
        end = publication_day + timedelta(days=14)
        sessions = self._calendar.sessions_in_range(start, end)
        if len(sessions) == 0:
            raise RuntimeError(
                f"XNYS calendar has no session in {start.isoformat()}..{end.isoformat()}"
            )
        return sessions[0].date()
