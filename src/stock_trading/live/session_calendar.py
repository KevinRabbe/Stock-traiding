from __future__ import annotations

from datetime import date, datetime, timedelta

from stock_trading.core import as_utc
from stock_trading.market.execution_time import decision_market_date


class XnysExecutionSessionResolver:
    """Resolve conservative NYSE regular sessions without future price rows.

    ``execution_date(publication_time)`` implements the strategy contract: a
    public event always belongs to the first XNYS session strictly after its
    Eastern publication date.

    ``cycle_execution_date(as_of)`` answers a different question: which session
    can a PAPER cycle still legitimately target *now*? Before today's XNYS open it
    returns today; at/after the open (or on a non-session day) it returns the next
    session. Keeping these concepts separate prevents a restart before the bell
    from incorrectly declaring yesterday's still-actionable filing stale.
    """

    def __init__(self) -> None:
        try:
            import exchange_calendars as xcals
        except ImportError as exc:  # pragma: no cover - package dependency guard
            raise RuntimeError("exchange_calendars is required for XNYS sessions") from exc
        self._calendar = xcals.get_calendar("XNYS")

    def execution_date(self, publication_time: datetime) -> date:
        publication_day = decision_market_date(as_utc(publication_time))
        return self._first_session_after(publication_day)

    def cycle_execution_date(self, as_of: datetime) -> date:
        cutoff = as_utc(as_of)
        eastern_day = decision_market_date(cutoff)
        sessions = self._calendar.sessions_in_range(eastern_day, eastern_day)
        if len(sessions):
            session = sessions[0]
            open_at = self._calendar.session_open(session).to_pydatetime()
            if cutoff < open_at:
                return eastern_day
        return self._first_session_after(eastern_day)

    def _first_session_after(self, day: date) -> date:
        # Search strictly after day. Fourteen calendar days is much wider than
        # any ordinary NYSE closure window while remaining fail-closed if the
        # installed calendar unexpectedly lacks coverage.
        start = day + timedelta(days=1)
        end = day + timedelta(days=14)
        sessions = self._calendar.sessions_in_range(start, end)
        if len(sessions) == 0:
            raise RuntimeError(
                f"XNYS calendar has no session in {start.isoformat()}..{end.isoformat()}"
            )
        return sessions[0].date()
