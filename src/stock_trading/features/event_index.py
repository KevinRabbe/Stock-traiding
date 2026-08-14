from bisect import bisect_right
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta

from stock_trading.core import Event, EventType, as_utc


class CompanyEventIndex:
    """Sorted per-company event index for repeated point-in-time feature queries.

    Historical feature construction asks the same company history questions at
    many consecutive decision times. Re-scanning and re-sorting the full company
    history for every opportunity is quadratic work. This index materializes the
    canonical company history once and answers temporal windows with binary
    search while preserving the exact public-time boundaries used by the feature
    builders.
    """

    __slots__ = ("company_id", "_events_by_type", "_times_by_type", "event_count")

    def __init__(self, company_id: str, events: Iterable[Event]) -> None:
        if not company_id:
            raise ValueError("company_id is required")
        grouped: dict[EventType, list[Event]] = {}
        count = 0
        for event in events:
            if event.company_id != company_id:
                continue
            grouped.setdefault(event.event_type, []).append(event)
            count += 1

        events_by_type: dict[EventType, tuple[Event, ...]] = {}
        times_by_type: dict[EventType, tuple[datetime, ...]] = {}
        for event_type, typed_events in grouped.items():
            ordered = tuple(
                sorted(
                    typed_events,
                    key=lambda event: (as_utc(event.public_time), event.event_id),
                )
            )
            events_by_type[event_type] = ordered
            times_by_type[event_type] = tuple(as_utc(event.public_time) for event in ordered)

        self.company_id = company_id
        self._events_by_type = events_by_type
        self._times_by_type = times_by_type
        self.event_count = count

    def all_of_type(self, event_type: EventType) -> tuple[Event, ...]:
        return self._events_by_type.get(event_type, ())

    def through(self, event_type: EventType, decision_time: datetime) -> tuple[Event, ...]:
        """Return events with ``public_time <= decision_time``."""

        decision = as_utc(decision_time)
        events = self._events_by_type.get(event_type, ())
        times = self._times_by_type.get(event_type, ())
        return events[: bisect_right(times, decision)]

    def within(
        self,
        event_type: EventType,
        decision_time: datetime,
        days: int,
    ) -> tuple[Event, ...]:
        """Return events in ``decision-days < public_time <= decision``."""

        if days < 0:
            raise ValueError("days must be >= 0")
        decision = as_utc(decision_time)
        cutoff = decision - timedelta(days=days)
        events = self._events_by_type.get(event_type, ())
        times = self._times_by_type.get(event_type, ())
        left = bisect_right(times, cutoff)
        right = bisect_right(times, decision)
        return events[left:right]

    def between(
        self,
        event_type: EventType,
        decision_time: datetime,
        older_days: int,
        newer_days: int,
    ) -> tuple[Event, ...]:
        """Return events in ``decision-older < public_time <= decision-newer``."""

        if older_days < newer_days or newer_days < 0:
            raise ValueError("require older_days >= newer_days >= 0")
        decision = as_utc(decision_time)
        older = decision - timedelta(days=older_days)
        newer = decision - timedelta(days=newer_days)
        events = self._events_by_type.get(event_type, ())
        times = self._times_by_type.get(event_type, ())
        left = bisect_right(times, older)
        right = bisect_right(times, newer)
        return events[left:right]

    def latest(self, event_type: EventType, decision_time: datetime) -> Event | None:
        decision = as_utc(decision_time)
        events = self._events_by_type.get(event_type, ())
        times = self._times_by_type.get(event_type, ())
        right = bisect_right(times, decision)
        return events[right - 1] if right else None

    def latest_matching(
        self,
        event_type: EventType,
        decision_time: datetime,
        predicate: Callable[[Event], bool],
    ) -> Event | None:
        """Find the latest visible event satisfying a feature-specific predicate."""

        decision = as_utc(decision_time)
        events = self._events_by_type.get(event_type, ())
        times = self._times_by_type.get(event_type, ())
        right = bisect_right(times, decision)
        for index in range(right - 1, -1, -1):
            event = events[index]
            if predicate(event):
                return event
        return None

    def has_at_or_before(self, event_type: EventType, cutoff_time: datetime) -> bool:
        cutoff = as_utc(cutoff_time)
        times = self._times_by_type.get(event_type, ())
        return bisect_right(times, cutoff) > 0


def ensure_company_event_index(
    events: Iterable[Event] | CompanyEventIndex,
    *,
    company_id: str,
) -> CompanyEventIndex:
    if isinstance(events, CompanyEventIndex):
        if events.company_id != company_id:
            raise ValueError(
                f"event index company mismatch: {events.company_id} != {company_id}"
            )
        return events
    return CompanyEventIndex(company_id, events)
