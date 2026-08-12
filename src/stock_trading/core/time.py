from datetime import datetime, timezone


def as_utc(value: datetime) -> datetime:
    """Return an aware datetime normalized to UTC.

    Naive datetimes are rejected rather than guessed. Source-specific parsers are
    responsible for applying the correct source timezone before entering core.
    """

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(timezone.utc)


def is_public_at(public_time: datetime, decision_time: datetime) -> bool:
    """Whether information was publicly available by a decision timestamp."""

    return as_utc(public_time) <= as_utc(decision_time)


def is_tradable_at(first_tradable_time: datetime | None, decision_time: datetime) -> bool:
    """Whether an event has reached its first permitted execution timestamp."""

    if first_tradable_time is None:
        return False
    return as_utc(first_tradable_time) <= as_utc(decision_time)
