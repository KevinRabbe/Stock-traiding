from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from .events import Event
from .raw import RawRecord


@runtime_checkable
class Collector(Protocol):
    """Source acquisition boundary.

    Collectors fetch source data only. They must not build trading features,
    infer alpha, or perform semantic interpretation.
    """

    def backfill(self, start: datetime, end: datetime) -> Iterable[RawRecord]:
        ...

    def poll(self, since: datetime) -> Iterable[RawRecord]:
        ...


@runtime_checkable
class Normalizer(Protocol):
    """Translate one immutable raw record into zero or more canonical events."""

    def normalize(self, raw: RawRecord) -> Sequence[Event]:
        ...
