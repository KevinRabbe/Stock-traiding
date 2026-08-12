from datetime import date

from .models import SecurityMapping
from .tiingo import normalize_tiingo_ticker


class SecurityRegistry:
    """Point-in-time ticker mappings with overlap protection."""

    def __init__(self) -> None:
        self._mappings: list[SecurityMapping] = []

    def add(self, mapping: SecurityMapping) -> None:
        normalized = mapping.model_copy(update={"ticker": normalize_tiingo_ticker(mapping.ticker)})
        for existing in self._mappings:
            if existing == normalized:
                return
            if existing.ticker != normalized.ticker:
                continue
            if _intervals_overlap(existing, normalized) and existing.company_id != normalized.company_id:
                raise ValueError(
                    f"ticker {normalized.ticker} overlaps multiple companies in the same period"
                )
        self._mappings.append(normalized)

    def company_for_ticker(self, ticker: str, day: date) -> str | None:
        normalized = normalize_tiingo_ticker(ticker)
        matches = [
            mapping.company_id
            for mapping in self._mappings
            if mapping.ticker == normalized and mapping.contains(day)
        ]
        unique = set(matches)
        if len(unique) > 1:
            raise ValueError(f"ambiguous ticker mapping for {normalized} on {day}")
        return next(iter(unique), None)

    def ticker_for_company(self, company_id: str, day: date) -> str | None:
        matches = [
            mapping.ticker
            for mapping in self._mappings
            if mapping.company_id == company_id and mapping.contains(day)
        ]
        unique = set(matches)
        if len(unique) > 1:
            raise ValueError(f"multiple active ticker mappings for {company_id} on {day}")
        return next(iter(unique), None)

    def mappings(self) -> tuple[SecurityMapping, ...]:
        return tuple(self._mappings)


def _intervals_overlap(left: SecurityMapping, right: SecurityMapping) -> bool:
    left_end = left.valid_to or date.max
    right_end = right.valid_to or date.max
    return max(left.valid_from, right.valid_from) <= min(left_end, right_end)
