from datetime import date

from .models import SecurityMapping
from .tiingo import normalize_tiingo_ticker


class SecurityRegistry:
    """Point-in-time company/security mappings with ticker-reuse protection.

    Multiple legal companies may reference the same ``security_id`` over an
    overlapping provider-history interval. That is expected for successor or
    reincorporated entities. What remains forbidden is one ticker resolving to
    two *different* securities over the same interval.
    """

    def __init__(self) -> None:
        self._mappings: list[SecurityMapping] = []

    def add(self, mapping: SecurityMapping) -> None:
        normalized = mapping.model_copy(update={"ticker": normalize_tiingo_ticker(mapping.ticker)})
        for existing in self._mappings:
            if existing == normalized:
                return
            if existing.ticker != normalized.ticker:
                continue
            if (
                _intervals_overlap(existing, normalized)
                and existing.security_id != normalized.security_id
            ):
                raise ValueError(
                    f"ticker {normalized.ticker} overlaps multiple securities in the same period"
                )
        self._mappings.append(normalized)

    def security_for_ticker(self, ticker: str, day: date) -> str | None:
        normalized = normalize_tiingo_ticker(ticker)
        matches = [
            mapping.security_id
            for mapping in self._mappings
            if mapping.ticker == normalized and mapping.contains(day)
        ]
        unique = set(matches)
        if len(unique) > 1:
            raise ValueError(f"ambiguous security mapping for {normalized} on {day}")
        return next(iter(unique), None)

    def security_for_company(self, company_id: str, day: date) -> str | None:
        matches = [
            mapping.security_id
            for mapping in self._mappings
            if mapping.company_id == company_id and mapping.contains(day)
        ]
        unique = set(matches)
        if len(unique) > 1:
            raise ValueError(f"multiple active securities for {company_id} on {day}")
        return next(iter(unique), None)

    def company_for_ticker(self, ticker: str, day: date) -> str | None:
        """Compatibility helper that fails closed for legal-successor ambiguity."""

        normalized = normalize_tiingo_ticker(ticker)
        matches = [
            mapping.company_id
            for mapping in self._mappings
            if mapping.ticker == normalized and mapping.contains(day)
        ]
        unique = set(matches)
        if len(unique) > 1:
            raise ValueError(f"multiple companies reference {normalized} on {day}")
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

    def companies_for_security(self, security_id: str, day: date) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    mapping.company_id
                    for mapping in self._mappings
                    if mapping.security_id == security_id and mapping.contains(day)
                }
            )
        )

    def mappings(self) -> tuple[SecurityMapping, ...]:
        return tuple(self._mappings)


def _intervals_overlap(left: SecurityMapping, right: SecurityMapping) -> bool:
    left_end = left.valid_to or date.max
    right_end = right.valid_to or date.max
    return max(left.valid_from, right.valid_from) <= min(left_end, right_end)
