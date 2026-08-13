import re
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from stock_trading.entities import company_id_from_sec_cik, normalize_sec_cik

from .models import SecurityMapping
from .tiingo import normalize_tiingo_ticker


class ResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class IssuerObservation:
    """Point-in-time issuer identity observed in a source filing."""

    sec_cik: str
    issuer_name: str
    ticker: str
    observed_date: date

    @property
    def company_id(self) -> str:
        return company_id_from_sec_cik(self.sec_cik)


@dataclass(frozen=True, slots=True)
class SecurityResolution:
    status: ResolutionStatus
    observation: IssuerObservation
    mapping: SecurityMapping | None
    reason: str

    @property
    def resolved(self) -> bool:
        return self.status is ResolutionStatus.RESOLVED


class ConservativeTiingoResolver:
    """Resolve historical SEC issuer observations without guessing ticker reuse.

    V1 requires all of the following:
    - the SEC and Tiingo symbols normalize to the same ticker;
    - the SEC observation date falls inside Tiingo's available date interval;
    - the normalized company names match after conservative source-artifact cleanup.

    Anything else remains unresolved for later manual/permaTicker handling.
    """

    def resolve(
        self,
        observation: IssuerObservation,
        *,
        tiingo_ticker: str,
        tiingo_name: str,
        tiingo_start: date,
        tiingo_end: date | None,
        exchange_code: str | None = None,
    ) -> SecurityResolution:
        normalized_cik = normalize_sec_cik(observation.sec_cik)
        normalized_observation = IssuerObservation(
            sec_cik=normalized_cik,
            issuer_name=observation.issuer_name.strip(),
            ticker=normalize_tiingo_ticker(observation.ticker),
            observed_date=observation.observed_date,
        )
        normalized_tiingo_ticker = normalize_tiingo_ticker(tiingo_ticker)

        if normalized_observation.ticker != normalized_tiingo_ticker:
            return SecurityResolution(
                ResolutionStatus.UNRESOLVED,
                normalized_observation,
                None,
                "ticker_mismatch",
            )

        if normalized_observation.observed_date < tiingo_start:
            return SecurityResolution(
                ResolutionStatus.UNRESOLVED,
                normalized_observation,
                None,
                "observation_predates_tiingo_history",
            )
        if tiingo_end is not None and normalized_observation.observed_date > tiingo_end:
            return SecurityResolution(
                ResolutionStatus.UNRESOLVED,
                normalized_observation,
                None,
                "observation_after_tiingo_history",
            )

        observation_name = normalize_company_name(normalized_observation.issuer_name)
        if observation_name in _UNAVAILABLE_COMPANY_NAMES:
            return SecurityResolution(
                ResolutionStatus.UNRESOLVED,
                normalized_observation,
                None,
                "issuer_name_unavailable",
            )

        if observation_name != normalize_company_name(tiingo_name):
            return SecurityResolution(
                ResolutionStatus.UNRESOLVED,
                normalized_observation,
                None,
                "company_name_mismatch",
            )

        mapping = SecurityMapping(
            company_id=normalized_observation.company_id,
            ticker=normalized_tiingo_ticker,
            exchange_code=exchange_code,
            valid_from=tiingo_start,
            valid_to=tiingo_end,
        )
        return SecurityResolution(
            ResolutionStatus.RESOLVED,
            normalized_observation,
            mapping,
            "ticker_date_name_match",
        )


_LEGAL_SUFFIXES = {
    "AG",
    "CO",
    "COMPANY",
    "CORP",
    "CORPORATION",
    "INC",
    "INCORPORATED",
    "LTD",
    "LIMITED",
    "LLC",
    "LP",
    "PLC",
    "SA",
}
_SHARE_CLASS_WORDS = {"CLASS", "CL"}
_SHARE_CLASS_VALUES = {"A", "B", "C"}
_UNAVAILABLE_COMPANY_NAMES = {"", "UNKNOWN", "N A", "NA", "NOT AVAILABLE"}
_SEC_TRAILING_MARKER = re.compile(r"\s*/(?:NEW|[A-Z]{2})/?\s*$", re.IGNORECASE)


def normalize_company_name(value: str) -> str:
    cleaned = value.upper().replace("&", " AND ").strip()

    # SEC issuer names sometimes carry jurisdiction/reincorporation markers such
    # as ``/DE/``, ``/MA/`` or ``/NEW``. They are source presentation metadata,
    # not part of the legal company identity.
    while True:
        without_marker = _SEC_TRAILING_MARKER.sub("", cleaned).strip()
        if without_marker == cleaned:
            break
        cleaned = without_marker

    tokens = re.findall(r"[A-Z0-9]+", cleaned)

    # Tiingo metadata can append a security-class descriptor to the company
    # name (for example ``Alphabet Inc - Class A``). The ticker and date checks
    # already identify the security; the descriptor should not make the company
    # identity comparison fail.
    if (
        len(tokens) >= 2
        and tokens[-2] in _SHARE_CLASS_WORDS
        and tokens[-1] in _SHARE_CLASS_VALUES
    ):
        tokens = tokens[:-2]

    while tokens and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)
