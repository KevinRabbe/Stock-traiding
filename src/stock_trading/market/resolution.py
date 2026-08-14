import hashlib
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
    - the normalized company names match after conservative legal-suffix cleanup.

    The resulting ``security_id`` identifies the Tiingo security history rather
    than the SEC legal entity. This is deliberate: a continuous traded security
    can survive a legal-company reorganization and therefore be shared by more
    than one SEC CIK without duplicating or mis-owning its market bars.

    Anything else remains unresolved for later manual/permaTicker/succession
    handling.
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

        if normalize_company_name(normalized_observation.issuer_name) != normalize_company_name(
            tiingo_name
        ):
            return SecurityResolution(
                ResolutionStatus.UNRESOLVED,
                normalized_observation,
                None,
                "company_name_mismatch",
            )

        mapping = SecurityMapping(
            company_id=normalized_observation.company_id,
            security_id=tiingo_security_id(normalized_tiingo_ticker, tiingo_start),
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
_SEC_TRAILING_QUALIFIER = re.compile(r"\s*/(?:[A-Z]{2}|NEW)/?\s*$", re.IGNORECASE)
_SHARE_CLASS_MARKERS = {"CLASS", "CL"}


def tiingo_security_id(ticker: str, history_start: date) -> str:
    """Return a stable ID for one Tiingo EOD security history.

    The ID intentionally excludes company name and SEC CIK so legal successor
    entities can reference the same price series. ``history_start`` prevents a
    later reuse of the same ticker from silently becoming the same security.
    Active-history end dates are excluded because they change over time.
    """

    normalized_ticker = normalize_tiingo_ticker(ticker)
    material = f"tiingo-eod|{normalized_ticker}|{history_start.isoformat()}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
    return f"security_tiingo_{digest}"


def normalize_company_name(value: str) -> str:
    """Normalize issuer names without erasing substantive corporate identity.

    SEC flattened filings sometimes append presentation-only jurisdiction or
    reincorporation markers such as ``/MA/`` or ``/NEW``. Market metadata may
    also append a share-class descriptor (for example ``Class A``) to the
    corporate issuer name. Neither should make an otherwise exact issuer-name
    comparison fail once ticker and point-in-time date checks already agree.
    """

    cleaned = _SEC_TRAILING_QUALIFIER.sub("", value.strip())
    tokens = re.findall(r"[A-Z0-9]+", cleaned.upper().replace("&", " AND "))

    if (
        len(tokens) >= 2
        and tokens[-2] in _SHARE_CLASS_MARKERS
        and 1 <= len(tokens[-1]) <= 3
    ):
        tokens = tokens[:-2]

    while tokens and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)
