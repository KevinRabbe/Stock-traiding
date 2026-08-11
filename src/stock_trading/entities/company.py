from dataclasses import dataclass
from hashlib import sha256


def normalize_sec_cik(value: str) -> str:
    cik = str(value).strip()
    if not cik or not cik.isdigit():
        raise ValueError("SEC CIK must contain only digits")
    if len(cik) > 10:
        raise ValueError("SEC CIK must be at most 10 digits")
    return cik.zfill(10)


def company_id_from_sec_cik(cik: str) -> str:
    """Return a stable internal company ID anchored to a SEC CIK.

    External source identifiers are never exposed as the canonical company ID.
    Other source aliases can later map to this same ID through CompanyRegistry.
    """

    normalized = normalize_sec_cik(cik)
    digest = sha256(f"sec-cik:{normalized}".encode("utf-8")).hexdigest()[:20]
    return f"cmp_{digest}"


@dataclass(frozen=True, slots=True)
class CompanyIdentity:
    company_id: str
    canonical_name: str
    sec_cik: str


class CompanyRegistry:
    """Minimal canonical company registry for source-to-company resolution."""

    def __init__(self) -> None:
        self._by_id: dict[str, CompanyIdentity] = {}
        self._by_sec_cik: dict[str, str] = {}

    def register_sec_issuer(self, cik: str, name: str) -> CompanyIdentity:
        normalized_cik = normalize_sec_cik(cik)
        canonical_name = name.strip()
        if not canonical_name:
            raise ValueError("company name must not be empty")

        company_id = company_id_from_sec_cik(normalized_cik)
        existing_id = self._by_sec_cik.get(normalized_cik)
        if existing_id is not None:
            return self._by_id[existing_id]

        identity = CompanyIdentity(
            company_id=company_id,
            canonical_name=canonical_name,
            sec_cik=normalized_cik,
        )
        self._by_id[company_id] = identity
        self._by_sec_cik[normalized_cik] = company_id
        return identity

    def resolve_sec_cik(self, cik: str) -> str | None:
        return self._by_sec_cik.get(normalize_sec_cik(cik))

    def get(self, company_id: str) -> CompanyIdentity | None:
        return self._by_id.get(company_id)

    def __len__(self) -> int:
        return len(self._by_id)
