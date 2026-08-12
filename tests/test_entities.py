import pytest

from stock_trading.entities import (
    CompanyRegistry,
    company_id_from_sec_cik,
    normalize_sec_cik,
)


def test_sec_cik_normalization_and_company_id_are_stable() -> None:
    assert normalize_sec_cik("12345") == "0000012345"
    assert company_id_from_sec_cik("12345") == company_id_from_sec_cik("0000012345")
    assert company_id_from_sec_cik("12345").startswith("cmp_")
    assert "12345" not in company_id_from_sec_cik("12345")


def test_company_registry_deduplicates_same_sec_issuer() -> None:
    registry = CompanyRegistry()
    first = registry.register_sec_issuer("12345", "Example Corp")
    second = registry.register_sec_issuer("0000012345", "Example Corporation")

    assert first == second
    assert len(registry) == 1
    assert registry.resolve_sec_cik("12345") == first.company_id
    assert registry.get(first.company_id) == first


def test_invalid_sec_cik_is_rejected() -> None:
    with pytest.raises(ValueError, match="digits"):
        normalize_sec_cik("12A45")
