from datetime import date

from stock_trading.entities import company_id_from_sec_cik
from stock_trading.market import (
    IssuerObservation,
    ResolutionStatus,
    SecurityMapping,
    SecurityResolution,
    normalize_company_name,
)
from stock_trading.market.backfill import _promote_validated_company_aliases


def test_company_name_normalization_ignores_presentation_only_suffixes() -> None:
    assert normalize_company_name("COSTCO WHOLESALE CORP /NEW") == normalize_company_name(
        "Costco Wholesale Corporation"
    )
    assert normalize_company_name("AWARE INC /MA/") == normalize_company_name("Aware Inc")
    assert normalize_company_name("Alphabet Inc.") == normalize_company_name(
        "Alphabet Inc - Class A"
    )
    assert normalize_company_name("Meta Platforms, Inc.") == normalize_company_name(
        "Meta Platforms Inc Class A"
    )
    assert normalize_company_name("WALT DISNEY CO/") == normalize_company_name(
        "Walt Disney Co (The)"
    )
    assert normalize_company_name("MCDONALDS CORP") == normalize_company_name(
        "McDonald's Corp"
    )
    assert normalize_company_name("MCDONALDS CORP") == normalize_company_name(
        "McDonald’s Corp"
    )
    assert normalize_company_name("MCDONALDS CORP") == normalize_company_name(
        "McDonald`s Corp"
    )


def test_alias_promotion_requires_same_cik_ticker_and_validated_name_match() -> None:
    company_id = company_id_from_sec_cik("12345")
    mapping = SecurityMapping(
        company_id=company_id,
        security_id="security_good",
        ticker="GOOD",
        valid_from=date(2010, 1, 1),
        valid_to=None,
    )
    exact = SecurityResolution(
        ResolutionStatus.RESOLVED,
        IssuerObservation(
            sec_cik="12345",
            issuer_name="Good Corp",
            ticker="GOOD",
            observed_date=date(2020, 1, 2),
        ),
        mapping,
        "ticker_date_name_match",
    )
    same_company_alias = SecurityResolution(
        ResolutionStatus.UNRESOLVED,
        IssuerObservation(
            sec_cik="12345",
            issuer_name="UNKNOWN",
            ticker="GOOD",
            observed_date=date(2021, 1, 2)),
        None,
        "company_name_mismatch",
    )
    different_company = SecurityResolution(
        ResolutionStatus.UNRESOLVED,
        IssuerObservation(
            sec_cik="99999",
            issuer_name="UNKNOWN",
            ticker="GOOD",
            observed_date=date(2021, 1, 2)),
        None,
        "company_name_mismatch",
    )

    promoted = _promote_validated_company_aliases(
        (same_company_alias, exact, different_company)
    )

    assert promoted[0].resolved
    assert promoted[0].mapping == mapping
    assert promoted[0].reason == "sec_cik_ticker_validated_by_matching_alias"
    assert promoted[1] == exact
    assert not promoted[2].resolved
    assert promoted[2].reason == "company_name_mismatch"
