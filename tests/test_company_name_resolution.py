from datetime import date

from stock_trading.market import ConservativeTiingoResolver, IssuerObservation, normalize_company_name


def test_sec_name_markers_do_not_change_company_identity() -> None:
    assert normalize_company_name("COSTCO WHOLESALE CORP /NEW") == normalize_company_name("Costco Wholesale Corp")
    assert normalize_company_name("AWARE INC /MA/") == normalize_company_name("Aware Inc")


def test_share_class_descriptors_do_not_change_company_identity() -> None:
    assert normalize_company_name("Alphabet Inc.") == normalize_company_name("Alphabet Inc - Class A")
    assert normalize_company_name("Meta Platforms, Inc.") == normalize_company_name("Meta Platforms Inc CL A")


def test_unknown_issuer_name_remains_unresolved() -> None:
    resolution = ConservativeTiingoResolver().resolve(
        IssuerObservation(
            sec_cik="789019",
            issuer_name="UNKNOWN",
            ticker="MSFT",
            observed_date=date(2022, 3, 2),
        ),
        tiingo_ticker="MSFT",
        tiingo_name="Microsoft Corp",
        tiingo_start=date(1986, 3, 13),
        tiingo_end=None,
        exchange_code="NASDAQ",
    )
    assert not resolution.resolved
    assert resolution.reason == "issuer_name_unavailable"


def test_historical_rename_is_not_guessed() -> None:
    resolution = ConservativeTiingoResolver().resolve(
        IssuerObservation(
            sec_cik="104169",
            issuer_name="WAL MART STORES INC",
            ticker="WMT",
            observed_date=date(2012, 1, 3),
        ),
        tiingo_ticker="WMT",
        tiingo_name="Walmart Inc",
        tiingo_start=date(1972, 8, 25),
        tiingo_end=None,
        exchange_code="NYSE",
    )
    assert not resolution.resolved
    assert resolution.reason == "company_name_mismatch"
