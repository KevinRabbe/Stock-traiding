import pytest

from stock_trading.market import normalize_tiingo_ticker


def test_normalize_tiingo_ticker_removes_common_sec_wrappers() -> None:
    assert normalize_tiingo_ticker('"PODD"') == "PODD"
    assert normalize_tiingo_ticker("(AFOP)") == "AFOP"
    assert normalize_tiingo_ticker("'LTRX") == "LTRX"
    assert normalize_tiingo_ticker("$FEED") == "FEED"


def test_normalize_tiingo_ticker_keeps_documented_dash_style() -> None:
    assert normalize_tiingo_ticker("brk.b") == "BRK-B"


def test_normalize_tiingo_ticker_still_rejects_arbitrary_junk() -> None:
    with pytest.raises(ValueError, match="unsupported characters"):
        normalize_tiingo_ticker("#rmvwt9d")
