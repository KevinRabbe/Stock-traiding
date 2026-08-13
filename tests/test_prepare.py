from datetime import date

import pytest

from stock_trading.experiments.prepare import (
    _parser,
    _select_observations,
    estimate_tiingo_requests,
    latest_completed_quarter,
    quarter_range,
)
from stock_trading.market import IssuerObservation
from stock_trading.ml import TrainingDatasetBuilder


def test_latest_completed_quarter() -> None:
    assert latest_completed_quarter(date(2026, 8, 12)) == (2026, 2)
    assert latest_completed_quarter(date(2026, 4, 1)) == (2026, 1)
    assert latest_completed_quarter(date(2026, 1, 15)) == (2025, 4)


def test_quarter_range_crosses_year_boundary() -> None:
    assert quarter_range(2025, 3, 2026, 2) == (
        (2025, 3),
        (2025, 4),
        (2026, 1),
        (2026, 2),
    )


def test_estimated_tiingo_requests_are_explicit() -> None:
    assert estimate_tiingo_requests(50) == {
        "metadata": 50,
        "price_series": 50,
        "benchmark": 1,
        "minimum_total": 101,
    }
    with pytest.raises(ValueError, match=">= 0"):
        estimate_tiingo_requests(-1)


def test_sec_only_cli_does_not_require_market_arguments() -> None:
    args = _parser().parse_args(
        [
            "--sec-only",
            "--sec-user-agent",
            "Stock-traiding test@example.com",
        ]
    )
    assert args.sec_only is True
    assert args.start_year == 2012
    assert args.max_unique_tickers is None
    assert args.tickers is None
    assert args.refresh_sec_raw is False


def test_sec_raw_cache_can_be_explicitly_refreshed() -> None:
    args = _parser().parse_args(
        [
            "--sec-only",
            "--refresh-sec-raw",
            "--sec-user-agent",
            "Stock-traiding test@example.com",
        ]
    )
    assert args.refresh_sec_raw is True


def test_explicit_ticker_cli_is_mutually_exclusive_with_lexical_limit() -> None:
    args = _parser().parse_args(
        [
            "--tickers",
            "AAPL",
            "MSFT",
            "NVDA",
            "--sec-user-agent",
            "Stock-traiding test@example.com",
        ]
    )
    assert args.tickers == ["AAPL", "MSFT", "NVDA"]
    assert args.max_unique_tickers is None

    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "--tickers",
                "AAPL",
                "--max-unique-tickers",
                "20",
                "--sec-user-agent",
                "Stock-traiding test@example.com",
            ]
        )


def test_explicit_ticker_selection_uses_market_canonicalization_and_reports_missing() -> None:
    observations = (
        IssuerObservation(
            sec_cik="1001",
            issuer_name="Apple Inc",
            ticker='"AAPL"',
            observed_date=date(2020, 1, 1),
        ),
        IssuerObservation(
            sec_cik="1002",
            issuer_name="Microsoft Corp",
            ticker="MSFT",
            observed_date=date(2020, 1, 1),
        ),
    )

    selected, unique_count, requested, missing = _select_observations(
        observations,
        max_unique_tickers=None,
        requested_tickers=("aapl", "NVDA"),
    )

    assert selected == (observations[0],)
    assert unique_count == 1
    assert requested == ("AAPL", "NVDA")
    assert missing == ("NVDA",)


def test_training_dataset_rejects_non_20_day_target() -> None:
    with pytest.raises(ValueError, match="20-day"):
        TrainingDatasetBuilder(object(), target_horizon=60)
