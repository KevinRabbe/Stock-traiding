import pytest

from stock_trading.experiments.market_prepare import (
    DEFAULT_RATE_LIMIT_WAIT_SECONDS,
    _retry_delay_seconds,
    _run_with_rate_limit_resume,
)
from stock_trading.market.tiingo import TiingoAccountError


def test_rate_limit_resume_retries_after_retry_after_hint() -> None:
    attempts = 0
    sleeps: list[int] = []

    def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TiingoAccountError(429, "https://api.tiingo.com/test", retry_after="17")
        return "complete"

    result = _run_with_rate_limit_resume(
        operation,
        wait_on_rate_limit=True,
        sleep_fn=sleeps.append,
    )

    assert result == "complete"
    assert attempts == 2
    assert sleeps == [17]


def test_rate_limit_resume_uses_hour_fallback_without_retry_after() -> None:
    error = TiingoAccountError(429, "https://api.tiingo.com/test")

    assert _retry_delay_seconds(error) == DEFAULT_RATE_LIMIT_WAIT_SECONDS


def test_rate_limit_without_wait_flag_still_fails_fast() -> None:
    error = TiingoAccountError(429, "https://api.tiingo.com/test")

    with pytest.raises(TiingoAccountError) as caught:
        _run_with_rate_limit_resume(
            lambda: (_ for _ in ()).throw(error),
            wait_on_rate_limit=False,
            sleep_fn=lambda _: pytest.fail("must not sleep"),
        )

    assert caught.value is error


def test_auth_error_never_waits() -> None:
    error = TiingoAccountError(403, "https://api.tiingo.com/test")

    with pytest.raises(TiingoAccountError) as caught:
        _run_with_rate_limit_resume(
            lambda: (_ for _ in ()).throw(error),
            wait_on_rate_limit=True,
            sleep_fn=lambda _: pytest.fail("must not sleep"),
        )

    assert caught.value is error
