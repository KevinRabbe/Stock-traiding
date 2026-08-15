import httpx
import pytest

from stock_trading.market.tiingo import TiingoAccountError, TiingoClient


def _client_for_status(status_code: int, *, retry_after: str | None = None) -> TiingoClient:
    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"Retry-After": retry_after} if retry_after is not None else None
        return httpx.Response(status_code, request=request, headers=headers)

    transport = httpx.MockTransport(handler)
    return TiingoClient("test-token", client=httpx.Client(transport=transport))


@pytest.mark.parametrize("status_code", [401, 403])
def test_tiingo_client_fails_fast_on_auth_or_account_errors(status_code: int) -> None:
    client = _client_for_status(status_code)

    with pytest.raises(TiingoAccountError) as caught:
        client.fetch_metadata("AAPL")

    assert caught.value.status_code == status_code
    assert "authentication or account access failed" in str(caught.value)


def test_tiingo_client_fails_fast_on_rate_limit_and_preserves_retry_hint() -> None:
    client = _client_for_status(429, retry_after="120")

    with pytest.raises(TiingoAccountError) as caught:
        client.fetch_metadata("AAPL")

    assert caught.value.status_code == 429
    assert caught.value.retry_after == "120"
    assert "request quota reached" in str(caught.value)
    assert "already cached" in str(caught.value)
    assert "Retry-After: 120" in str(caught.value)


def test_ticker_specific_http_errors_remain_http_errors() -> None:
    client = _client_for_status(404)

    with pytest.raises(httpx.HTTPStatusError) as caught:
        client.fetch_metadata("DEAD")

    assert caught.value.response.status_code == 404
