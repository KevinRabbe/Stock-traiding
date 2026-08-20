from datetime import date

import httpx
import pytest

from stock_trading.lobbying import LdaApiError, LdaClient


def _status_client(status: int, *, body: str = "", headers: dict[str, str] | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, request=request, text=body, headers=headers)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_lda_client_prefers_official_api_key_environment_name(monkeypatch) -> None:
    monkeypatch.setenv("LDA_API_KEY", "registered-key")
    monkeypatch.setenv("LDA_API_TOKEN", "legacy-key")

    with LdaClient() as client:
        assert client.api_key == "registered-key"
        assert client.requests_per_minute == 120
        assert client._client.headers["Authorization"] == "Token registered-key"  # noqa: SLF001


def test_lda_client_explicit_key_overrides_environment(monkeypatch) -> None:
    monkeypatch.setenv("LDA_API_KEY", "environment-key")

    with LdaClient(api_key="explicit-key") as client:
        assert client.api_key == "explicit-key"
        assert client._client.headers["Authorization"] == "Token explicit-key"  # noqa: SLF001


def test_lda_client_keeps_legacy_token_environment_alias(monkeypatch) -> None:
    monkeypatch.delenv("LDA_API_KEY", raising=False)
    monkeypatch.setenv("LDA_API_TOKEN", "legacy-key")

    with LdaClient() as client:
        assert client.api_key == "legacy-key"
        assert client.requests_per_minute == 120


def test_lda_anonymous_403_explains_registered_key_path(monkeypatch) -> None:
    monkeypatch.delenv("LDA_API_KEY", raising=False)
    monkeypatch.delenv("LDA_API_TOKEN", raising=False)
    http_client = _status_client(403, body="Forbidden")
    client = LdaClient(client=http_client)

    with pytest.raises(LdaApiError, match="LDA_API_KEY") as raised:
        client.fetch_filings_page(
            posted_after=date(2026, 8, 18),
            posted_before=date(2026, 8, 20),
        )

    assert raised.value.status_code == 403
    assert raised.value.authenticated is False
    assert "anonymous access" in str(raised.value)
    assert "Forbidden" in str(raised.value)
    http_client.close()


def test_lda_authenticated_403_reports_key_rejection(monkeypatch) -> None:
    monkeypatch.setenv("LDA_API_KEY", "registered-key")
    http_client = _status_client(403, body="Access denied")
    client = LdaClient(client=http_client)

    with pytest.raises(LdaApiError, match="registered API key") as raised:
        client.fetch_filings_page(filing_year=2026)

    assert raised.value.status_code == 403
    assert raised.value.authenticated is True
    assert "valid and active" in str(raised.value)
    http_client.close()


def test_lda_429_preserves_retry_after(monkeypatch) -> None:
    monkeypatch.delenv("LDA_API_KEY", raising=False)
    monkeypatch.delenv("LDA_API_TOKEN", raising=False)
    http_client = _status_client(429, body="Too many requests", headers={"Retry-After": "60"})
    client = LdaClient(client=http_client)

    with pytest.raises(LdaApiError, match="rate limit") as raised:
        client.fetch_filings_page(filing_year=2026)

    assert raised.value.status_code == 429
    assert raised.value.retry_after == "60"
    assert "Retry-After=60" in str(raised.value)
    http_client.close()
