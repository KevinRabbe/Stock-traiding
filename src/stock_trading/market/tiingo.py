from datetime import date, datetime, timezone
from urllib.parse import quote

import httpx

from stock_trading.core import RawRecord, Source, content_sha256


class TiingoClient:
    BASE_URL = "https://api.tiingo.com"

    def __init__(
        self,
        token: str,
        *,
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        token = token.strip()
        if not token:
            raise ValueError("Tiingo API token must not be empty")
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=timeout_seconds,
            headers={
                "Authorization": f"Token {token}",
                "Content-Type": "application/json",
            },
            follow_redirects=True,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "TiingoClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def fetch_metadata(self, ticker: str) -> RawRecord:
        normalized = normalize_tiingo_ticker(ticker)
        response = self._get(f"/tiingo/daily/{quote(normalized, safe='-')}")
        content = response.content
        return RawRecord(
            source=Source.TIINGO,
            source_record_id=f"metadata:{normalized}",
            fetched_at=datetime.now(timezone.utc),
            content_type="application/json",
            content=content,
            sha256=content_sha256(content),
        )

    def fetch_prices(self, ticker: str, start: date, end: date) -> RawRecord:
        if end < start:
            raise ValueError("end must be >= start")
        normalized = normalize_tiingo_ticker(ticker)
        response = self._get(
            f"/tiingo/daily/{quote(normalized, safe='-')}/prices",
            params={"startDate": start.isoformat(), "endDate": end.isoformat()},
        )
        content = response.content
        return RawRecord(
            source=Source.TIINGO,
            source_record_id=f"prices:{normalized}:{start.isoformat()}:{end.isoformat()}",
            fetched_at=datetime.now(timezone.utc),
            content_type="application/json",
            content=content,
            sha256=content_sha256(content),
        )

    def _get(self, path: str, *, params: dict | None = None) -> httpx.Response:
        response = self._client.get(f"{self.BASE_URL}{path}", params=params)
        response.raise_for_status()
        return response


_ALLOWED_TICKER_CHARACTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")
_BALANCED_TICKER_WRAPPERS = {'"': '"', "'": "'", "(": ")"}


def normalize_tiingo_ticker(value: str) -> str:
    """Normalize narrow SEC presentation artifacts without guessing junk symbols."""
    ticker = value.strip().upper()
    if not ticker:
        raise ValueError("ticker must not be empty")

    if len(ticker) >= 2 and _BALANCED_TICKER_WRAPPERS.get(ticker[0]) == ticker[-1]:
        ticker = ticker[1:-1].strip()

    if ticker.startswith(("'", "$")):
        candidate = ticker[1:].strip().replace(".", "-")
        if candidate and all(character in _ALLOWED_TICKER_CHARACTERS for character in candidate):
            ticker = ticker[1:].strip()

    ticker = ticker.replace(".", "-")
    if not ticker:
        raise ValueError("ticker must not be empty")
    if any(character not in _ALLOWED_TICKER_CHARACTERS for character in ticker):
        raise ValueError("ticker contains unsupported characters")
    return ticker
