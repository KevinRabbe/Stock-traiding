import os
import time
from datetime import date, datetime, timezone
from threading import Lock

import httpx

from stock_trading.core import RawRecord, Source, content_sha256


class LdaApiError(RuntimeError):
    """Source-specific LDA HTTP failure with authentication/rate-limit context."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        authenticated: bool,
        retry_after: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.authenticated = authenticated
        self.retry_after = retry_after


class LdaClient:
    BASE_URL = "https://lda.gov/api/v1"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        explicit_key = api_key.strip() if api_key else ""
        environment_key = os.environ.get("LDA_API_KEY", "").strip()
        legacy_key = os.environ.get("LDA_API_TOKEN", "").strip()
        self.api_key = explicit_key or environment_key or legacy_key or None
        self.requests_per_minute = 120 if self.api_key else 15
        self._minimum_interval = 60.0 / self.requests_per_minute
        self._lock = Lock()
        self._last_request_at = 0.0
        self._owns_client = client is None

        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Token {self.api_key}"
        self._client = client or httpx.Client(
            timeout=timeout_seconds,
            headers=headers,
            follow_redirects=True,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "LdaClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def fetch_filings_page(
        self,
        *,
        filing_year: int | None = None,
        posted_after: date | None = None,
        posted_before: date | None = None,
        page: int = 1,
        page_size: int = 25,
    ) -> RawRecord:
        if page <= 0:
            raise ValueError("page must be > 0")
        if not 1 <= page_size <= 25:
            raise ValueError("LDA page_size must be between 1 and 25")
        if filing_year is None and posted_after is None and posted_before is None:
            raise ValueError("at least one LDA filing filter is required")

        params: dict[str, str | int] = {
            "page": page,
            "page_size": page_size,
        }
        if filing_year is not None:
            params["filing_year"] = filing_year
        if posted_after is not None:
            params["filing_dt_posted_after"] = posted_after.isoformat()
        if posted_before is not None:
            params["filing_dt_posted_before"] = posted_before.isoformat()

        response = self._get("/filings/", params=params)
        content = response.content
        filter_id = ":".join(
            [
                f"year={filing_year or ''}",
                f"after={posted_after.isoformat() if posted_after else ''}",
                f"before={posted_before.isoformat() if posted_before else ''}",
                f"page={page}",
            ]
        )
        return RawRecord(
            source=Source.LDA,
            source_record_id=f"filings:{filter_id}",
            fetched_at=datetime.now(timezone.utc),
            content_type="application/json",
            content=content,
            sha256=content_sha256(content),
        )

    def fetch_filing(self, filing_uuid: str) -> RawRecord:
        filing_id = filing_uuid.strip()
        if not filing_id:
            raise ValueError("filing_uuid must not be empty")
        response = self._get(f"/filings/{filing_id}/")
        content = response.content
        return RawRecord(
            source=Source.LDA,
            source_record_id=f"filing:{filing_id}",
            fetched_at=datetime.now(timezone.utc),
            content_type="application/json",
            content=content,
            sha256=content_sha256(content),
        )

    def _get(self, path: str, *, params: dict | None = None) -> httpx.Response:
        self._throttle()
        response = self._client.get(f"{self.BASE_URL}{path}", params=params)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise _lda_api_error(response, authenticated=self.api_key is not None) from exc
        return response

    def _throttle(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._minimum_interval - (now - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = time.monotonic()


def _lda_api_error(response: httpx.Response, *, authenticated: bool) -> LdaApiError:
    status = response.status_code
    mode = "registered API key" if authenticated else "anonymous"
    retry_after = response.headers.get("retry-after")
    detail = _response_detail(response)

    if status == 403 and not authenticated:
        message = (
            "LDA API returned 403 Forbidden for anonymous access. LDA.gov documents "
            "anonymous API access, but it may throttle or deny a client/IP. Configure "
            "a registered key with LDA_API_KEY and retry."
        )
    elif status == 403:
        message = (
            "LDA API returned 403 Forbidden while using a registered API key. Verify "
            "that LDA_API_KEY is valid and active; the service may also have denied "
            "the client/IP."
        )
    elif status == 429:
        message = f"LDA API rate limit exceeded for {mode} access."
        if retry_after:
            message += f" Retry-After={retry_after}."
    else:
        message = f"LDA API returned HTTP {status} for {mode} access."

    if detail:
        message += f" Response: {detail}"
    return LdaApiError(
        message,
        status_code=status,
        authenticated=authenticated,
        retry_after=retry_after,
    )


def _response_detail(response: httpx.Response, *, limit: int = 300) -> str:
    try:
        text = response.text
    except Exception:  # pragma: no cover - httpx normally decodes response text
        return ""
    compact = " ".join(text.split())
    if len(compact) > limit:
        return compact[: limit - 3] + "..."
    return compact
