import json
import os
import time
from datetime import date, datetime, timedelta, timezone
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
        edge_denied: bool = False,
        base_url: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.authenticated = authenticated
        self.retry_after = retry_after
        self.edge_denied = edge_denied
        self.base_url = base_url


class LdaRelayError(RuntimeError):
    """The public GitHub LDA relay is unavailable, stale, or incomplete."""


class LdaClient:
    DEFAULT_BASE_URL = "https://lda.gov/api/v1"
    DEFAULT_RELAY_URL = (
        "https://raw.githubusercontent.com/KevinRabbe/Stock-traiding/"
        "lda-feed/lda_snapshot.json"
    )
    DEFAULT_RELAY_MAX_AGE_HOURS = 18.0
    RELAY_SCHEMA_VERSION = 1
    DEFAULT_USER_AGENT = (
        "Stock-traiding/0.1 "
        "(+https://github.com/KevinRabbe/Stock-traiding; LDA research client)"
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        user_agent: str | None = None,
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
        relay_fallback: bool = True,
        relay_url: str | None = None,
        relay_max_age_hours: float = DEFAULT_RELAY_MAX_AGE_HOURS,
        relay_client: httpx.Client | None = None,
    ) -> None:
        explicit_key = api_key.strip() if api_key else ""
        environment_key = os.environ.get("LDA_API_KEY", "").strip()
        legacy_key = os.environ.get("LDA_API_TOKEN", "").strip()
        self.api_key = explicit_key or environment_key or legacy_key or None

        resolved_base_url = (
            base_url or os.environ.get("LDA_BASE_URL", "") or self.DEFAULT_BASE_URL
        ).strip().rstrip("/")
        if not resolved_base_url.startswith("https://"):
            raise ValueError("LDA base URL must use https://")
        self.base_url = resolved_base_url

        resolved_user_agent = (
            user_agent or os.environ.get("LDA_USER_AGENT", "") or self.DEFAULT_USER_AGENT
        ).strip()
        if not resolved_user_agent:
            raise ValueError("LDA user agent must not be empty")
        self.user_agent = resolved_user_agent

        if relay_max_age_hours <= 0:
            raise ValueError("relay_max_age_hours must be > 0")
        self.relay_max_age_hours = float(relay_max_age_hours)
        if relay_fallback:
            resolved_relay_url = (
                relay_url or os.environ.get("LDA_RELAY_URL", "") or self.DEFAULT_RELAY_URL
            ).strip()
            if not resolved_relay_url.startswith("https://"):
                raise ValueError("LDA relay URL must use https://")
            self.relay_url: str | None = resolved_relay_url
        else:
            self.relay_url = None

        self.requests_per_minute = 120 if self.api_key else 15
        self._minimum_interval = 60.0 / self.requests_per_minute
        self._lock = Lock()
        self._last_request_at = 0.0
        self._owns_client = client is None
        self._owns_relay_client = relay_client is None
        self._edge_blocked = False
        self._relay_snapshot: dict | None = None
        self.transport_mode = "direct"
        self.relay_snapshot_generated_at: datetime | None = None

        headers = {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
        if self.api_key:
            headers["Authorization"] = f"Token {self.api_key}"
        self._client = client or httpx.Client(
            timeout=timeout_seconds,
            headers=headers,
            follow_redirects=True,
        )
        # Never reuse the LDA-authenticated client for the GitHub relay: doing so
        # would leak the Senate API key to a different host via its default header.
        self._relay_client = relay_client or httpx.Client(
            timeout=timeout_seconds,
            headers={
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
            follow_redirects=True,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
        if self._owns_relay_client:
            self._relay_client.close()

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
        if self._edge_blocked and self.relay_url is not None:
            return self._get_from_relay(path, params=params)

        self._throttle()
        response = self._client.get(f"{self.base_url}{path}", params=params)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            api_error = _lda_api_error(
                response,
                authenticated=self.api_key is not None,
                base_url=self.base_url,
            )
            if api_error.edge_denied and self.relay_url is not None:
                self._edge_blocked = True
                try:
                    return self._get_from_relay(path, params=params)
                except Exception as relay_exc:
                    raise LdaRelayError(
                        "LDA direct access was edge-blocked and the GitHub relay could not "
                        f"satisfy the request: {relay_exc}"
                    ) from api_error
            raise api_error from exc
        self.transport_mode = "direct"
        return response

    def _get_from_relay(self, path: str, *, params: dict | None) -> httpx.Response:
        snapshot = self._load_relay_snapshot()
        self.transport_mode = "relay"
        if path == "/filings/":
            return self._relay_filings_response(snapshot, params=params or {})
        prefix = "/filings/"
        suffix = "/"
        if path.startswith(prefix) and path.endswith(suffix):
            filing_uuid = path[len(prefix) : -len(suffix)].strip()
            if filing_uuid:
                for filing in snapshot["filings"]:
                    if str(filing.get("filing_uuid") or "").strip() == filing_uuid:
                        return self._json_response(filing, path=path)
                raise LdaRelayError(
                    f"filing {filing_uuid} is not present in the rolling relay snapshot"
                )
        raise LdaRelayError(f"relay does not support LDA path: {path}")

    def _load_relay_snapshot(self) -> dict:
        if self._relay_snapshot is not None:
            return self._relay_snapshot
        if self.relay_url is None:
            raise LdaRelayError("relay fallback is disabled")

        response = self._relay_client.get(self.relay_url)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LdaRelayError(
                f"relay returned HTTP {response.status_code}: {self.relay_url}"
            ) from exc
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise LdaRelayError("relay response is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise LdaRelayError("relay snapshot must be a JSON object")
        if payload.get("schema_version") != self.RELAY_SCHEMA_VERSION:
            raise LdaRelayError("unsupported relay snapshot schema")
        if payload.get("source_base_url") != self.DEFAULT_BASE_URL:
            raise LdaRelayError("relay snapshot does not identify the canonical LDA source")

        generated_at = _parse_relay_timestamp(payload.get("generated_at"), "generated_at")
        now = datetime.now(timezone.utc)
        if generated_at > now + timedelta(minutes=5):
            raise LdaRelayError("relay snapshot generated_at is unexpectedly in the future")
        age_hours = (now - generated_at).total_seconds() / 3600.0
        if age_hours > self.relay_max_age_hours:
            raise LdaRelayError(
                f"relay snapshot is stale ({age_hours:.1f}h old; max {self.relay_max_age_hours:.1f}h)"
            )

        try:
            coverage_after = date.fromisoformat(str(payload["posted_after"]))
            coverage_before = date.fromisoformat(str(payload["posted_before"]))
        except (KeyError, ValueError) as exc:
            raise LdaRelayError("relay snapshot has invalid date coverage") from exc
        if coverage_after > coverage_before:
            raise LdaRelayError("relay snapshot date coverage is inverted")
        filings = payload.get("filings")
        if not isinstance(filings, list) or not all(isinstance(item, dict) for item in filings):
            raise LdaRelayError("relay snapshot filings must be a list of objects")
        if payload.get("filing_count") != len(filings):
            raise LdaRelayError("relay snapshot filing_count does not match filings")

        self.relay_snapshot_generated_at = generated_at
        self._relay_snapshot = payload
        return payload

    def _relay_filings_response(self, snapshot: dict, *, params: dict) -> httpx.Response:
        try:
            page = int(params.get("page", 1))
            page_size = int(params.get("page_size", 25))
        except (TypeError, ValueError) as exc:
            raise LdaRelayError("relay pagination parameters must be integers") from exc
        if page <= 0 or not 1 <= page_size <= 25:
            raise LdaRelayError("relay pagination parameters are out of range")

        posted_after = _date_param(params.get("filing_dt_posted_after"))
        posted_before = _date_param(params.get("filing_dt_posted_before"))
        filing_year_raw = params.get("filing_year")
        filing_year = int(filing_year_raw) if filing_year_raw is not None else None

        coverage_after = date.fromisoformat(str(snapshot["posted_after"]))
        coverage_before = date.fromisoformat(str(snapshot["posted_before"]))
        if posted_after is None or posted_before is None:
            raise LdaRelayError(
                "rolling relay requires both filing_dt_posted_after and filing_dt_posted_before"
            )
        if posted_after < coverage_after or posted_before > coverage_before:
            raise LdaRelayError(
                "requested LDA date range is outside relay coverage "
                f"[{coverage_after.isoformat()}, {coverage_before.isoformat()}]"
            )

        filtered: list[dict] = []
        for filing in snapshot["filings"]:
            posted = _filing_posted_date(filing)
            if posted < posted_after or posted > posted_before:
                continue
            if filing_year is not None and _int_or_none(filing.get("filing_year")) != filing_year:
                continue
            filtered.append(filing)

        filtered.sort(key=lambda item: (str(item.get("dt_posted") or ""), str(item.get("filing_uuid") or "")))
        start = (page - 1) * page_size
        end = start + page_size
        results = filtered[start:end]
        has_next = end < len(filtered)
        has_previous = page > 1 and start < len(filtered)
        payload = {
            "count": len(filtered),
            "next": f"relay://filings?page={page + 1}" if has_next else None,
            "previous": f"relay://filings?page={page - 1}" if has_previous else None,
            "results": results,
        }
        return self._json_response(payload, path="/filings/")

    def _json_response(self, payload: dict, *, path: str) -> httpx.Response:
        request = httpx.Request("GET", f"https://relay.local{path}")
        return httpx.Response(200, request=request, json=payload)

    def _throttle(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._minimum_interval - (now - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = time.monotonic()


def _lda_api_error(
    response: httpx.Response,
    *,
    authenticated: bool,
    base_url: str,
) -> LdaApiError:
    status = response.status_code
    mode = "registered API key" if authenticated else "anonymous"
    retry_after = response.headers.get("retry-after")
    detail = _response_detail(response)
    edge_denied = _looks_like_edge_access_denied(response, detail=detail)

    if status == 403 and edge_denied:
        message = (
            "LDA edge/CDN returned 403 Access Denied before a normal API response. "
            "The registered API key may not have been evaluated. "
            f"Base URL: {base_url}. The client sends an identifiable User-Agent. "
            "Do not rotate the API key solely from this response; treat a repeated "
            "edge denial as an access/CDN condition. LDA_BASE_URL is available only "
            "for a replacement API endpoint published by the Senate."
        )
    elif status == 403 and not authenticated:
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
        edge_denied=edge_denied,
        base_url=base_url,
    )


def _looks_like_edge_access_denied(
    response: httpx.Response,
    *,
    detail: str,
) -> bool:
    if response.status_code != 403:
        return False
    content_type = response.headers.get("content-type", "").lower()
    normalized = detail.lower()
    return (
        "access denied" in normalized
        and ("text/html" in content_type or normalized.startswith("<html"))
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


def _parse_relay_timestamp(value, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise LdaRelayError(f"relay snapshot has invalid {field}") from exc
    if parsed.tzinfo is None:
        raise LdaRelayError(f"relay snapshot {field} must include timezone")
    return parsed.astimezone(timezone.utc)


def _date_param(value) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise LdaRelayError(f"invalid relay date parameter: {value}") from exc


def _filing_posted_date(filing: dict) -> date:
    value = str(filing.get("dt_posted") or "").strip()
    if not value:
        raise LdaRelayError("relay filing is missing dt_posted")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LdaRelayError(f"relay filing has invalid dt_posted: {value}") from exc
    return parsed.date()


def _int_or_none(value) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(value)
