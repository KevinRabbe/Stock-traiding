from datetime import datetime, timezone
from urllib.parse import quote

import httpx

from stock_trading.core import RawRecord, Source, content_sha256


class UsaSpendingClient:
    BASE_URL = "https://api.usaspending.gov/api/v2"

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=timeout_seconds,
            headers={"Accept": "application/json"},
            follow_redirects=True,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "UsaSpendingClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def fetch_award(self, award_id: str) -> RawRecord:
        normalized = award_id.strip()
        if not normalized:
            raise ValueError("award_id must not be empty")
        response = self._client.get(f"{self.BASE_URL}/awards/{quote(normalized, safe='_-')}/")
        response.raise_for_status()
        content = response.content
        return RawRecord(
            source=Source.USASPENDING,
            source_record_id=f"award:{normalized}",
            fetched_at=datetime.now(timezone.utc),
            content_type="application/json",
            content=content,
            sha256=content_sha256(content),
        )

    def fetch_transactions(
        self,
        award_id: str,
        *,
        page: int = 1,
        limit: int = 5000,
    ) -> RawRecord:
        normalized = award_id.strip()
        if not normalized:
            raise ValueError("award_id must not be empty")
        if page <= 0:
            raise ValueError("page must be > 0")
        if not 1 <= limit <= 5000:
            raise ValueError("USAspending transaction limit must be between 1 and 5000")

        response = self._client.post(
            f"{self.BASE_URL}/transactions/",
            json={
                "award_id": normalized,
                "page": page,
                "limit": limit,
                "sort": "action_date",
                "order": "asc",
            },
        )
        response.raise_for_status()
        content = response.content
        return RawRecord(
            source=Source.USASPENDING,
            source_record_id=f"transactions:{normalized}:page={page}:limit={limit}",
            fetched_at=datetime.now(timezone.utc),
            content_type="application/json",
            content=content,
            sha256=content_sha256(content),
        )
