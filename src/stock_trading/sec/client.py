import threading
import time
from datetime import datetime, timezone

import httpx

from stock_trading.core import RawRecord, Source, content_sha256


class SecClient:
    """Small SEC HTTP client with declared identity and conservative throttling."""

    WWW_BASE = "https://www.sec.gov"
    DATA_BASE = "https://data.sec.gov"
    INSIDER_DATASET_BASE = (
        "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets"
    )
    INSIDER_DATASET_FALLBACK_BASE = (
        "https://www.sec.gov/files/datastandardsinnovation/data/"
        "insider-transactions-data-sets"
    )

    def __init__(
        self,
        user_agent: str,
        *,
        max_requests_per_second: float = 5.0,
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not user_agent.strip():
            raise ValueError("SEC user_agent must identify the application/contact")
        if not 0 < max_requests_per_second <= 10:
            raise ValueError("max_requests_per_second must be in (0, 10]")

        self._interval = 1.0 / max_requests_per_second
        self._lock = threading.Lock()
        self._last_request_at = 0.0
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=timeout_seconds,
            headers={
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
            follow_redirects=True,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "SecClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @classmethod
    def quarterly_archive_url(cls, year: int, quarter: int) -> str:
        return cls.quarterly_archive_urls(year, quarter)[0]

    @classmethod
    def quarterly_archive_urls(cls, year: int, quarter: int) -> tuple[str, ...]:
        if year < 2006:
            raise ValueError("SEC insider quarterly data starts in 2006")
        if quarter not in {1, 2, 3, 4}:
            raise ValueError("quarter must be 1..4")
        filename = f"{year}q{quarter}_form345.zip"
        return (
            f"{cls.INSIDER_DATASET_BASE}/{filename}",
            f"{cls.INSIDER_DATASET_FALLBACK_BASE}/{filename}",
        )

    @staticmethod
    def submissions_url(cik: str) -> str:
        normalized = cik.strip().lstrip("0").zfill(10)
        return f"https://data.sec.gov/submissions/CIK{normalized}.json"

    @staticmethod
    def filing_document_url(cik: str, accession_number: str, primary_document: str) -> str:
        normalized_cik = str(int(cik))
        accession_path = accession_number.replace("-", "")
        return (
            f"https://www.sec.gov/Archives/edgar/data/{normalized_cik}/"
            f"{accession_path}/{primary_document}"
        )

    def fetch_quarterly_archive(self, year: int, quarter: int) -> RawRecord:
        last_not_found: httpx.HTTPStatusError | None = None
        response: httpx.Response | None = None
        for url in self.quarterly_archive_urls(year, quarter):
            try:
                response = self._get(url)
                break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 404:
                    raise
                last_not_found = exc

        if response is None:
            if last_not_found is not None:
                raise last_not_found
            raise RuntimeError(f"SEC quarterly archive lookup failed for {year}Q{quarter}")

        content = response.content
        return RawRecord(
            source=Source.SEC_QUARTERLY,
            source_record_id=f"{year}Q{quarter}",
            fetched_at=datetime.now(timezone.utc),
            content_type="application/zip",
            content=content,
            sha256=content_sha256(content),
        )

    def fetch_submissions_raw(self, cik: str) -> RawRecord:
        """Fetch one mutable SEC submissions document as an immutable raw snapshot."""

        normalized = cik.strip().lstrip("0").zfill(10)
        content = self._get(self.submissions_url(normalized)).content
        return RawRecord(
            source=Source.SEC_EDGAR,
            source_record_id=f"submissions:CIK{normalized}",
            fetched_at=datetime.now(timezone.utc),
            content_type="application/json",
            content=content,
            sha256=content_sha256(content),
        )

    def fetch_submissions(self, cik: str) -> dict:
        return self._get(self.submissions_url(cik)).json()

    def fetch_filing_xml(
        self,
        cik: str,
        accession_number: str,
        primary_document: str,
    ) -> RawRecord:
        url = self.filing_document_url(cik, accession_number, primary_document)
        content = self._get(url).content
        return RawRecord(
            source=Source.SEC_EDGAR,
            source_record_id=accession_number,
            fetched_at=datetime.now(timezone.utc),
            content_type="application/xml",
            content=content,
            sha256=content_sha256(content),
        )

    def _get(self, url: str) -> httpx.Response:
        self._throttle()
        response = self._client.get(url)
        response.raise_for_status()
        return response

    def _throttle(self) -> None:
        with self._lock:
            now = time.monotonic()
            remaining = self._interval - (now - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)
            self._last_request_at = time.monotonic()