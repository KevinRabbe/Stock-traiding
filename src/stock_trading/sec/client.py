import threading
import time
from datetime import datetime, timezone
from pathlib import PurePosixPath
from xml.etree import ElementTree

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
    def filing_document_url(cik: str, accession_number: str, document: str) -> str:
        normalized_cik = str(int(cik))
        accession_path = accession_number.replace("-", "")
        return (
            f"https://www.sec.gov/Archives/edgar/data/{normalized_cik}/"
            f"{accession_path}/{document}"
        )

    @staticmethod
    def filing_directory_index_url(cik: str, accession_number: str) -> str:
        normalized_cik = str(int(cik))
        accession_path = accession_number.replace("-", "")
        return (
            f"https://www.sec.gov/Archives/edgar/data/{normalized_cik}/"
            f"{accession_path}/index.json"
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
        primary_document: str | None = None,
    ) -> RawRecord:
        """Fetch the raw Form 4 ownership XML, not an EDGAR HTML rendering.

        The submissions API exposes a ``primaryDocument`` field, but ownership
        filing pages can expose both a rendered HTML representation and a raw XML
        representation. Trusting the filename and unconditionally labeling its
        bytes as XML can therefore quarantine valid filings. We verify the primary
        bytes first and, when necessary, discover XML candidates from the filing
        directory's SEC ``index.json`` and accept only a genuine ownershipDocument
        for Form 4/4-A.

        If no valid ownership XML can be discovered, return the originally fetched
        primary artifact (when one was supplied) with its real HTTP content type so
        the strict normalizer/quarantine boundary can fail closed with the raw bytes
        preserved. With no primary document to fall back to, raise ValueError.
        """

        primary_response: httpx.Response | None = None
        if primary_document:
            primary_response = self._get(
                self.filing_document_url(cik, accession_number, primary_document)
            )
            if self._is_form4_ownership_xml(primary_response.content):
                return self._filing_raw_record(
                    accession_number,
                    primary_response.content,
                    content_type="application/xml",
                )

        candidate_names = self._filing_xml_candidates(
            cik,
            accession_number,
            primary_document=primary_document,
        )
        for candidate in candidate_names:
            if primary_document and candidate == primary_document:
                continue
            response = self._get(
                self.filing_document_url(cik, accession_number, candidate)
            )
            if not self._is_form4_ownership_xml(response.content):
                continue
            return self._filing_raw_record(
                accession_number,
                response.content,
                content_type="application/xml",
            )

        if primary_response is not None:
            return self._filing_raw_record(
                accession_number,
                primary_response.content,
                content_type=self._response_content_type(primary_response),
            )
        raise ValueError(
            f"SEC filing has no discoverable Form 4 ownership XML: {accession_number}"
        )

    def _filing_xml_candidates(
        self,
        cik: str,
        accession_number: str,
        *,
        primary_document: str | None,
    ) -> tuple[str, ...]:
        response = self._get(self.filing_directory_index_url(cik, accession_number))
        payload = response.json()
        try:
            items = payload["directory"]["item"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"invalid SEC filing directory index for {accession_number}"
            ) from exc
        if not isinstance(items, list):
            raise ValueError(
                f"invalid SEC filing directory item list for {accession_number}"
            )

        primary_name = PurePosixPath(primary_document).name if primary_document else ""
        primary_stem = PurePosixPath(primary_name).stem.lower() if primary_name else ""
        candidates: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name or not name.lower().endswith(".xml"):
                continue
            if PurePosixPath(name).name.lower() == "index.xml":
                continue
            candidates.add(name)

        def rank(name: str) -> tuple[int, str]:
            basename = PurePosixPath(name).name
            stem = PurePosixPath(basename).stem.lower()
            lowered = basename.lower()
            if primary_stem and stem == primary_stem:
                priority = 0
            elif "form4" in lowered or "ownership" in lowered:
                priority = 1
            else:
                priority = 2
            return priority, lowered

        return tuple(sorted(candidates, key=rank))

    @staticmethod
    def _is_form4_ownership_xml(content: bytes) -> bool:
        try:
            root = ElementTree.fromstring(content)
        except ElementTree.ParseError:
            return False

        def local_name(tag: str) -> str:
            return tag.rsplit("}", 1)[-1]

        if local_name(str(root.tag)).lower() != "ownershipdocument":
            return False
        document_type: str | None = None
        for node in root.iter():
            if local_name(str(node.tag)).lower() == "documenttype":
                document_type = (node.text or "").strip().upper()
                break
        return document_type in {"4", "4/A"}

    @staticmethod
    def _response_content_type(response: httpx.Response) -> str:
        value = response.headers.get("content-type", "application/octet-stream")
        return value.split(";", 1)[0].strip().lower() or "application/octet-stream"

    @staticmethod
    def _filing_raw_record(
        accession_number: str,
        content: bytes,
        *,
        content_type: str,
    ) -> RawRecord:
        return RawRecord(
            source=Source.SEC_EDGAR,
            source_record_id=accession_number,
            fetched_at=datetime.now(timezone.utc),
            content_type=content_type,
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
