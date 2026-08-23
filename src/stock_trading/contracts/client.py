from datetime import date, datetime, timezone
from urllib.parse import quote

import httpx

from stock_trading.core import RawRecord, Source, content_sha256


CONTRACT_TYPE_CODES = ("A", "B", "C", "D")
CONTRACT_AWARD_SEARCH_FIELDS = (
    "Award ID",
    "Recipient Name",
    "Recipient UEI",
    "Awarding Agency",
    "Awarding Sub Agency",
    "Description",
    "Last Modified Date",
    "Award Amount",
    "Contract Award Type",
    "NAICS",
    "PSC",
    "generated_internal_id",
)
CONTRACT_TRANSACTION_SEARCH_FIELDS = (
    "Award ID",
    "Mod",
    "Recipient Name",
    "Recipient UEI",
    "Action Date",
    "Action Type",
    "Transaction Amount",
    "Transaction Description",
    "Awarding Agency",
    "Awarding Sub Agency",
    "Award Type",
    "NAICS",
    "PSC",
    "generated_internal_id",
)


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

    def search_contract_awards_page(
        self,
        *,
        modified_after: date,
        modified_before: date,
        page: int = 1,
        limit: int = 100,
    ) -> RawRecord:
        """Discover contract awards changed in a bounded source-modification window."""

        self._validate_search_window(modified_after, modified_before, page=page, limit=limit)
        response = self._client.post(
            f"{self.BASE_URL}/search/spending_by_award/",
            json={
                "spending_level": "awards",
                "filters": {
                    "award_type_codes": list(CONTRACT_TYPE_CODES),
                    "time_period": [
                        {
                            "start_date": modified_after.isoformat(),
                            "end_date": modified_before.isoformat(),
                            "date_type": "last_modified_date",
                        }
                    ],
                },
                "fields": list(CONTRACT_AWARD_SEARCH_FIELDS),
                "page": page,
                "limit": limit,
                "sort": "Last Modified Date",
                "order": "asc",
            },
        )
        response.raise_for_status()
        return self._raw_record(
            f"contract-awards:last-modified={modified_after.isoformat()}..{modified_before.isoformat()}:page={page}:limit={limit}",
            response,
        )

    def search_contract_transactions_page(
        self,
        award_search_id: str,
        *,
        modified_after: date,
        modified_before: date,
        page: int = 1,
        limit: int = 100,
    ) -> RawRecord:
        """Discover changed transactions for one award display ID from search results."""

        normalized = award_search_id.strip()
        if not normalized:
            raise ValueError("award_search_id must not be empty")
        self._validate_search_window(modified_after, modified_before, page=page, limit=limit)
        response = self._client.post(
            f"{self.BASE_URL}/search/spending_by_transaction/",
            json={
                "filters": {
                    "award_type_codes": list(CONTRACT_TYPE_CODES),
                    "award_ids": [normalized],
                    "time_period": [
                        {
                            "start_date": modified_after.isoformat(),
                            "end_date": modified_before.isoformat(),
                            "date_type": "last_modified_date",
                        }
                    ],
                },
                "fields": list(CONTRACT_TRANSACTION_SEARCH_FIELDS),
                "page": page,
                "limit": limit,
                "sort": "Action Date",
                "order": "asc",
            },
        )
        response.raise_for_status()
        return self._raw_record(
            f"contract-transactions:{normalized}:last-modified={modified_after.isoformat()}..{modified_before.isoformat()}:page={page}:limit={limit}",
            response,
        )

    def fetch_award(self, award_id: str) -> RawRecord:
        normalized = award_id.strip()
        if not normalized:
            raise ValueError("award_id must not be empty")
        response = self._client.get(f"{self.BASE_URL}/awards/{quote(normalized, safe='_-')}/")
        response.raise_for_status()
        return self._raw_record(f"award:{normalized}", response)

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
        return self._raw_record(
            f"transactions:{normalized}:page={page}:limit={limit}",
            response,
        )

    @staticmethod
    def _validate_search_window(
        modified_after: date,
        modified_before: date,
        *,
        page: int,
        limit: int,
    ) -> None:
        if modified_after > modified_before:
            raise ValueError("USAspending search start date must not exceed end date")
        if page <= 0:
            raise ValueError("page must be > 0")
        if not 1 <= limit <= 100:
            raise ValueError("USAspending search limit must be between 1 and 100")

    @staticmethod
    def _raw_record(record_id: str, response: httpx.Response) -> RawRecord:
        content = response.content
        return RawRecord(
            source=Source.USASPENDING,
            source_record_id=record_id,
            fetched_at=datetime.now(timezone.utc),
            content_type="application/json",
            content=content,
            sha256=content_sha256(content),
        )
