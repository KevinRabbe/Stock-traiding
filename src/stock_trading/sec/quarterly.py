import csv
import io
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath
from zoneinfo import ZoneInfo

from stock_trading.core import (
    Event,
    EventType,
    InsiderTransactionPayload,
    RawRecord,
    Source,
    as_utc,
    deterministic_event_id,
)
from stock_trading.entities import company_id_from_sec_cik

from .codes import classify_direction, classify_intent


_EASTERN = ZoneInfo("America/New_York")


@dataclass(frozen=True, slots=True)
class ReportingOwner:
    cik: str
    name: str
    relationship: str | None
    title: str | None


@dataclass(frozen=True, slots=True)
class QuarterlyTransaction:
    accession_number: str
    transaction_key: str
    document_type: str
    filing_date: date
    issuer_cik: str
    issuer_name: str
    issuer_symbol: str | None
    transaction_date: date
    security_title: str
    transaction_code: str | None
    acquired_disposed: str
    shares: Decimal | None
    price_per_share: Decimal | None
    shares_owned_after: Decimal | None
    direct_indirect_ownership: str | None
    nature_of_ownership: str | None
    filing_has_10b5_1: bool | None
    reporting_owners: tuple[ReportingOwner, ...]


class QuarterlyArchiveParser:
    """Parse the SEC's quarterly flattened Forms 3/4/5 ZIP archive.

    V1 deliberately emits only non-derivative Form 4 / 4-A transactions.
    Derivative transactions remain available for a later extension without
    changing the canonical event contract.
    """

    def parse(self, archive_bytes: bytes) -> tuple[QuarterlyTransaction, ...]:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            submissions = self._read_table(archive, "SUBMISSION")
            owners = self._read_table(archive, "REPORTINGOWNER")
            transactions = self._read_table(archive, "NONDERIV_TRANS")

        submission_by_accession = {
            self._clean(row.get("ACCESSION_NUMBER")): row
            for row in submissions
            if self._clean(row.get("ACCESSION_NUMBER"))
        }

        owners_by_accession: dict[str, list[ReportingOwner]] = {}
        for row in owners:
            accession = self._clean(row.get("ACCESSION_NUMBER"))
            cik = self._clean(row.get("RPTOWNERCIK"))
            if not accession or not cik:
                continue
            owners_by_accession.setdefault(accession, []).append(
                ReportingOwner(
                    cik=cik.zfill(10),
                    name=self._clean(row.get("RPTOWNERNAME")) or "UNKNOWN",
                    relationship=self._clean(row.get("RPTOWNER_RELATIONSHIP")),
                    title=self._clean(row.get("RPTOWNER_TITLE")),
                )
            )

        parsed: list[QuarterlyTransaction] = []
        for row in transactions:
            accession = self._clean(row.get("ACCESSION_NUMBER"))
            submission = submission_by_accession.get(accession)
            if not accession or submission is None:
                continue

            document_type = (self._clean(submission.get("DOCUMENT_TYPE")) or "").upper()
            if document_type not in {"4", "4/A"}:
                continue

            transaction_date = self._parse_date(row.get("TRANS_DATE"))
            filing_date = self._parse_date(submission.get("FILING_DATE"))
            transaction_key = self._clean(row.get("NONDERIV_TRANS_SK"))
            issuer_cik = self._clean(submission.get("ISSUERCIK"))
            security_title = self._clean(row.get("SECURITY_TITLE"))
            acquired_disposed = self._clean(row.get("TRANS_ACQUIRED_DISP_CD"))
            if not all(
                (transaction_date, filing_date, transaction_key, issuer_cik, security_title, acquired_disposed)
            ):
                continue

            parsed.append(
                QuarterlyTransaction(
                    accession_number=accession,
                    transaction_key=transaction_key,
                    document_type=document_type,
                    filing_date=filing_date,
                    issuer_cik=issuer_cik.zfill(10),
                    issuer_name=self._clean(submission.get("ISSUERNAME")) or "UNKNOWN",
                    issuer_symbol=self._clean(submission.get("ISSUERTRADINGSYMBOL")),
                    transaction_date=transaction_date,
                    security_title=security_title,
                    transaction_code=self._clean(row.get("TRANS_CODE")),
                    acquired_disposed=acquired_disposed,
                    shares=self._parse_decimal(row.get("TRANS_SHARES")),
                    price_per_share=self._parse_decimal(row.get("TRANS_PRICEPERSHARE")),
                    shares_owned_after=self._parse_decimal(row.get("SHRS_OWND_FOLWNG_TRANS")),
                    direct_indirect_ownership=self._clean(row.get("DIRECT_INDIRECT_OWNERSHIP")),
                    nature_of_ownership=self._clean(row.get("NATURE_OF_OWNERSHIP")),
                    filing_has_10b5_1=self._parse_bool(submission.get("AFF10B5ONE")),
                    reporting_owners=tuple(owners_by_accession.get(accession, ())),
                )
            )

        return tuple(parsed)

    @staticmethod
    def has_temporal_anomaly(transaction: QuarterlyTransaction) -> bool:
        """Return True when an as-filed row implies disclosure before the transaction."""
        return transaction.transaction_date > transaction.filing_date

    def to_events(
        self,
        archive: RawRecord,
        *,
        ingested_at: datetime,
        transactions: tuple[QuarterlyTransaction, ...] | None = None,
    ) -> tuple[Event, ...]:
        if archive.source is not Source.SEC_QUARTERLY:
            raise ValueError("quarterly archive RawRecord must use Source.SEC_QUARTERLY")

        ingested_at = as_utc(ingested_at)
        events: list[Event] = []
        parsed_transactions = transactions
        if parsed_transactions is None:
            parsed_transactions = self.parse(
                archive.content if isinstance(archive.content, bytes) else archive.content.encode("utf-8")
            )

        for transaction in parsed_transactions:
            # Preserve the source row in the immutable raw archive, but never
            # manufacture a canonical point-in-time event from impossible
            # chronology. In particular, do not clamp or rewrite either date.
            if self.has_temporal_anomaly(transaction):
                continue

            source_record_id = (
                f"{transaction.accession_number}:NONDERIV_TRANS:{transaction.transaction_key}"
            )
            event_id = deterministic_event_id(
                Source.SEC_QUARTERLY,
                source_record_id,
                EventType.INSIDER_TRANSACTION,
            )

            event_time = datetime.combine(
                transaction.transaction_date,
                time.min,
                tzinfo=_EASTERN,
            ).astimezone(timezone.utc)
            public_time = conservative_historical_public_time(transaction.filing_date)

            owner = transaction.reporting_owners[0] if len(transaction.reporting_owners) == 1 else None
            role = None
            if owner is not None:
                role = owner.relationship
                if owner.title:
                    role = f"{role}:{owner.title}" if role else owner.title
            elif transaction.reporting_owners:
                role = "MULTIPLE_REPORTING_OWNERS"

            ownership = transaction.direct_indirect_ownership
            if transaction.nature_of_ownership:
                ownership = (
                    f"{ownership}:{transaction.nature_of_ownership}"
                    if ownership
                    else transaction.nature_of_ownership
                )

            value = None
            if transaction.shares is not None and transaction.price_per_share is not None:
                value = transaction.shares * transaction.price_per_share

            events.append(
                Event(
                    event_id=event_id,
                    event_type=EventType.INSIDER_TRANSACTION,
                    company_id=company_id_from_sec_cik(transaction.issuer_cik),
                    actor_id=(f"sec_owner_cik_{owner.cik}" if owner else None),
                    event_time=event_time,
                    public_time=public_time,
                    first_tradable_time=None,
                    source=Source.SEC_QUARTERLY,
                    source_record_id=source_record_id,
                    payload=InsiderTransactionPayload(
                        source_transaction_code=transaction.transaction_code or "UNKNOWN",
                        direction=classify_direction(
                            transaction.transaction_code,
                            transaction.acquired_disposed,
                        ),
                        shares=transaction.shares,
                        price=transaction.price_per_share,
                        value=value,
                        insider_role=role,
                        ownership_type=ownership,
                        shares_owned_after=transaction.shares_owned_after,
                        intent_class=classify_intent(
                            transaction.transaction_code,
                            transaction.acquired_disposed,
                        ),
                        is_10b5_1=transaction.filing_has_10b5_1,
                    ),
                    semantic=None,
                    raw_artifact_id=archive.artifact_id,
                    ingested_at=ingested_at,
                )
            )

        return tuple(events)

    @staticmethod
    def _read_table(archive: zipfile.ZipFile, table_name: str) -> list[dict[str, str]]:
        target = table_name.upper()
        candidates = []
        for name in archive.namelist():
            stem = PurePosixPath(name).stem.upper().replace("-", "_")
            if stem == target or stem.startswith(f"{target}_"):
                candidates.append(name)
        if not candidates:
            raise ValueError(f"missing {table_name} table in SEC quarterly archive")

        with archive.open(sorted(candidates)[0]) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            return list(csv.DictReader(text, delimiter="\t"))

    @staticmethod
    def _clean(value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @classmethod
    def _parse_date(cls, value: str | None) -> date | None:
        cleaned = cls._clean(value)
        if not cleaned:
            return None
        for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(cleaned, fmt).date()
            except ValueError:
                continue
        raise ValueError(f"unsupported SEC date format: {cleaned}")

    @classmethod
    def _parse_decimal(cls, value: str | None) -> Decimal | None:
        cleaned = cls._clean(value)
        if not cleaned:
            return None
        cleaned = cleaned.replace(",", "").replace("$", "")
        try:
            return Decimal(cleaned)
        except InvalidOperation as exc:
            raise ValueError(f"invalid SEC numeric value: {value}") from exc

    @classmethod
    def _parse_bool(cls, value: str | None) -> bool | None:
        cleaned = cls._clean(value)
        if cleaned is None:
            return None
        normalized = cleaned.upper()
        if normalized in {"1", "Y", "YES", "TRUE"}:
            return True
        if normalized in {"0", "N", "NO", "FALSE"}:
            return False
        return None


def conservative_historical_public_time(filing_date: date) -> datetime:
    """Conservative point-in-time timestamp for quarterly rows.

    The flattened quarterly files preserve filing date but not full acceptance
    metadata. Treat the information as available only at the end of that filing
    date. Daily backtests therefore cannot execute from it until a later market
    session, avoiding same-day look-ahead.
    """

    local = datetime.combine(filing_date, time.max, tzinfo=_EASTERN)
    return local.astimezone(timezone.utc)
