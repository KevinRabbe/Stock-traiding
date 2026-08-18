from dataclasses import dataclass
from datetime import datetime, time, timezone
from decimal import Decimal, InvalidOperation
from xml.etree import ElementTree
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
class Form4IssuerIdentity:
    cik: str
    name: str
    ticker: str


class Form4XmlParser:
    """Normalize one live EDGAR Form 4 ownership XML document."""

    def issuer_identity(self, raw: RawRecord) -> Form4IssuerIdentity:
        """Read issuer identity directly from a verified Form 4 ownership document."""

        if raw.source is not Source.SEC_EDGAR:
            raise ValueError("Form 4 XML RawRecord must use Source.SEC_EDGAR")
        xml_content = (
            raw.content
            if isinstance(raw.content, bytes)
            else raw.content.encode("utf-8")
        )
        root = ElementTree.fromstring(xml_content)
        document_type = (self._text(root, "documentType") or "").upper()
        if document_type not in {"4", "4/A"}:
            raise ValueError("ownership document is not Form 4/4-A")

        cik = self._text(root, "issuer/issuerCik")
        name = self._text(root, "issuer/issuerName")
        ticker = self._text(root, "issuer/issuerTradingSymbol")
        if not cik:
            raise ValueError("Form 4 XML is missing issuer CIK")
        if not name:
            raise ValueError("Form 4 XML is missing issuer name")
        if not ticker:
            raise ValueError("Form 4 XML is missing issuer trading symbol")
        return Form4IssuerIdentity(
            cik=cik.zfill(10),
            name=name,
            ticker=ticker,
        )

    def to_events(
        self,
        raw: RawRecord,
        *,
        accepted_at: datetime,
        ingested_at: datetime,
    ) -> tuple[Event, ...]:
        if raw.source is not Source.SEC_EDGAR:
            raise ValueError("Form 4 XML RawRecord must use Source.SEC_EDGAR")

        accepted_at = as_utc(accepted_at)
        ingested_at = as_utc(ingested_at)
        xml_content = raw.content if isinstance(raw.content, bytes) else raw.content.encode("utf-8")
        root = ElementTree.fromstring(xml_content)

        document_type = (self._text(root, "documentType") or "").upper()
        if document_type not in {"4", "4/A"}:
            return ()

        issuer_cik = self._text(root, "issuer/issuerCik")
        if not issuer_cik:
            raise ValueError("Form 4 XML is missing issuer CIK")
        issuer_cik = issuer_cik.zfill(10)

        owners = self._reporting_owners(root)
        sole_owner = owners[0] if len(owners) == 1 else None
        filing_has_10b5_1 = self._bool(self._text(root, "aff10b5One"))

        transactions = root.findall("./nonDerivativeTable/nonDerivativeTransaction")
        events: list[Event] = []
        for index, transaction in enumerate(transactions):
            transaction_date_text = self._value(transaction, "transactionDate")
            if not transaction_date_text:
                continue
            transaction_date = datetime.strptime(transaction_date_text, "%Y-%m-%d").date()
            event_time = datetime.combine(
                transaction_date,
                time.min,
                tzinfo=_EASTERN,
            ).astimezone(timezone.utc)

            transaction_code = self._text(transaction, "transactionCoding/transactionCode")
            acquired_disposed = self._value(
                transaction,
                "transactionAmounts/transactionAcquiredDisposedCode",
            )
            shares = self._decimal(self._value(transaction, "transactionAmounts/transactionShares"))
            price = self._decimal(
                self._value(transaction, "transactionAmounts/transactionPricePerShare")
            )
            shares_after = self._decimal(
                self._value(transaction, "postTransactionAmounts/sharesOwnedFollowingTransaction")
            )
            direct_indirect = self._value(
                transaction,
                "ownershipNature/directOrIndirectOwnership",
            )
            nature = self._value(transaction, "ownershipNature/natureOfOwnership")
            ownership = direct_indirect
            if nature:
                ownership = f"{ownership}:{nature}" if ownership else nature

            role = None
            actor_id = None
            if sole_owner is not None:
                actor_id = f"sec_owner_cik_{sole_owner['cik']}"
                role_parts = [
                    relationship
                    for relationship in sole_owner["relationships"]
                    if relationship
                ]
                if sole_owner["officer_title"]:
                    role_parts.append(sole_owner["officer_title"])
                role = ":".join(role_parts) or None
            elif owners:
                role = "MULTIPLE_REPORTING_OWNERS"

            source_record_id = f"{raw.source_record_id}:NONDERIV_TRANS:{index}"
            value = shares * price if shares is not None and price is not None else None

            events.append(
                Event(
                    event_id=deterministic_event_id(
                        Source.SEC_EDGAR,
                        source_record_id,
                        EventType.INSIDER_TRANSACTION,
                    ),
                    event_type=EventType.INSIDER_TRANSACTION,
                    company_id=company_id_from_sec_cik(issuer_cik),
                    actor_id=actor_id,
                    event_time=event_time,
                    public_time=accepted_at,
                    first_tradable_time=None,
                    source=Source.SEC_EDGAR,
                    source_record_id=source_record_id,
                    payload=InsiderTransactionPayload(
                        source_transaction_code=transaction_code or "UNKNOWN",
                        direction=classify_direction(transaction_code, acquired_disposed),
                        shares=shares,
                        price=price,
                        value=value,
                        insider_role=role,
                        ownership_type=ownership,
                        shares_owned_after=shares_after,
                        intent_class=classify_intent(transaction_code, acquired_disposed),
                        is_10b5_1=filing_has_10b5_1,
                    ),
                    semantic=None,
                    raw_artifact_id=raw.artifact_id,
                    ingested_at=ingested_at,
                )
            )

        return tuple(events)

    @classmethod
    def _reporting_owners(cls, root: ElementTree.Element) -> tuple[dict[str, object], ...]:
        owners: list[dict[str, object]] = []
        for owner in root.findall("./reportingOwner"):
            cik = cls._text(owner, "reportingOwnerId/rptOwnerCik")
            if not cik:
                continue
            relationship = owner.find("reportingOwnerRelationship")
            relationships: list[str] = []
            if relationship is not None:
                flags = (
                    ("isDirector", "DIRECTOR"),
                    ("isOfficer", "OFFICER"),
                    ("isTenPercentOwner", "TENPERCENTOWNER"),
                    ("isOther", "OTHER"),
                )
                for tag, name in flags:
                    if cls._bool(cls._text(relationship, tag)):
                        relationships.append(name)
                officer_title = cls._text(relationship, "officerTitle")
            else:
                officer_title = None

            owners.append(
                {
                    "cik": cik.zfill(10),
                    "name": cls._text(owner, "reportingOwnerId/rptOwnerName"),
                    "relationships": tuple(relationships),
                    "officer_title": officer_title,
                }
            )
        return tuple(owners)

    @staticmethod
    def _text(root: ElementTree.Element, path: str) -> str | None:
        node = root.find(path)
        if node is None or node.text is None:
            return None
        value = node.text.strip()
        return value or None

    @classmethod
    def _value(cls, root: ElementTree.Element, path: str) -> str | None:
        return cls._text(root, f"{path}/value")

    @staticmethod
    def _bool(value: str | None) -> bool | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if normalized in {"1", "Y", "YES", "TRUE"}:
            return True
        if normalized in {"0", "N", "NO", "FALSE"}:
            return False
        return None

    @staticmethod
    def _decimal(value: str | None) -> Decimal | None:
        if value is None or not value.strip():
            return None
        try:
            return Decimal(value.replace(",", "").replace("$", "").strip())
        except InvalidOperation as exc:
            raise ValueError(f"invalid Form 4 numeric value: {value}") from exc
