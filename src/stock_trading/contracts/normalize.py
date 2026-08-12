import json
from dataclasses import dataclass
from datetime import datetime, time, timezone
from decimal import Decimal

from stock_trading.core import (
    Event,
    EventType,
    GovernmentContractPayload,
    RawRecord,
    Source,
    as_utc,
    deterministic_event_id,
)


@dataclass(frozen=True, slots=True)
class AwardContext:
    award_id: str
    recipient_name: str | None
    recipient_uei: str | None
    parent_recipient_name: str | None
    parent_recipient_uei: str | None
    awarding_agency: str | None
    awarding_subagency: str | None
    total_obligation: Decimal | None
    potential_award_amount: Decimal | None
    award_type: str | None
    award_description: str | None
    naics_code: str | None
    psc_code: str | None


class UsaSpendingNormalizer:
    """Normalize USAspending award context plus transaction history.

    Historical action dates are not treated as historical publication times.
    `observed_at` is the safe information boundary until a separate publication
    reconstruction process provides stronger evidence.
    """

    def parse_award(self, raw: RawRecord) -> AwardContext:
        self._require_source(raw)
        payload = json.loads(self._text(raw))
        if not isinstance(payload, dict):
            raise ValueError("USAspending award response must be an object")

        recipient = payload.get("recipient") or {}
        agency = payload.get("awarding_agency") or {}
        toptier = agency.get("toptier_agency") or {} if isinstance(agency, dict) else {}
        subtier = agency.get("subtier_agency") or {} if isinstance(agency, dict) else {}
        contract_data = payload.get("latest_transaction_contract_data") or {}

        award_id = str(payload.get("generated_unique_award_id") or "").strip()
        if not award_id:
            raise ValueError("USAspending award response has no generated_unique_award_id")

        return AwardContext(
            award_id=award_id,
            recipient_name=self._text_value(recipient, "recipient_name"),
            recipient_uei=self._text_value(recipient, "recipient_uei"),
            parent_recipient_name=self._text_value(recipient, "parent_recipient_name"),
            parent_recipient_uei=self._text_value(recipient, "parent_recipient_uei"),
            awarding_agency=self._text_value(toptier, "name"),
            awarding_subagency=self._text_value(subtier, "name"),
            total_obligation=self._decimal_or_none(payload.get("total_obligation")),
            potential_award_amount=self._decimal_or_none(payload.get("base_and_all_options")),
            award_type=str(payload.get("type_description") or payload.get("type") or "").strip()
            or None,
            award_description=str(payload.get("description") or "").strip() or None,
            naics_code=self._text_value(contract_data, "naics"),
            psc_code=self._text_value(contract_data, "product_or_service_code"),
        )

    def to_events(
        self,
        transactions_raw: RawRecord,
        *,
        award: AwardContext,
        observed_at: datetime,
        company_ids_by_uei: dict[str, str] | None = None,
    ) -> tuple[Event, ...]:
        self._require_source(transactions_raw)
        public_time = as_utc(observed_at)
        payload = json.loads(self._text(transactions_raw))
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            raise ValueError("USAspending transactions response has no results list")

        company_map = {key.strip(): value for key, value in (company_ids_by_uei or {}).items()}
        company_id = None
        matched_uei = None
        for uei in (award.recipient_uei, award.parent_recipient_uei):
            if uei and uei in company_map:
                company_id = company_map[uei]
                matched_uei = uei
                break
        actor_uei = matched_uei or award.recipient_uei or award.parent_recipient_uei

        events: list[Event] = []
        for row in results:
            if not isinstance(row, dict):
                continue
            transaction_id = str(row.get("id") or "").strip()
            action_date_text = str(row.get("action_date") or "").strip()
            if not transaction_id or not action_date_text:
                continue

            action_date = datetime.strptime(action_date_text[:10], "%Y-%m-%d").date()
            event_time = datetime.combine(action_date, time.min, tzinfo=timezone.utc)
            source_record_id = f"{award.award_id}:{transaction_id}"
            transaction_description = str(row.get("description") or "").strip() or None
            action_type = str(
                row.get("action_type_description") or row.get("action_type") or ""
            ).strip() or None

            events.append(
                Event(
                    event_id=deterministic_event_id(
                        Source.USASPENDING,
                        source_record_id,
                        EventType.GOVERNMENT_CONTRACT,
                    ),
                    event_type=EventType.GOVERNMENT_CONTRACT,
                    company_id=company_id,
                    actor_id=(f"usaspending_uei_{actor_uei}" if actor_uei else None),
                    event_time=event_time,
                    public_time=public_time,
                    first_tradable_time=None,
                    source=Source.USASPENDING,
                    source_record_id=source_record_id,
                    payload=GovernmentContractPayload(
                        award_id=award.award_id,
                        transaction_id=transaction_id,
                        agency=award.awarding_agency,
                        subagency=award.awarding_subagency,
                        obligation_amount=self._decimal_or_none(row.get("federal_action_obligation")),
                        total_obligation=award.total_obligation,
                        potential_award_amount=award.potential_award_amount,
                        award_type=str(row.get("type_description") or award.award_type or "").strip()
                        or None,
                        action_type=action_type,
                        modification_number=str(row.get("modification_number") or "").strip()
                        or None,
                        naics_code=award.naics_code,
                        psc_code=award.psc_code,
                        description=transaction_description or award.award_description,
                        recipient_uei=award.recipient_uei,
                        recipient_name=award.recipient_name,
                    ),
                    semantic=None,
                    raw_artifact_id=transactions_raw.artifact_id,
                    ingested_at=max(transactions_raw.fetched_at, public_time),
                )
            )

        return tuple(events)

    @staticmethod
    def semantic_text(event: Event) -> str:
        if event.event_type is not EventType.GOVERNMENT_CONTRACT:
            raise ValueError("semantic_text requires a government contract event")
        payload = event.payload
        return "\n".join(
            [
                f"Recipient: {payload.recipient_name or 'unknown'}",
                f"Awarding agency: {payload.agency or 'unknown'}",
                f"Awarding subagency: {payload.subagency or 'unknown'}",
                f"Award type: {payload.award_type or 'unknown'}",
                f"Action type: {payload.action_type or 'unknown'}",
                f"NAICS: {payload.naics_code or 'unknown'}",
                f"PSC: {payload.psc_code or 'unknown'}",
                f"Description: {payload.description or 'none'}",
            ]
        )

    @staticmethod
    def _require_source(raw: RawRecord) -> None:
        if raw.source is not Source.USASPENDING:
            raise ValueError("USAspending normalizer requires Source.USASPENDING")

    @staticmethod
    def _text(raw: RawRecord) -> str:
        return raw.content.decode("utf-8") if isinstance(raw.content, bytes) else raw.content

    @staticmethod
    def _text_value(container, key: str) -> str | None:
        if not isinstance(container, dict):
            return None
        value = str(container.get(key) or "").strip()
        return value or None

    @staticmethod
    def _decimal_or_none(value) -> Decimal | None:
        if value is None or str(value).strip() == "":
            return None
        return Decimal(str(value))
