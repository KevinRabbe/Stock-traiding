import json
from datetime import datetime
from decimal import Decimal, InvalidOperation

from stock_trading.core import (
    Event,
    EventType,
    LobbyingActivityPayload,
    RawRecord,
    Source,
    as_utc,
    deterministic_event_id,
)


class LdaFilingNormalizer:
    """Normalize LDA LD-1/LD-2 filings using dt_posted as the information boundary."""

    def to_events(
        self,
        raw: RawRecord,
        *,
        company_ids_by_client_id: dict[int, str] | None = None,
    ) -> tuple[Event, ...]:
        if raw.source is not Source.LDA:
            raise ValueError("LDA normalizer requires Source.LDA")
        payload = json.loads(self._text(raw))
        filings = payload.get("results") if isinstance(payload, dict) and "results" in payload else [payload]
        if not isinstance(filings, list):
            raise ValueError("LDA filing response must contain a results list or one filing object")

        company_map = company_ids_by_client_id or {}
        events: list[Event] = []
        for filing in filings:
            if not isinstance(filing, dict):
                continue
            filing_uuid = str(filing.get("filing_uuid") or "").strip()
            posted_text = str(filing.get("dt_posted") or "").strip()
            client = filing.get("client") or {}
            if not filing_uuid or not posted_text or not isinstance(client, dict):
                continue

            client_name = str(client.get("name") or "").strip()
            if not client_name:
                continue
            client_id = self._int_or_none(client.get("id"))
            public_time = self._parse_timestamp(posted_text)
            registrant = filing.get("registrant") or {}
            registrant_name = (
                str(registrant.get("name") or "").strip() if isinstance(registrant, dict) else ""
            )

            issue_codes: list[str] = []
            government_entities: list[str] = []
            specific_issues: list[str] = []
            activities = filing.get("lobbying_activities") or []
            if isinstance(activities, list):
                for activity in activities:
                    if not isinstance(activity, dict):
                        continue
                    code = str(activity.get("general_issue_code") or "").strip()
                    if code:
                        issue_codes.append(code)
                    description = str(activity.get("description") or "").strip()
                    if description:
                        specific_issues.append(description)
                    self._collect_entities(activity.get("government_entities"), government_entities)

            # Legacy filings may only expose filing-level government entities.
            self._collect_entities(filing.get("government_entities"), government_entities)

            filing_year = self._int_or_none(filing.get("filing_year"))
            filing_period = str(filing.get("filing_period") or "").strip() or None
            amount = self._reported_amount(filing)
            source_record_id = filing_uuid

            events.append(
                Event(
                    event_id=deterministic_event_id(
                        Source.LDA,
                        source_record_id,
                        EventType.LOBBYING_ACTIVITY,
                    ),
                    event_type=EventType.LOBBYING_ACTIVITY,
                    company_id=company_map.get(client_id) if client_id is not None else None,
                    actor_id=(f"lda_client_{client_id}" if client_id is not None else None),
                    # The filing covers prior lobbying activity, but no exact activity timestamp exists.
                    # Using dt_posted for both fields avoids inventing earlier point-in-time knowledge.
                    event_time=public_time,
                    public_time=public_time,
                    first_tradable_time=None,
                    source=Source.LDA,
                    source_record_id=source_record_id,
                    payload=LobbyingActivityPayload(
                        filing_id=filing_uuid,
                        client_name=client_name,
                        registrant_name=registrant_name or None,
                        filing_year=filing_year,
                        filing_period=filing_period,
                        amount=amount,
                        issue_codes=tuple(dict.fromkeys(issue_codes)),
                        government_entities=tuple(dict.fromkeys(government_entities)),
                        specific_issues=tuple(dict.fromkeys(specific_issues)),
                    ),
                    semantic=None,
                    raw_artifact_id=raw.artifact_id,
                    ingested_at=max(raw.fetched_at, public_time),
                )
            )

        return tuple(events)

    @staticmethod
    def semantic_text(event: Event) -> str:
        if event.event_type is not EventType.LOBBYING_ACTIVITY:
            raise ValueError("semantic_text requires a lobbying event")
        payload = event.payload
        parts = [
            f"Client: {payload.client_name}",
            f"Registrant: {payload.registrant_name or 'unknown'}",
            f"Issue codes: {', '.join(payload.issue_codes) or 'none'}",
            f"Government entities: {', '.join(payload.government_entities) or 'none'}",
            "Specific lobbying issues:",
            *payload.specific_issues,
        ]
        return "\n".join(parts)

    @staticmethod
    def _text(raw: RawRecord) -> str:
        return raw.content.decode("utf-8") if isinstance(raw.content, bytes) else raw.content

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        parsed = datetime.fromisoformat(normalized)
        return as_utc(parsed)

    @classmethod
    def _reported_amount(cls, filing: dict) -> Decimal | None:
        for field in ("expenses", "income"):
            amount = cls._decimal_or_none(filing.get(field))
            if amount is not None:
                return amount
        return None

    @staticmethod
    def _collect_entities(value, destination: list[str]) -> None:
        if not isinstance(value, list):
            return
        for entity in value:
            if isinstance(entity, dict):
                name = str(entity.get("name") or "").strip()
            else:
                name = str(entity or "").strip()
            if name:
                destination.append(name)

    @staticmethod
    def _decimal_or_none(value) -> Decimal | None:
        if value is None:
            return None
        cleaned = str(value).strip().replace(",", "").replace("$", "")
        if not cleaned:
            return None
        try:
            return Decimal(cleaned)
        except InvalidOperation as exc:
            raise ValueError(f"invalid LDA amount: {value}") from exc

    @staticmethod
    def _int_or_none(value) -> int | None:
        if value is None or str(value).strip() == "":
            return None
        return int(value)
