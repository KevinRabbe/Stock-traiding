from datetime import datetime
from decimal import Decimal
from typing import TypeAlias

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .enums import EventType, SemanticDirection, Source, TradeDirection
from .ids import deterministic_event_id
from .time import as_utc, is_public_at, is_tradable_at


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MarketBarPayload(FrozenModel):
    open: Decimal = Field(gt=0)
    high: Decimal = Field(gt=0)
    low: Decimal = Field(gt=0)
    close: Decimal = Field(gt=0)
    volume: Decimal = Field(ge=0)
    adjusted_close: Decimal | None = Field(default=None, gt=0)
    dividend_cash: Decimal | None = Field(default=None, ge=0)
    split_factor: Decimal | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_ohlc(self) -> "MarketBarPayload":
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high must be >= open, close, and low")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low must be <= open, close, and high")
        return self


class InsiderTransactionPayload(FrozenModel):
    source_transaction_code: str = Field(min_length=1)
    direction: TradeDirection
    shares: Decimal | None = Field(default=None, ge=0)
    price: Decimal | None = Field(default=None, ge=0)
    value: Decimal | None = Field(default=None, ge=0)
    insider_role: str | None = None
    ownership_type: str | None = None
    shares_owned_after: Decimal | None = Field(default=None, ge=0)
    intent_class: str | None = None
    is_10b5_1: bool | None = None


class GovernmentContractPayload(FrozenModel):
    award_id: str = Field(min_length=1)
    transaction_id: str | None = None
    agency: str | None = None
    subagency: str | None = None
    obligation_amount: Decimal | None = None
    total_obligation: Decimal | None = None
    potential_award_amount: Decimal | None = None
    award_type: str | None = None
    action_type: str | None = None
    modification_number: str | None = None
    naics_code: str | None = None
    psc_code: str | None = None
    description: str | None = None
    recipient_uei: str | None = None
    recipient_name: str | None = None


class LobbyingActivityPayload(FrozenModel):
    filing_id: str = Field(min_length=1)
    client_name: str
    registrant_name: str | None = None
    filing_year: int | None = None
    filing_period: str | None = None
    amount: Decimal | None = Field(default=None, ge=0)
    issue_codes: tuple[str, ...] = ()
    government_entities: tuple[str, ...] = ()
    specific_issues: tuple[str, ...] = ()


class FxRatePayload(FrozenModel):
    base: str = Field(min_length=3, max_length=12)
    quote: str = Field(min_length=3, max_length=12)
    rate: Decimal = Field(gt=0)


class CongressTransactionPayload(FrozenModel):
    politician_id: str = Field(min_length=1)
    direction: TradeDirection
    owner: str | None = None
    amount_min: Decimal | None = Field(default=None, ge=0)
    amount_max: Decimal | None = Field(default=None, ge=0)
    transaction_description: str | None = None

    # Context is deliberately first-class because pooled congressional trade
    # direction is not treated as a standalone alpha signal.
    politician_role: str | None = None
    is_leadership: bool | None = None
    leadership_rank: float | None = Field(default=None, ge=0.0, le=1.0)
    committees: tuple[str, ...] = ()
    committee_relevance: float | None = Field(default=None, ge=0.0, le=1.0)
    trade_size_percentile: float | None = Field(default=None, ge=0.0, le=1.0)
    public_sentiment_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    sentiment_divergence: float | None = Field(default=None, ge=-2.0, le=2.0)
    corporate_connection_score: float | None = Field(default=None, ge=0.0, le=1.0)
    home_state_connection: bool | None = None
    donor_connection: bool | None = None

    @model_validator(mode="after")
    def validate_amount_band(self) -> "CongressTransactionPayload":
        if (
            self.amount_min is not None
            and self.amount_max is not None
            and self.amount_min > self.amount_max
        ):
            raise ValueError("amount_min must be <= amount_max")
        return self


class CorporateActionPayload(FrozenModel):
    action_type: str = Field(min_length=1)
    split_factor: Decimal | None = Field(default=None, gt=0)
    dividend_cash: Decimal | None = Field(default=None, ge=0)
    description: str | None = None


Payload: TypeAlias = (
    MarketBarPayload
    | InsiderTransactionPayload
    | GovernmentContractPayload
    | LobbyingActivityPayload
    | FxRatePayload
    | CongressTransactionPayload
    | CorporateActionPayload
)


class SemanticAnnotation(FrozenModel):
    topics: tuple[str, ...] = ()
    direction: SemanticDirection = SemanticDirection.UNKNOWN
    novelty: float | None = Field(default=None, ge=0.0, le=1.0)
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    company_relevance: float | None = Field(default=None, ge=0.0, le=1.0)
    policy_relevance: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    model: str = Field(min_length=1)
    extractor_version: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)


_PAYLOAD_BY_EVENT_TYPE: dict[EventType, type[BaseModel]] = {
    EventType.MARKET_BAR: MarketBarPayload,
    EventType.INSIDER_TRANSACTION: InsiderTransactionPayload,
    EventType.GOVERNMENT_CONTRACT: GovernmentContractPayload,
    EventType.LOBBYING_ACTIVITY: LobbyingActivityPayload,
    EventType.FX_RATE: FxRatePayload,
    EventType.CONGRESS_TRANSACTION: CongressTransactionPayload,
    EventType.CORPORATE_ACTION: CorporateActionPayload,
}


class Event(FrozenModel):
    """Canonical immutable sparse-information event.

    `public_time` is the information firewall. Features may only use an event
    when public_time <= decision_time. `first_tradable_time` separately defines
    when execution is permitted.
    """

    event_id: str = Field(min_length=1)
    event_type: EventType
    event_index: int = Field(default=0, ge=0)

    company_id: str | None = None
    actor_id: str | None = None

    event_time: AwareDatetime
    public_time: AwareDatetime
    first_tradable_time: AwareDatetime | None = None

    source: Source
    source_record_id: str = Field(min_length=1)
    payload: Payload
    semantic: SemanticAnnotation | None = None
    raw_artifact_id: str = Field(min_length=1)
    ingested_at: AwareDatetime

    @field_validator(
        "event_time",
        "public_time",
        "first_tradable_time",
        "ingested_at",
        mode="before",
    )
    @classmethod
    def normalize_timestamps(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return as_utc(value)

    @model_validator(mode="after")
    def validate_contract(self) -> "Event":
        expected_payload = _PAYLOAD_BY_EVENT_TYPE[self.event_type]
        if not isinstance(self.payload, expected_payload):
            raise ValueError(
                f"{self.event_type.value} requires payload {expected_payload.__name__}"
            )

        expected_id = deterministic_event_id(
            self.source,
            self.source_record_id,
            self.event_type,
            self.event_index,
        )
        if self.event_id != expected_id:
            raise ValueError("event_id is not deterministic for this source record")

        if self.event_time > self.public_time:
            raise ValueError("event_time cannot be after public_time")
        if self.first_tradable_time is not None and self.first_tradable_time < self.public_time:
            raise ValueError("first_tradable_time cannot precede public_time")
        if self.ingested_at < self.public_time:
            raise ValueError("ingested_at cannot precede public_time")
        return self

    def is_public_at(self, decision_time: datetime) -> bool:
        return is_public_at(self.public_time, decision_time)

    def is_tradable_at(self, decision_time: datetime) -> bool:
        return is_tradable_at(self.first_tradable_time, decision_time)
