from collections.abc import Iterable
from datetime import datetime

from stock_trading.core import Event, EventType, TradeDirection, as_utc

from .event_index import CompanyEventIndex, ensure_company_event_index


def build_congress_features(
    events: Iterable[Event] | CompanyEventIndex,
    *,
    company_id: str,
    decision_time: datetime,
    enabled: bool = False,
) -> dict[str, float | None]:
    """Build conditional congressional features without treating raw direction as alpha.

    Congressional financial-disclosure data remains disabled by default. When enabled,
    the representation emphasizes leadership/power, committee relevance, corporate
    connections, abnormal size, and divergence from public sentiment rather than a
    pooled "member bought/sold" directional signal.
    """

    if not enabled:
        return {"congress.enabled": 0.0}

    decision = as_utc(decision_time)
    index = ensure_company_event_index(events, company_id=company_id)
    recent_90 = index.within(EventType.CONGRESS_TRANSACTION, decision, 90)
    buys = [event for event in recent_90 if event.payload.direction is TradeDirection.BUY]

    leadership_buys = [
        event for event in buys if getattr(event.payload, "is_leadership", None) is True
    ]
    connected_buys = [
        event
        for event in buys
        if (
            getattr(event.payload, "corporate_connection_score", None) is not None
            and float(event.payload.corporate_connection_score) > 0
        )
        or getattr(event.payload, "home_state_connection", None) is True
        or getattr(event.payload, "donor_connection", None) is True
    ]

    return {
        "congress.enabled": 1.0,
        "congress.trade_count_90d": float(len(recent_90)),
        "congress.leadership_buy_count_90d": float(len(leadership_buys)),
        "congress.connected_buy_count_90d": float(len(connected_buys)),
        "congress.max_committee_relevance_90d": _max_optional(
            recent_90, "committee_relevance"
        ),
        "congress.avg_leadership_rank_90d": _avg_optional(
            recent_90, "leadership_rank"
        ),
        "congress.max_trade_size_percentile_90d": _max_optional(
            recent_90, "trade_size_percentile"
        ),
        "congress.avg_sentiment_divergence_90d": _avg_optional(
            recent_90, "sentiment_divergence"
        ),
        "congress.power_committee_buy_score_90d": sum(
            _conditional_buy_weight(event) for event in buys
        ),
    }


def _avg_optional(events: Iterable[Event], attribute: str) -> float | None:
    values = [
        float(value)
        for event in events
        if (value := getattr(event.payload, attribute, None)) is not None
    ]
    return sum(values) / len(values) if values else None


def _max_optional(events: Iterable[Event], attribute: str) -> float | None:
    values = [
        float(value)
        for event in events
        if (value := getattr(event.payload, attribute, None)) is not None
    ]
    return max(values) if values else None


def _conditional_buy_weight(event: Event) -> float:
    payload = event.payload
    leadership = float(getattr(payload, "leadership_rank", None) or 0.0)
    committee = float(getattr(payload, "committee_relevance", None) or 0.0)
    connection = float(getattr(payload, "corporate_connection_score", None) or 0.0)
    size = float(getattr(payload, "trade_size_percentile", None) or 0.0)
    return leadership + committee + connection + size
