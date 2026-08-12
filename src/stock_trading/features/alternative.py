from collections.abc import Iterable
from datetime import datetime, timedelta
from statistics import mean

from stock_trading.core import Event, EventType, TradeDirection, as_utc


def build_contract_features(
    events: Iterable[Event],
    *,
    company_id: str,
    decision_time: datetime,
) -> dict[str, float | None]:
    decision = as_utc(decision_time)
    contracts = _events_for(events, company_id, EventType.GOVERNMENT_CONTRACT, decision)
    features: dict[str, float | None] = {}

    for days in (7, 30, 90, 365):
        recent = _within(contracts, decision, days)
        values = [_contract_obligation(event) for event in recent]
        numeric = [value for value in values if value is not None]
        features[f"contracts.count_{days}d"] = float(len(recent))
        features[f"contracts.obligation_{days}d"] = sum(numeric) if numeric else 0.0

    recent_90 = _within(contracts, decision, 90)
    agencies = {
        event.payload.agency
        for event in recent_90
        if getattr(event.payload, "agency", None)
    }
    obligations_90 = [
        value
        for value in (_contract_obligation(event) for event in recent_90)
        if value is not None
    ]
    features["contracts.unique_agencies_90d"] = float(len(agencies))
    features["contracts.largest_obligation_90d"] = max(obligations_90, default=0.0)

    current_30 = _sum_contracts(_within(contracts, decision, 30))
    previous_30 = _sum_contracts(_between(contracts, decision, 60, 30))
    features["contracts.obligation_change_30d"] = current_30 - previous_30
    features["contracts.obligation_ratio_30d"] = (
        current_30 / previous_30 if previous_30 > 0 else None
    )

    prior_months = [
        _sum_contracts(_between(contracts, decision, end, end - 30))
        for end in range(60, 361, 30)
    ]
    positive_history = [value for value in prior_months if value > 0]
    baseline = mean(positive_history) if positive_history else 0.0
    features["contracts.surprise_30d"] = current_30 / baseline if baseline > 0 else None
    features["contracts.high_importance_semantic_90d"] = float(
        sum(
            1
            for event in recent_90
            if event.semantic is not None
            and event.semantic.importance is not None
            and event.semantic.importance >= 0.75
        )
    )
    return features


def build_lobbying_features(
    events: Iterable[Event],
    *,
    company_id: str,
    decision_time: datetime,
) -> dict[str, float | None]:
    decision = as_utc(decision_time)
    filings = _events_for(events, company_id, EventType.LOBBYING_ACTIVITY, decision)
    features: dict[str, float | None] = {}

    for days in (90, 365):
        recent = _within(filings, decision, days)
        amounts = [_lobbying_amount(event) for event in recent]
        numeric = [value for value in amounts if value is not None]
        features[f"lobbying.count_{days}d"] = float(len(recent))
        features[f"lobbying.amount_{days}d"] = sum(numeric) if numeric else 0.0

    recent_90 = _within(filings, decision, 90)
    prior_90 = _between(filings, decision, 180, 90)
    recent_amount = sum(
        value for value in (_lobbying_amount(event) for event in recent_90) if value is not None
    )
    prior_amount = sum(
        value for value in (_lobbying_amount(event) for event in prior_90) if value is not None
    )
    features["lobbying.amount_change_90d"] = recent_amount - prior_amount
    features["lobbying.amount_ratio_90d"] = (
        recent_amount / prior_amount if prior_amount > 0 else None
    )

    recent_topics = _lobbying_issue_codes(recent_90)
    older_365 = _between(filings, decision, 365, 90)
    older_topics = _lobbying_issue_codes(older_365)
    features["lobbying.unique_issue_codes_365d"] = float(
        len(_lobbying_issue_codes(_within(filings, decision, 365)))
    )
    features["lobbying.new_issue_codes_90d"] = float(len(recent_topics - older_topics))

    recent_entities = {
        entity
        for event in recent_90
        for entity in getattr(event.payload, "government_entities", ())
    }
    older_entities = {
        entity
        for event in older_365
        for entity in getattr(event.payload, "government_entities", ())
    }
    features["lobbying.unique_government_entities_90d"] = float(len(recent_entities))
    features["lobbying.new_government_entities_90d"] = float(
        len(recent_entities - older_entities)
    )
    features["lobbying.high_importance_semantic_90d"] = float(
        sum(
            1
            for event in recent_90
            if event.semantic is not None
            and event.semantic.importance is not None
            and event.semantic.importance >= 0.75
        )
    )
    return features


def build_cross_source_features(
    events: Iterable[Event],
    *,
    company_id: str,
    decision_time: datetime,
) -> dict[str, float | None]:
    decision = as_utc(decision_time)
    visible = [
        event
        for event in events
        if event.company_id == company_id and event.public_time <= decision
    ]

    insider_buys = [
        event
        for event in visible
        if event.event_type is EventType.INSIDER_TRANSACTION
        and (
            getattr(event.payload, "intent_class", None) == "DISCRETIONARY_BUY"
            or getattr(event.payload, "direction", None) is TradeDirection.BUY
        )
    ]
    contracts = [event for event in visible if event.event_type is EventType.GOVERNMENT_CONTRACT]
    lobbying = [event for event in visible if event.event_type is EventType.LOBBYING_ACTIVITY]

    recent_insider_buys = _within(insider_buys, decision, 30)
    recent_contracts = _within(contracts, decision, 30)
    recent_lobbying = _within(lobbying, decision, 30)

    latest_insider = _latest(insider_buys)
    latest_contract = _latest(contracts)
    latest_lobbying = _latest(lobbying)

    features: dict[str, float | None] = {
        "cross.insider_plus_contract_30d": float(
            bool(recent_insider_buys) and bool(recent_contracts)
        ),
        "cross.insider_plus_lobbying_30d": float(
            bool(recent_insider_buys) and bool(recent_lobbying)
        ),
        "cross.signal_family_count_30d": float(
            sum(
                (
                    bool(recent_insider_buys),
                    bool(recent_contracts),
                    bool(recent_lobbying),
                )
            )
        ),
        "cross.days_insider_to_contract": _days_between(latest_insider, latest_contract),
        "cross.days_lobbying_to_contract": _days_between(latest_lobbying, latest_contract),
    }

    contract_topics = _semantic_topics(_within(contracts, decision, 90))
    lobbying_topics = _semantic_topics(_within(lobbying, decision, 90))
    features["cross.shared_contract_lobbying_topics_90d"] = float(
        len(contract_topics & lobbying_topics)
    )
    return features


def build_alternative_features(
    events: Iterable[Event],
    *,
    company_id: str,
    decision_time: datetime,
) -> dict[str, float | None]:
    materialized = tuple(events)
    return {
        **build_contract_features(materialized, company_id=company_id, decision_time=decision_time),
        **build_lobbying_features(materialized, company_id=company_id, decision_time=decision_time),
        **build_cross_source_features(materialized, company_id=company_id, decision_time=decision_time),
    }


def _events_for(
    events: Iterable[Event],
    company_id: str,
    event_type: EventType,
    decision_time: datetime,
) -> list[Event]:
    return sorted(
        (
            event
            for event in events
            if event.company_id == company_id
            and event.event_type is event_type
            and event.public_time <= decision_time
        ),
        key=lambda event: event.public_time,
    )


def _within(events: Iterable[Event], decision: datetime, days: int) -> list[Event]:
    cutoff = decision - timedelta(days=days)
    return [event for event in events if cutoff < event.public_time <= decision]


def _between(
    events: Iterable[Event],
    decision: datetime,
    older_days: int,
    newer_days: int,
) -> list[Event]:
    older = decision - timedelta(days=older_days)
    newer = decision - timedelta(days=newer_days)
    return [event for event in events if older < event.public_time <= newer]


def _contract_obligation(event: Event) -> float | None:
    value = getattr(event.payload, "obligation_amount", None)
    return float(value) if value is not None else None


def _sum_contracts(events: Iterable[Event]) -> float:
    return sum(
        value
        for value in (_contract_obligation(event) for event in events)
        if value is not None
    )


def _lobbying_amount(event: Event) -> float | None:
    value = getattr(event.payload, "amount", None)
    return float(value) if value is not None else None


def _lobbying_issue_codes(events: Iterable[Event]) -> set[str]:
    return {
        code
        for event in events
        for code in getattr(event.payload, "issue_codes", ())
    }


def _semantic_topics(events: Iterable[Event]) -> set[str]:
    return {
        topic
        for event in events
        if event.semantic is not None
        for topic in event.semantic.topics
    }


def _latest(events: Iterable[Event]) -> Event | None:
    return max(events, key=lambda event: event.public_time, default=None)


def _days_between(left: Event | None, right: Event | None) -> float | None:
    if left is None or right is None:
        return None
    return (right.public_time - left.public_time).total_seconds() / 86400.0
