from collections.abc import Iterable
from datetime import datetime, timedelta
from statistics import mean

from stock_trading.core import Event, EventType, TradeDirection, as_utc

from .event_index import CompanyEventIndex, ensure_company_event_index


def build_contract_features(
    events: Iterable[Event] | CompanyEventIndex,
    *,
    company_id: str,
    decision_time: datetime,
) -> dict[str, float | None]:
    decision = as_utc(decision_time)
    index = ensure_company_event_index(events, company_id=company_id)
    features: dict[str, float | None] = {}

    windows = {
        days: index.within(EventType.GOVERNMENT_CONTRACT, decision, days)
        for days in (7, 30, 90, 365)
    }
    for days, recent in windows.items():
        values = [_contract_obligation(event) for event in recent]
        numeric = [value for value in values if value is not None]
        features[f"contracts.count_{days}d"] = float(len(recent))
        features[f"contracts.obligation_{days}d"] = sum(numeric) if numeric else 0.0

    recent_90 = windows[90]
    older_365 = index.between(EventType.GOVERNMENT_CONTRACT, decision, 365, 90)
    agencies = {
        event.payload.agency
        for event in recent_90
        if getattr(event.payload, "agency", None)
    }
    older_agencies = {
        event.payload.agency
        for event in older_365
        if getattr(event.payload, "agency", None)
    }
    obligations_90 = [
        value
        for value in (_contract_obligation(event) for event in recent_90)
        if value is not None
    ]
    features["contracts.unique_agencies_90d"] = float(len(agencies))
    features["contracts.new_agencies_90d"] = float(len(agencies - older_agencies))
    features["contracts.has_new_agency_relationship_90d"] = float(bool(agencies - older_agencies))
    features["contracts.first_contract_in_90d"] = float(
        bool(recent_90)
        and not index.has_at_or_before(
            EventType.GOVERNMENT_CONTRACT,
            decision - timedelta(days=90),
        )
    )
    features["contracts.largest_obligation_90d"] = max(obligations_90, default=0.0)
    features["contracts.agency_concentration_365d"] = _agency_concentration(windows[365])

    current_30 = _sum_contracts(windows[30])
    previous_30 = _sum_contracts(
        index.between(EventType.GOVERNMENT_CONTRACT, decision, 60, 30)
    )
    features["contracts.obligation_change_30d"] = current_30 - previous_30
    features["contracts.obligation_ratio_30d"] = (
        current_30 / previous_30 if previous_30 > 0 else None
    )

    prior_months = [
        _sum_contracts(
            index.between(
                EventType.GOVERNMENT_CONTRACT,
                decision,
                end,
                end - 30,
            )
        )
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
    events: Iterable[Event] | CompanyEventIndex,
    *,
    company_id: str,
    decision_time: datetime,
) -> dict[str, float | None]:
    decision = as_utc(decision_time)
    index = ensure_company_event_index(events, company_id=company_id)
    features: dict[str, float | None] = {}

    windows = {
        days: index.within(EventType.LOBBYING_ACTIVITY, decision, days)
        for days in (90, 365)
    }
    for days, recent in windows.items():
        amounts = [_lobbying_amount(event) for event in recent]
        numeric = [value for value in amounts if value is not None]
        features[f"lobbying.count_{days}d"] = float(len(recent))
        features[f"lobbying.amount_{days}d"] = sum(numeric) if numeric else 0.0

    recent_90 = windows[90]
    prior_90 = index.between(EventType.LOBBYING_ACTIVITY, decision, 180, 90)
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
    older_365 = index.between(EventType.LOBBYING_ACTIVITY, decision, 365, 90)
    older_topics = _lobbying_issue_codes(older_365)
    features["lobbying.unique_issue_codes_365d"] = float(
        len(_lobbying_issue_codes(windows[365]))
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
    events: Iterable[Event] | CompanyEventIndex,
    *,
    company_id: str,
    decision_time: datetime,
) -> dict[str, float | None]:
    decision = as_utc(decision_time)
    index = ensure_company_event_index(events, company_id=company_id)

    insider_30 = index.within(EventType.INSIDER_TRANSACTION, decision, 30)
    insider_90 = index.within(EventType.INSIDER_TRANSACTION, decision, 90)
    insider_180 = index.within(EventType.INSIDER_TRANSACTION, decision, 180)
    recent_insider_buys = [event for event in insider_30 if _is_discretionary_buy(event)]
    insider_buys_90 = [event for event in insider_90 if _is_discretionary_buy(event)]
    insider_buys_180 = [event for event in insider_180 if _is_discretionary_buy(event)]

    recent_contracts = index.within(EventType.GOVERNMENT_CONTRACT, decision, 30)
    contracts_90 = index.within(EventType.GOVERNMENT_CONTRACT, decision, 90)
    contracts_180 = index.within(EventType.GOVERNMENT_CONTRACT, decision, 180)
    recent_lobbying = index.within(EventType.LOBBYING_ACTIVITY, decision, 30)
    lobbying_90 = index.within(EventType.LOBBYING_ACTIVITY, decision, 90)
    lobbying_180 = index.within(EventType.LOBBYING_ACTIVITY, decision, 180)

    latest_insider = index.latest_matching(
        EventType.INSIDER_TRANSACTION,
        decision,
        _is_discretionary_buy,
    )
    latest_contract = index.latest(EventType.GOVERNMENT_CONTRACT, decision)
    latest_lobbying = index.latest(EventType.LOBBYING_ACTIVITY, decision)

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
        "cross.insider_buy_before_contract_90d": float(
            _ordered(insider_buys_90, contracts_90)
        ),
        "cross.lobbying_before_contract_90d": float(
            _ordered(lobbying_90, contracts_90)
        ),
        "cross.lobbying_then_insider_then_contract_180d": float(
            _three_stage_sequence(lobbying_180, insider_buys_180, contracts_180)
        ),
    }

    contract_topics = _semantic_topics(contracts_90)
    lobbying_topics = _semantic_topics(lobbying_90)
    shared_topics = contract_topics & lobbying_topics
    features["cross.shared_contract_lobbying_topics_90d"] = float(len(shared_topics))
    features["cross.topic_aligned_contract_lobbying_90d"] = float(bool(shared_topics))
    features["cross.relational_convergence_score"] = float(
        bool(recent_insider_buys)
        + bool(recent_contracts)
        + bool(recent_lobbying)
        + bool(shared_topics)
    )
    return features


def build_alternative_features(
    events: Iterable[Event] | CompanyEventIndex,
    *,
    company_id: str,
    decision_time: datetime,
) -> dict[str, float | None]:
    index = ensure_company_event_index(events, company_id=company_id)
    return {
        **build_contract_features(index, company_id=company_id, decision_time=decision_time),
        **build_lobbying_features(index, company_id=company_id, decision_time=decision_time),
        **build_cross_source_features(index, company_id=company_id, decision_time=decision_time),
    }


def _contract_obligation(event: Event) -> float | None:
    value = getattr(event.payload, "obligation_amount", None)
    return float(value) if value is not None else None


def _sum_contracts(events: Iterable[Event]) -> float:
    return sum(
        value
        for value in (_contract_obligation(event) for event in events)
        if value is not None
    )


def _agency_concentration(events: Iterable[Event]) -> float | None:
    by_agency: dict[str, float] = {}
    for event in events:
        agency = getattr(event.payload, "agency", None)
        value = _contract_obligation(event)
        if not agency or value is None or value <= 0:
            continue
        by_agency[agency] = by_agency.get(agency, 0.0) + value
    total = sum(by_agency.values())
    if total <= 0:
        return None
    return max(by_agency.values()) / total


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


def _days_between(left: Event | None, right: Event | None) -> float | None:
    if left is None or right is None:
        return None
    return (as_utc(right.public_time) - as_utc(left.public_time)).total_seconds() / 86400.0


def _ordered(first_events: Iterable[Event], second_events: Iterable[Event]) -> bool:
    return any(
        as_utc(left.public_time) < as_utc(right.public_time)
        for left in first_events
        for right in second_events
    )


def _three_stage_sequence(
    first_events: Iterable[Event],
    second_events: Iterable[Event],
    third_events: Iterable[Event],
) -> bool:
    return any(
        as_utc(one.public_time) < as_utc(two.public_time) < as_utc(three.public_time)
        for one in first_events
        for two in second_events
        for three in third_events
    )


def _is_discretionary_buy(event: Event) -> bool:
    intent = getattr(event.payload, "intent_class", None)
    if intent is not None:
        return intent == "DISCRETIONARY_BUY"
    return getattr(event.payload, "direction", None) is TradeDirection.BUY
