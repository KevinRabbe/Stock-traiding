from collections.abc import Iterable
from datetime import datetime, timedelta

from stock_trading.core import Event, EventType, TradeDirection, as_utc


def build_insider_features(
    events: Iterable[Event],
    *,
    company_id: str,
    decision_time: datetime,
) -> dict[str, float | None]:
    decision = as_utc(decision_time)
    insiders = sorted(
        (
            event
            for event in events
            if event.company_id == company_id
            and event.event_type is EventType.INSIDER_TRANSACTION
            and event.public_time <= decision
        ),
        key=lambda event: event.public_time,
    )

    features: dict[str, float | None] = {}
    for days in (7, 30, 90):
        recent = _within(insiders, decision, days)
        buys = [event for event in recent if _is_discretionary_buy(event)]
        sells = [event for event in recent if _is_discretionary_sell(event)]

        features[f"insider.buy_count_{days}d"] = float(len(buys))
        features[f"insider.sell_count_{days}d"] = float(len(sells))
        features[f"insider.buy_value_{days}d"] = _sum_values(buys)
        features[f"insider.sell_value_{days}d"] = _sum_values(sells)
        features[f"insider.net_value_{days}d"] = (
            features[f"insider.buy_value_{days}d"]
            - features[f"insider.sell_value_{days}d"]
        )
        features[f"insider.unique_buyers_{days}d"] = float(
            len({event.actor_id for event in buys if event.actor_id})
        )

    recent_90 = _within(insiders, decision, 90)
    buys_90 = [event for event in recent_90 if _is_discretionary_buy(event)]
    features["insider.ceo_buy_count_90d"] = float(
        sum(_role_contains(event, "CEO", "CHIEF EXECUTIVE") for event in buys_90)
    )
    features["insider.cfo_buy_count_90d"] = float(
        sum(_role_contains(event, "CFO", "CHIEF FINANCIAL") for event in buys_90)
    )
    features["insider.director_buy_count_90d"] = float(
        sum(_role_contains(event, "DIRECTOR") for event in buys_90)
    )
    features["insider.non_10b5_1_buy_value_90d"] = _sum_values(
        event for event in buys_90 if getattr(event.payload, "is_10b5_1", None) is not True
    )
    features["insider.10b5_1_sell_value_90d"] = _sum_values(
        event
        for event in recent_90
        if _is_discretionary_sell(event)
        and getattr(event.payload, "is_10b5_1", None) is True
    )

    latest_buy = max(buys_90, key=lambda event: event.public_time, default=None)
    latest_sell = max(
        (event for event in recent_90 if _is_discretionary_sell(event)),
        key=lambda event: event.public_time,
        default=None,
    )
    features["insider.days_since_latest_buy"] = _days_since(latest_buy, decision)
    features["insider.days_since_latest_sell"] = _days_since(latest_sell, decision)
    features["insider.cluster_buy_30d"] = float(features["insider.unique_buyers_30d"] >= 2)
    return features


def _within(events: Iterable[Event], decision: datetime, days: int) -> list[Event]:
    cutoff = decision - timedelta(days=days)
    return [event for event in events if cutoff < event.public_time <= decision]


def _is_discretionary_buy(event: Event) -> bool:
    intent = getattr(event.payload, "intent_class", None)
    if intent is not None:
        return intent == "DISCRETIONARY_BUY"
    return getattr(event.payload, "direction", None) is TradeDirection.BUY


def _is_discretionary_sell(event: Event) -> bool:
    intent = getattr(event.payload, "intent_class", None)
    if intent is not None:
        return intent == "DISCRETIONARY_SELL"
    return getattr(event.payload, "direction", None) is TradeDirection.SELL


def _sum_values(events: Iterable[Event]) -> float:
    total = 0.0
    for event in events:
        value = getattr(event.payload, "value", None)
        if value is not None:
            total += float(value)
    return total


def _role_contains(event: Event, *needles: str) -> bool:
    role = str(getattr(event.payload, "insider_role", None) or "").upper()
    return any(needle in role for needle in needles)


def _days_since(event: Event | None, decision: datetime) -> float | None:
    if event is None:
        return None
    return (decision - event.public_time).total_seconds() / 86400.0
