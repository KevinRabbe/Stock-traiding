from collections.abc import Iterable
from datetime import datetime

from stock_trading.core import Event, EventType, TradeDirection, as_utc

from .event_index import CompanyEventIndex, ensure_company_event_index


def build_insider_features(
    events: Iterable[Event] | CompanyEventIndex,
    *,
    company_id: str,
    decision_time: datetime,
) -> dict[str, float | None]:
    decision = as_utc(decision_time)
    index = ensure_company_event_index(events, company_id=company_id)

    features: dict[str, float | None] = {}
    for days in (7, 30, 90):
        recent = index.within(EventType.INSIDER_TRANSACTION, decision, days)
        buys = [event for event in recent if _is_discretionary_buy(event)]
        sells = [event for event in recent if _is_discretionary_sell(event)]
        open_market_buys = [event for event in buys if _is_open_market_purchase(event)]

        features[f"insider.buy_count_{days}d"] = float(len(buys))
        features[f"insider.sell_count_{days}d"] = float(len(sells))
        features[f"insider.open_market_buy_count_{days}d"] = float(len(open_market_buys))
        features[f"insider.buy_value_{days}d"] = _sum_values(buys)
        features[f"insider.sell_value_{days}d"] = _sum_values(sells)
        features[f"insider.open_market_buy_value_{days}d"] = _sum_values(open_market_buys)
        features[f"insider.net_value_{days}d"] = (
            features[f"insider.buy_value_{days}d"]
            - features[f"insider.sell_value_{days}d"]
        )
        features[f"insider.unique_buyers_{days}d"] = float(
            len({event.actor_id for event in buys if event.actor_id})
        )

    recent_90 = index.within(EventType.INSIDER_TRANSACTION, decision, 90)
    buys_90 = [event for event in recent_90 if _is_discretionary_buy(event)]
    sells_90 = [event for event in recent_90 if _is_discretionary_sell(event)]
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
    features["insider.non_10b5_1_sell_value_90d"] = _sum_values(
        event for event in sells_90 if getattr(event.payload, "is_10b5_1", None) is not True
    )
    features["insider.10b5_1_sell_value_90d"] = _sum_values(
        event for event in sells_90 if getattr(event.payload, "is_10b5_1", None) is True
    )

    holding_fractions = [
        fraction
        for event in buys_90
        if (fraction := _purchase_fraction_of_post_holdings(event)) is not None
    ]
    features["insider.max_buy_fraction_post_holdings_90d"] = (
        max(holding_fractions) if holding_fractions else None
    )
    features["insider.avg_buy_fraction_post_holdings_90d"] = (
        sum(holding_fractions) / len(holding_fractions) if holding_fractions else None
    )

    latest_buy = max(buys_90, key=lambda event: event.public_time, default=None)
    latest_sell = max(sells_90, key=lambda event: event.public_time, default=None)
    features["insider.days_since_latest_buy"] = _days_since(latest_buy, decision)
    features["insider.days_since_latest_sell"] = _days_since(latest_sell, decision)
    features["insider.latest_buy_fraction_post_holdings"] = (
        _purchase_fraction_of_post_holdings(latest_buy) if latest_buy is not None else None
    )
    features["insider.cluster_buy_30d"] = float(features["insider.unique_buyers_30d"] >= 2)
    return features


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


def _is_open_market_purchase(event: Event) -> bool:
    return str(getattr(event.payload, "source_transaction_code", "")).upper() == "P"


def _purchase_fraction_of_post_holdings(event: Event) -> float | None:
    shares = getattr(event.payload, "shares", None)
    shares_after = getattr(event.payload, "shares_owned_after", None)
    if shares is None or shares_after is None or float(shares_after) <= 0:
        return None
    return min(1.0, max(0.0, float(shares) / float(shares_after)))


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
    return (decision - as_utc(event.public_time)).total_seconds() / 86400.0
