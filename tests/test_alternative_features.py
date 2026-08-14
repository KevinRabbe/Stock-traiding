from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from stock_trading.core import (
    Event,
    EventType,
    GovernmentContractPayload,
    InsiderTransactionPayload,
    LobbyingActivityPayload,
    SemanticAnnotation,
    SemanticDirection,
    Source,
    TradeDirection,
    deterministic_event_id,
)
from stock_trading.features import (
    CompanyEventIndex,
    build_alternative_features,
    build_cross_source_features,
)


_COMPANY = "cmp_test"
_DECISION = datetime(2026, 8, 12, 20, 0, tzinfo=timezone.utc)


def _semantic(topic: str, importance: float = 0.9) -> SemanticAnnotation:
    return SemanticAnnotation(
        topics=(topic,),
        direction=SemanticDirection.POSITIVE,
        novelty=0.7,
        importance=importance,
        company_relevance=0.9,
        policy_relevance=0.8,
        confidence=0.9,
        model="Qwen/Qwen3.5-4B",
        extractor_version="semantic-v1",
        schema_version="semantic-v1",
    )


def _event(
    event_type: EventType,
    *,
    days_ago: int,
    suffix: str,
    payload,
    source: Source,
    semantic: SemanticAnnotation | None = None,
) -> Event:
    public_time = _DECISION - timedelta(days=days_ago)
    source_record_id = f"test:{suffix}"
    return Event(
        event_id=deterministic_event_id(source, source_record_id, event_type),
        event_type=event_type,
        company_id=_COMPANY,
        actor_id=None,
        event_time=public_time,
        public_time=public_time,
        first_tradable_time=None,
        source=source,
        source_record_id=source_record_id,
        payload=payload,
        semantic=semantic,
        raw_artifact_id=f"raw_{suffix}",
        ingested_at=public_time,
    )


def _insider(*, days_ago: int, buy: bool, suffix: str) -> Event:
    return _event(
        EventType.INSIDER_TRANSACTION,
        days_ago=days_ago,
        suffix=suffix,
        source=Source.SEC_EDGAR,
        payload=InsiderTransactionPayload(
            source_transaction_code="P" if buy else "S",
            direction=TradeDirection.BUY if buy else TradeDirection.SELL,
            shares=Decimal("100"),
            price=Decimal("10"),
            value=Decimal("1000"),
            intent_class="DISCRETIONARY_BUY" if buy else "DISCRETIONARY_SELL",
        ),
    )


def _contract(*, days_ago: int, amount: str, suffix: str) -> Event:
    return _event(
        EventType.GOVERNMENT_CONTRACT,
        days_ago=days_ago,
        suffix=suffix,
        source=Source.USASPENDING,
        payload=GovernmentContractPayload(
            award_id=f"award-{suffix}",
            transaction_id=f"tx-{suffix}",
            agency="Department of Defense",
            obligation_amount=Decimal(amount),
            description="Missile interceptor procurement",
        ),
        semantic=_semantic("DEFENSE.MISSILES"),
    )


def _lobbying(*, days_ago: int, amount: str, suffix: str) -> Event:
    return _event(
        EventType.LOBBYING_ACTIVITY,
        days_ago=days_ago,
        suffix=suffix,
        source=Source.LDA,
        payload=LobbyingActivityPayload(
            filing_id=f"filing-{suffix}",
            client_name="Example Defense Corp",
            amount=Decimal(amount),
            issue_codes=("DEF",),
            government_entities=("Department of Defense",),
            specific_issues=("Missile defense procurement",),
        ),
        semantic=_semantic("DEFENSE.MISSILES"),
    )


def test_alternative_features_use_only_information_public_by_decision_time() -> None:
    events = [
        _contract(days_ago=5, amount="300", suffix="recent-contract"),
        _contract(days_ago=40, amount="100", suffix="prior-contract"),
        _lobbying(days_ago=10, amount="200", suffix="recent-lobby"),
        _lobbying(days_ago=120, amount="50", suffix="prior-lobby"),
        _insider(days_ago=3, buy=True, suffix="buy"),
        # Negative days place the event in the future and it must not be visible.
        _contract(days_ago=-1, amount="999999", suffix="future-contract"),
    ]

    features = build_alternative_features(
        events,
        company_id=_COMPANY,
        decision_time=_DECISION,
    )

    assert features["contracts.obligation_30d"] == pytest.approx(300.0)
    assert features["contracts.obligation_change_30d"] == pytest.approx(200.0)
    assert features["lobbying.amount_90d"] == pytest.approx(200.0)
    assert features["lobbying.amount_change_90d"] == pytest.approx(150.0)
    assert features["cross.insider_plus_contract_30d"] == 1.0
    assert features["cross.insider_plus_lobbying_30d"] == 1.0
    assert features["cross.signal_family_count_30d"] == 3.0
    assert features["cross.shared_contract_lobbying_topics_90d"] == 1.0


def test_insider_sale_does_not_activate_buy_convergence() -> None:
    events = [
        _insider(days_ago=2, buy=False, suffix="sell"),
        _contract(days_ago=1, amount="500", suffix="contract"),
    ]

    features = build_cross_source_features(
        events,
        company_id=_COMPANY,
        decision_time=_DECISION,
    )

    assert features["cross.insider_plus_contract_30d"] == 0.0
    assert features["cross.insider_plus_lobbying_30d"] == 0.0
    assert features["cross.signal_family_count_30d"] == 1.0
    assert features["cross.days_insider_to_contract"] is None


def test_company_event_index_preserves_strict_temporal_boundaries() -> None:
    exact_30 = _contract(days_ago=30, amount="100", suffix="exact-30")
    inside_30 = _contract(days_ago=29, amount="200", suffix="inside-30")
    future = _contract(days_ago=-1, amount="999", suffix="future")
    index = CompanyEventIndex(_COMPANY, (future, inside_30, exact_30))

    recent = index.within(EventType.GOVERNMENT_CONTRACT, _DECISION, 30)
    assert [event.source_record_id for event in recent] == ["test:inside-30"]
    assert index.latest(EventType.GOVERNMENT_CONTRACT, _DECISION) == inside_30
    assert index.has_at_or_before(
        EventType.GOVERNMENT_CONTRACT,
        _DECISION - timedelta(days=30),
    )


def test_indexed_and_iterable_feature_inputs_are_identical() -> None:
    events = [
        _contract(days_ago=5, amount="300", suffix="recent-contract"),
        _contract(days_ago=40, amount="100", suffix="prior-contract"),
        _lobbying(days_ago=10, amount="200", suffix="recent-lobby"),
        _lobbying(days_ago=120, amount="50", suffix="prior-lobby"),
        _insider(days_ago=3, buy=True, suffix="buy"),
    ]
    index = CompanyEventIndex(_COMPANY, events)

    assert build_alternative_features(
        index,
        company_id=_COMPANY,
        decision_time=_DECISION,
    ) == build_alternative_features(
        events,
        company_id=_COMPANY,
        decision_time=_DECISION,
    )
