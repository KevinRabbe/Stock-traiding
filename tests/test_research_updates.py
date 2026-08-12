from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from stock_trading.core import (
    CongressTransactionPayload,
    Event,
    EventType,
    GovernmentContractPayload,
    InsiderTransactionPayload,
    LobbyingActivityPayload,
    Source,
    TradeDirection,
    deterministic_event_id,
)
from stock_trading.experiments.prepare import latest_completed_quarter
from stock_trading.features import (
    build_alternative_features,
    build_congress_features,
    build_insider_features,
)
from stock_trading.ml.dataset import build_research_interactions


UTC = timezone.utc
COMPANY = "cmp_test"
BASE = datetime(2026, 6, 30, 16, 0, tzinfo=UTC)


def _event(event_type, source, record_id, public_time, payload, actor_id=None):
    return Event(
        event_id=deterministic_event_id(source, record_id, event_type),
        event_type=event_type,
        company_id=COMPANY,
        actor_id=actor_id,
        event_time=public_time - timedelta(hours=1),
        public_time=public_time,
        first_tradable_time=None,
        source=source,
        source_record_id=record_id,
        payload=payload,
        semantic=None,
        raw_artifact_id=f"raw-{record_id}",
        ingested_at=public_time + timedelta(minutes=1),
    )


def test_sec_default_history_includes_completed_2026_q2() -> None:
    assert latest_completed_quarter(date(2026, 8, 13)) == (2026, 2)


def test_insider_features_capture_open_market_and_holding_fraction() -> None:
    buy = _event(
        EventType.INSIDER_TRANSACTION,
        Source.SEC_EDGAR,
        "buy-1",
        BASE - timedelta(days=5),
        InsiderTransactionPayload(
            source_transaction_code="P",
            direction=TradeDirection.BUY,
            shares=Decimal("100"),
            price=Decimal("20"),
            value=Decimal("2000"),
            shares_owned_after=Decimal("500"),
            insider_role="CEO",
            intent_class="DISCRETIONARY_BUY",
            is_10b5_1=False,
        ),
        actor_id="owner-1",
    )
    features = build_insider_features((buy,), company_id=COMPANY, decision_time=BASE)

    assert features["insider.open_market_buy_count_30d"] == 1.0
    assert features["insider.open_market_buy_value_30d"] == 2000.0
    assert features["insider.latest_buy_fraction_post_holdings"] == 0.2
    assert features["insider.non_10b5_1_buy_value_90d"] == 2000.0


def test_contract_and_sequence_features_reward_new_relationship_order() -> None:
    lobbying = _event(
        EventType.LOBBYING_ACTIVITY,
        Source.LDA,
        "lda-1",
        BASE - timedelta(days=70),
        LobbyingActivityPayload(
            filing_id="lda-1",
            client_name="Example Corp",
            amount=Decimal("100000"),
            issue_codes=("DEF",),
            government_entities=("Department of Defense",),
        ),
    )
    insider = _event(
        EventType.INSIDER_TRANSACTION,
        Source.SEC_EDGAR,
        "buy-2",
        BASE - timedelta(days=40),
        InsiderTransactionPayload(
            source_transaction_code="P",
            direction=TradeDirection.BUY,
            shares=Decimal("50"),
            value=Decimal("5000"),
            intent_class="DISCRETIONARY_BUY",
        ),
        actor_id="owner-2",
    )
    contract = _event(
        EventType.GOVERNMENT_CONTRACT,
        Source.USASPENDING,
        "contract-1",
        BASE - timedelta(days=10),
        GovernmentContractPayload(
            award_id="award-1",
            transaction_id="tx-1",
            agency="Department of Defense",
            obligation_amount=Decimal("25000000"),
            recipient_name="Example Corp",
        ),
    )

    features = build_alternative_features(
        (lobbying, insider, contract),
        company_id=COMPANY,
        decision_time=BASE,
    )
    assert features["contracts.new_agencies_90d"] == 1.0
    assert features["contracts.first_contract_in_90d"] == 1.0
    assert features["cross.insider_buy_before_contract_90d"] == 1.0
    assert features["cross.lobbying_before_contract_90d"] == 1.0
    assert features["cross.lobbying_then_insider_then_contract_180d"] == 1.0


def test_microcap_style_price_state_interactions_are_explicit() -> None:
    features = build_research_interactions(
        {
            "market.appreciation_gt_10pct_20d": 1.0,
            "market.within_10pct_252d_high": 1.0,
            "insider.open_market_buy_count_30d": 1.0,
            "insider.cluster_buy_30d": 1.0,
            "contracts.surprise_30d": 2.5,
            "lobbying.new_issue_codes_90d": 1.0,
            "cross.relational_convergence_30d": 4.0,
        }
    )
    assert features["interaction.insider_buy_after_10pct_appreciation_20d"] == 1.0
    assert features["interaction.cluster_buy_near_52w_high"] == 1.0
    assert features["interaction.contract_acceleration_plus_new_lobbying_topic"] == 1.0
    assert features["interaction.multi_source_convergence"] == 1.0


def test_congress_features_are_disabled_by_default_and_conditional_when_enabled() -> None:
    trade = _event(
        EventType.CONGRESS_TRANSACTION,
        Source.CONGRESS,
        "congress-1",
        BASE - timedelta(days=10),
        CongressTransactionPayload(
            politician_id="member-1",
            direction=TradeDirection.BUY,
            amount_min=Decimal("100000"),
            amount_max=Decimal("250000"),
            is_leadership=True,
            leadership_rank=0.95,
            committee_relevance=0.9,
            trade_size_percentile=0.98,
            corporate_connection_score=0.8,
            sentiment_divergence=0.6,
        ),
        actor_id="member-1",
    )

    disabled = build_congress_features(
        (trade,), company_id=COMPANY, decision_time=BASE
    )
    assert disabled == {"congress.enabled": 0.0}

    enabled = build_congress_features(
        (trade,), company_id=COMPANY, decision_time=BASE, enabled=True
    )
    assert enabled["congress.leadership_buy_count_90d"] == 1.0
    assert enabled["congress.max_committee_relevance_90d"] == 0.9
    assert enabled["congress.power_committee_buy_score_90d"] > 3.0
