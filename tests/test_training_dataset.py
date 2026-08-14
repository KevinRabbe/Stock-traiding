from datetime import date, datetime, timezone
from decimal import Decimal

from stock_trading.core import (
    Event,
    EventType,
    InsiderTransactionPayload,
    Source,
    TradeDirection,
    deterministic_event_id,
)
from stock_trading.market import CandidateSnapshot, ForwardLabel, LabeledCandidate
from stock_trading.ml import TrainingDatasetBuilder


class _FakeSnapshotBuilder:
    def build(self, event: Event) -> CandidateSnapshot:
        return CandidateSnapshot(
            event_id=event.event_id,
            company_id=event.company_id,
            security_id="security_example",
            decision_time=event.public_time,
            decision_market_date=date(2026, 8, 7),
            execution_date=date(2026, 8, 10),
            first_tradable_time=datetime(2026, 8, 10, 13, 30, tzinfo=timezone.utc),
            execution_ticker="EXM",
            market_features={
                "market.return_20d": 0.04,
                "market.volatility_20d": 0.02,
            },
        )

    def label(self, snapshot: CandidateSnapshot) -> LabeledCandidate:
        return LabeledCandidate(
            snapshot=snapshot,
            labels=(
                ForwardLabel(
                    horizon=20,
                    start_date=date(2026, 8, 10),
                    end_date=date(2026, 9, 4),
                    stock_return=0.08,
                    benchmark_return=0.02,
                    alpha=0.06,
                    max_favorable_excursion=0.12,
                    max_adverse_excursion=-0.03,
                ),
            ),
        )


def _trigger() -> Event:
    public_time = datetime(2026, 8, 7, 18, 0, tzinfo=timezone.utc)
    source_record_id = "filing:tx:0"
    return Event(
        event_id=deterministic_event_id(
            Source.SEC_EDGAR,
            source_record_id,
            EventType.INSIDER_TRANSACTION,
        ),
        event_type=EventType.INSIDER_TRANSACTION,
        company_id="cmp_example",
        actor_id="sec_owner_cik_0000054321",
        event_time=datetime(2026, 8, 5, 4, 0, tzinfo=timezone.utc),
        public_time=public_time,
        first_tradable_time=None,
        source=Source.SEC_EDGAR,
        source_record_id=source_record_id,
        payload=InsiderTransactionPayload(
            source_transaction_code="P",
            direction=TradeDirection.BUY,
            shares=Decimal("100"),
            price=Decimal("20"),
            value=Decimal("2000"),
            insider_role="OFFICER:Chief Executive Officer",
            intent_class="DISCRETIONARY_BUY",
            is_10b5_1=False,
        ),
        semantic=None,
        raw_artifact_id="raw_dataset_test",
        ingested_at=public_time,
    )


def test_training_dataset_combines_inputs_without_identity_leakage() -> None:
    trigger = _trigger()
    rows = TrainingDatasetBuilder(
        _FakeSnapshotBuilder(),
        positive_alpha_threshold=0.02,
        target_horizon=20,
    ).build([trigger], all_events=[trigger])

    assert len(rows) == 1
    row = rows[0]
    assert row.company_id == "cmp_example"
    assert row.execution_date == date(2026, 8, 10)
    assert row.exit_date_20d == date(2026, 9, 4)
    assert row.stock_return_20d == 0.08
    assert row.benchmark_return_20d == 0.02
    assert row.alpha_20d == 0.06
    assert row.downside_20d == 0.03
    assert row.positive_alpha_20d == 1

    assert row.features["market.return_20d"] == 0.04
    assert row.features["insider.buy_count_7d"] == 1.0
    assert row.features["insider.ceo_buy_count_90d"] == 1.0
    assert row.features["trigger.is_insider"] == 1.0
    assert row.features["trigger.source_value"] == 2000.0
    assert not any("company_id" in name or "ticker" in name for name in row.features)
