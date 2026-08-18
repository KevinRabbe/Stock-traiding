from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from stock_trading.core import (
    Event,
    EventType,
    InsiderTransactionPayload,
    Source,
    TradeDirection,
    deterministic_event_id,
)
from stock_trading.market import (
    CandidateSnapshot,
    CandidateSnapshotBuilder,
    DuckDbMarketStore,
    ForwardLabel,
    LabeledCandidate,
    MarketBar,
    SecurityMapping,
)
from stock_trading.ml import TrainingDatasetBuilder
from stock_trading.live.candidates import PitCandidateAssembler


def _bar(
    security_id: str,
    ticker: str,
    day: date,
    close: str,
) -> MarketBar:
    price = Decimal(close)
    return MarketBar(
        security_id=security_id,
        ticker=ticker,
        date=day,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=Decimal("1000000"),
        adj_open=price,
        adj_high=price,
        adj_low=price,
        adj_close=price,
        adj_volume=Decimal("1000000"),
    )


def _event(record_id: str, public_time: datetime, *, value: str = "1000") -> Event:
    return Event(
        event_id=deterministic_event_id(
            Source.SEC_EDGAR,
            record_id,
            EventType.INSIDER_TRANSACTION,
        ),
        event_type=EventType.INSIDER_TRANSACTION,
        company_id="cmp_example",
        actor_id=f"actor_{record_id}",
        event_time=public_time,
        public_time=public_time,
        first_tradable_time=None,
        source=Source.SEC_EDGAR,
        source_record_id=record_id,
        payload=InsiderTransactionPayload(
            source_transaction_code="P",
            direction=TradeDirection.BUY,
            shares=Decimal("100"),
            price=Decimal("10"),
            value=Decimal(value),
            insider_role="OFFICER:Chief Executive Officer",
            intent_class="DISCRETIONARY_BUY",
            is_10b5_1=False,
        ),
        semantic=None,
        raw_artifact_id=f"raw_{record_id}",
        ingested_at=public_time,
    )


def test_scheduled_snapshot_needs_no_execution_day_bar_and_rejects_skipped_session(
    tmp_path,
) -> None:
    pytest.importorskip("duckdb")
    stock = "security_stock"
    benchmark = "benchmark_spy"
    store = DuckDbMarketStore(tmp_path / "market.duckdb")
    store.register_mapping(
        SecurityMapping(
            company_id="cmp_example",
            security_id=stock,
            ticker="EXM",
            valid_from=date(2020, 1, 1),
        )
    )
    store.put_many(
        [
            _bar(stock, "EXM", date(2026, 8, 14), "100"),
            _bar(stock, "EXM", date(2026, 8, 17), "101"),
            _bar(benchmark, "SPY", date(2026, 8, 14), "500"),
            _bar(benchmark, "SPY", date(2026, 8, 17), "501"),
        ]
    )
    event = _event(
        "current",
        datetime(2026, 8, 17, 20, 30, tzinfo=timezone.utc),
    )
    builder = CandidateSnapshotBuilder(store, benchmark_security_id=benchmark)

    with pytest.raises(ValueError, match="no future market bar"):
        builder.build(event)

    snapshot = builder.build_for_execution_date(event, date(2026, 8, 18))
    assert snapshot.execution_date == date(2026, 8, 18)
    assert snapshot.security_id == stock
    assert snapshot.execution_ticker == "EXM"
    assert snapshot.market_features

    store.put_many(
        [
            _bar(stock, "EXM", date(2026, 8, 18), "102"),
            _bar(benchmark, "SPY", date(2026, 8, 18), "502"),
        ]
    )
    with pytest.raises(ValueError, match="skips a known candidate trading session"):
        builder.build_for_execution_date(event, date(2026, 8, 19))


class _ParitySnapshotBuilder:
    def __init__(
        self,
        current_event_id: str,
        *,
        current_historical_available: bool,
    ) -> None:
        self.current_event_id = current_event_id
        self.current_historical_available = current_historical_available

    def _snapshot(self, event: Event, execution_date: date) -> CandidateSnapshot:
        return CandidateSnapshot(
            event_id=event.event_id,
            company_id=event.company_id,
            security_id="security_example",
            decision_time=event.public_time,
            decision_market_date=event.public_time.date(),
            execution_date=execution_date,
            first_tradable_time=datetime.combine(
                execution_date,
                datetime.min.time(),
                tzinfo=timezone.utc,
            ),
            execution_ticker="EXM",
            market_features={
                "market.return_5d": 0.03,
                "market.return_20d": 0.04,
                "market.return_60d": 0.05,
                "market.benchmark_return_5d": 0.01,
                "market.benchmark_return_20d": 0.02,
                "market.benchmark_return_60d": 0.03,
                "market.relative_return_5d": 0.02,
                "market.relative_return_20d": 0.02,
                "market.relative_return_60d": 0.02,
                "market.volatility_5d": 0.01,
                "market.volatility_20d": 0.02,
                "market.volatility_60d": 0.03,
                "market.volume_zscore_20d": 0.5,
            },
        )

    def build(self, event: Event) -> CandidateSnapshot:
        if event.event_id == self.current_event_id:
            if not self.current_historical_available:
                raise ValueError("future bar intentionally unavailable")
            return self._snapshot(event, date(2026, 8, 18))
        return self._snapshot(event, date(2026, 8, 4))

    def build_for_execution_date(
        self,
        event: Event,
        execution_date: date,
    ) -> CandidateSnapshot:
        return self._snapshot(event, execution_date)

    def label(self, snapshot: CandidateSnapshot) -> LabeledCandidate:
        return LabeledCandidate(
            snapshot=snapshot,
            labels=(
                ForwardLabel(
                    horizon=20,
                    start_date=snapshot.execution_date,
                    end_date=date(2026, 9, 15),
                    stock_return=0.05,
                    benchmark_return=0.01,
                    alpha=0.04,
                    max_favorable_excursion=0.08,
                    max_adverse_excursion=-0.02,
                ),
            ),
        )


def test_current_candidate_preserves_training_features_and_prior_opportunity_history() -> None:
    prior = _event(
        "prior",
        datetime(2026, 8, 3, 18, 0, tzinfo=timezone.utc),
        value="500",
    )
    current = _event(
        "current",
        datetime(2026, 8, 17, 20, 0, tzinfo=timezone.utc),
        value="2000",
    )
    future = _event(
        "future",
        datetime(2026, 8, 19, 18, 0, tzinfo=timezone.utc),
        value="999999",
    )
    training_builder = _ParitySnapshotBuilder(
        current.event_id,
        current_historical_available=True,
    )
    live_builder = _ParitySnapshotBuilder(
        current.event_id,
        current_historical_available=False,
    )
    as_of = datetime(2026, 8, 17, 21, 0, tzinfo=timezone.utc)

    training_row = TrainingDatasetBuilder(training_builder).build(
        (current,),
        all_events=(prior, current),
    )[0]
    assembly = PitCandidateAssembler(live_builder).assemble(
        (current,),
        all_events=(prior, current, future),
        as_of=as_of,
        execution_date=date(2026, 8, 18),
    )

    assert assembly.candidate_count == 1
    assert assembly.context_opportunity_count == 2
    candidate = assembly.candidates[0]
    assert candidate.candidate_id == "opportunity:cmp_example:2026-08-18"
    assert candidate.execution_date == date(2026, 8, 18)

    # Every feature produced by the training dataset for the same current trigger
    # has exactly the same value at runtime. Runtime then adds only label-free
    # opportunity-history/system-context features.
    for name, value in training_row.features.items():
        assert candidate.features[name] == value

    assert candidate.features["opportunity_history.has_previous"] == 1.0
    assert candidate.features["opportunity_history.count_30d"] == 1.0
    assert candidate.features["opportunity_history.previous_trigger_count"] == 1.0
    assert candidate.features["system.cross_section.opportunity_count"] == 1.0
    assert candidate.features["insider.buy_value_30d"] == 2500.0
    # The event after as_of must not affect any current feature.
    assert candidate.features["insider.buy_value_30d"] != 1002499.0
