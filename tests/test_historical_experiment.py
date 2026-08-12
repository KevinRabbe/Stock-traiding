from datetime import date, datetime, timedelta, timezone
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
from stock_trading.experiments import HistoricalExperimentConfig, run_historical_experiment
from stock_trading.market import DuckDbMarketStore, MarketBar
from stock_trading.ml import LightGbmTrainingConfig
from stock_trading.storage import DuckDbEventStore


def _business_days(start: date, end: date):
    day = start
    while day <= end:
        if day.weekday() < 5:
            yield day
        day += timedelta(days=1)


def _series(company_id: str, ticker: str, daily_return: float) -> list[MarketBar]:
    price = 100.0
    bars: list[MarketBar] = []
    for day in _business_days(date(2021, 1, 4), date(2024, 8, 30)):
        open_price = price
        close_price = open_price * (1.0 + daily_return)
        high = max(open_price, close_price) * 1.005
        low = min(open_price, close_price) * 0.995
        bars.append(
            MarketBar(
                company_id=company_id,
                ticker=ticker,
                date=day,
                open=Decimal(str(open_price)),
                high=Decimal(str(high)),
                low=Decimal(str(low)),
                close=Decimal(str(close_price)),
                volume=Decimal("1000000"),
                adj_open=Decimal(str(open_price)),
                adj_high=Decimal(str(high)),
                adj_low=Decimal(str(low)),
                adj_close=Decimal(str(close_price)),
                adj_volume=Decimal("1000000"),
            )
        )
        price = close_price
    return bars


def _event(company_id: str, year: int, index: int, value: str) -> Event:
    public_time = datetime(year, 6, 1, 18, 0, tzinfo=timezone.utc)
    source_record_id = f"{company_id}:{year}:{index}"
    return Event(
        event_id=deterministic_event_id(
            Source.SEC_EDGAR,
            source_record_id,
            EventType.INSIDER_TRANSACTION,
        ),
        event_type=EventType.INSIDER_TRANSACTION,
        company_id=company_id,
        actor_id=f"owner-{company_id}",
        event_time=public_time - timedelta(days=2),
        public_time=public_time,
        first_tradable_time=None,
        source=Source.SEC_EDGAR,
        source_record_id=source_record_id,
        payload=InsiderTransactionPayload(
            source_transaction_code="P",
            direction=TradeDirection.BUY,
            shares=Decimal("100"),
            price=Decimal(value) / Decimal("100"),
            value=Decimal(value),
            insider_role="DIRECTOR",
            intent_class="DISCRETIONARY_BUY",
            is_10b5_1=False,
        ),
        semantic=None,
        raw_artifact_id=f"raw-{company_id}-{year}",
        ingested_at=public_time,
    )


def test_historical_experiment_runs_from_stores_to_report(tmp_path) -> None:
    pytest.importorskip("duckdb")
    pytest.importorskip("lightgbm")

    events_db = tmp_path / "events.duckdb"
    market_db = tmp_path / "market.duckdb"
    event_store = DuckDbEventStore(events_db)
    market_store = DuckDbMarketStore(market_db)

    benchmark_id = "benchmark_spy"
    market_store.put_many(_series(benchmark_id, "SPY", 0.0005))

    companies = (
        ("cmp_fast", "FAST", 0.0020, "5000"),
        ("cmp_mid", "MID", 0.0010, "3000"),
        ("cmp_flat", "FLAT", 0.0000, "2000"),
        ("cmp_down", "DOWN", -0.0010, "1000"),
    )
    for company_id, ticker, daily_return, _value in companies:
        market_store.put_many(_series(company_id, ticker, daily_return))

    events = []
    for year in (2022, 2023, 2024):
        for index, (company_id, _ticker, _return, value) in enumerate(companies):
            events.append(_event(company_id, year, index, value))
    event_store.put_many(events)

    output = tmp_path / "experiment"
    result = run_historical_experiment(
        HistoricalExperimentConfig(
            events_db=events_db,
            market_db=market_db,
            benchmark_company_id=benchmark_id,
            output_dir=output,
            first_test_year=2024,
            min_train_rows=4,
            min_validation_rows=4,
            min_test_rows=4,
            top_feature_count=5,
        ),
        training_config=LightGbmTrainingConfig(
            num_boost_round=12,
            early_stopping_rounds=3,
            learning_rate=0.2,
            num_leaves=3,
            min_data_in_leaf=1,
            feature_fraction=1.0,
            bagging_fraction=1.0,
            bagging_freq=0,
            seed=11,
        ),
    )

    assert result.event_count == 12
    assert result.trigger_count == 12
    assert result.training_row_count == 12
    assert result.tested_years == (2024,)
    assert (output / "training_rows.jsonl").exists()
    assert (output / "report.json").exists()
    assert (output / "models" / "2024" / "alpha.txt").exists()
    assert (output / "models" / "2024" / "metadata.json").exists()

    report = (output / "report.json").read_text(encoding="utf-8")
    assert '"schema_version": "historical-lightgbm-v1"' in report
    assert '"test_year": 2024' in report
