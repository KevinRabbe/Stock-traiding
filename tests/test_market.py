import json
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from stock_trading.core import (
    Event,
    EventType,
    InsiderTransactionPayload,
    RawRecord,
    Source,
    TradeDirection,
    content_sha256,
    deterministic_event_id,
)
from stock_trading.entities import company_id_from_sec_cik
from stock_trading.market import (
    CandidateSnapshotBuilder,
    ConservativeTiingoResolver,
    DuckDbMarketStore,
    IssuerObservation,
    MarketBar,
    SecurityMapping,
    SecurityRegistry,
    TiingoNormalizer,
    build_forward_label,
    conservative_first_tradable_time,
    normalize_tiingo_ticker,
)


def _raw_json(record_id: str, payload) -> RawRecord:
    content = json.dumps(payload)
    return RawRecord(
        source=Source.TIINGO,
        source_record_id=record_id,
        fetched_at=datetime(2026, 8, 11, 20, 0, tzinfo=timezone.utc),
        content_type="application/json",
        content=content,
        sha256=content_sha256(content),
    )


def _bar(
    company_id: str,
    ticker: str,
    day: date,
    open_: str,
    close: str,
    high: str,
    low: str,
    *,
    volume: str = "1000000",
) -> MarketBar:
    return MarketBar(
        company_id=company_id,
        ticker=ticker,
        date=day,
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal(volume),
        adj_open=Decimal(open_),
        adj_high=Decimal(high),
        adj_low=Decimal(low),
        adj_close=Decimal(close),
        adj_volume=Decimal(volume),
    )


def _insider_event(company_id: str, public_time: datetime) -> Event:
    source_record_id = "0000000001-26-000001:NONDERIV_TRANS:0"
    return Event(
        event_id=deterministic_event_id(
            Source.SEC_EDGAR,
            source_record_id,
            EventType.INSIDER_TRANSACTION,
        ),
        event_type=EventType.INSIDER_TRANSACTION,
        company_id=company_id,
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
            price=Decimal("10"),
            value=Decimal("1000"),
            intent_class="DISCRETIONARY_BUY",
        ),
        semantic=None,
        raw_artifact_id="raw_0123456789abcdef0123456789abcdef",
        ingested_at=public_time,
    )


def test_tiingo_normalizer_preserves_raw_adjusted_and_corporate_actions() -> None:
    company_id = company_id_from_sec_cik("12345")
    payload = [
        {
            "date": "2026-08-10T00:00:00.000Z",
            "open": 100,
            "high": 110,
            "low": 99,
            "close": 108,
            "volume": 1000000,
            "adjOpen": 50,
            "adjHigh": 55,
            "adjLow": 49.5,
            "adjClose": 54,
            "adjVolume": 2000000,
            "divCash": 0.25,
            "splitFactor": 2.0,
        }
    ]
    bars = TiingoNormalizer().parse_prices(
        _raw_json("prices:EXM", payload),
        company_id=company_id,
        ticker="EXM",
    )

    assert len(bars) == 1
    assert bars[0].close == Decimal("108")
    assert bars[0].adj_close == Decimal("54")
    assert bars[0].dividend_cash == Decimal("0.25")
    assert bars[0].split_factor == Decimal("2.0")


def test_tiingo_metadata_requires_explicit_entity_resolution() -> None:
    metadata = TiingoNormalizer().parse_metadata(
        _raw_json(
            "metadata:EXM",
            {
                "ticker": "EXM",
                "name": "Example Corp",
                "exchangeCode": "NASDAQ",
                "startDate": "2012-01-03",
                "endDate": "2026-08-10",
            },
        )
    )
    observation = IssuerObservation(
        sec_cik="12345",
        issuer_name="Example Corporation",
        ticker="EXM",
        observed_date=date(2020, 1, 2),
    )
    resolution = ConservativeTiingoResolver().resolve(
        observation,
        tiingo_ticker=metadata.ticker,
        tiingo_name=metadata.name,
        tiingo_start=metadata.start_date,
        tiingo_end=metadata.end_date,
        exchange_code=metadata.exchange_code,
    )

    assert resolution.resolved
    assert resolution.mapping is not None
    assert resolution.mapping.company_id == company_id_from_sec_cik("12345")
    assert resolution.mapping.contains(date(2020, 1, 2))


def test_tiingo_resolution_rejects_recycled_symbol_history() -> None:
    observation = IssuerObservation(
        sec_cik="12345",
        issuer_name="Old Example Corp",
        ticker="ABC",
        observed_date=date(2015, 1, 2),
    )
    resolution = ConservativeTiingoResolver().resolve(
        observation,
        tiingo_ticker="ABC",
        tiingo_name="Old Example Corp",
        tiingo_start=date(2021, 1, 1),
        tiingo_end=None,
    )

    assert not resolution.resolved
    assert resolution.reason == "observation_predates_tiingo_history"


def test_security_registry_blocks_overlapping_ticker_reuse() -> None:
    registry = SecurityRegistry()
    registry.add(
        SecurityMapping(
            company_id="cmp_a",
            ticker="ABC",
            valid_from=date(2010, 1, 1),
            valid_to=date(2020, 12, 31),
        )
    )
    registry.add(
        SecurityMapping(
            company_id="cmp_b",
            ticker="ABC",
            valid_from=date(2021, 1, 1),
            valid_to=None,
        )
    )

    assert registry.company_for_ticker("ABC", date(2015, 1, 1)) == "cmp_a"
    assert registry.company_for_ticker("ABC", date(2025, 1, 1)) == "cmp_b"

    with pytest.raises(ValueError, match="overlaps"):
        registry.add(
            SecurityMapping(
                company_id="cmp_c",
                ticker="ABC",
                valid_from=date(2020, 6, 1),
                valid_to=date(2021, 6, 1),
            )
        )


def test_forward_label_uses_adjusted_open_close_and_excursions() -> None:
    stock = [
        _bar("cmp_stock", "AAA", date(2026, 8, 10), "100", "105", "107", "98"),
        _bar("cmp_stock", "AAA", date(2026, 8, 11), "105", "110", "112", "103"),
    ]
    benchmark = [
        _bar("cmp_spy", "SPY", date(2026, 8, 10), "100", "101", "102", "99"),
        _bar("cmp_spy", "SPY", date(2026, 8, 11), "101", "102", "103", "100"),
    ]

    label = build_forward_label(stock, benchmark, horizon=2)

    assert label.stock_return == pytest.approx(0.10)
    assert label.benchmark_return == pytest.approx(0.02)
    assert label.alpha == pytest.approx(0.08)
    assert label.max_favorable_excursion == pytest.approx(0.12)
    assert label.max_adverse_excursion == pytest.approx(-0.02)


def test_conservative_execution_uses_next_actual_market_date() -> None:
    public_time = datetime(2026, 8, 7, 20, 15, tzinfo=timezone.utc)
    result = conservative_first_tradable_time(public_time, date(2026, 8, 10))
    assert result == datetime(2026, 8, 10, 13, 30, tzinfo=timezone.utc)


def test_tiingo_symbol_normalization_matches_documented_dash_style() -> None:
    assert normalize_tiingo_ticker("brk.b") == "BRK-B"


def test_market_store_returns_typed_actual_bars_and_excludes_decision_day(tmp_path) -> None:
    pytest.importorskip("duckdb")
    company_id = company_id_from_sec_cik("12345")
    bars = [
        _bar(company_id, "EXM", date(2026, 8, 6), "10", "10.5", "10.8", "9.9"),
        _bar(company_id, "EXM", date(2026, 8, 7), "10.6", "11", "11.2", "10.5"),
        _bar(company_id, "EXM", date(2026, 8, 10), "11.1", "11.4", "11.5", "11"),
    ]
    store = DuckDbMarketStore(tmp_path / "market.duckdb")
    store.put_many(bars)
    store.put_many(bars)

    next_bar = store.next_bar_after(company_id, date(2026, 8, 7))
    assert isinstance(next_bar, MarketBar)
    assert next_bar.date == date(2026, 8, 10)
    assert [bar.date for bar in store.bars_before(company_id, date(2026, 8, 7), 10)] == [
        date(2026, 8, 6)
    ]


def test_snapshot_builder_keeps_same_day_eod_data_out_of_features(tmp_path) -> None:
    pytest.importorskip("duckdb")
    stock_id = company_id_from_sec_cik("12345")
    benchmark_id = "cmp_spy"
    store = DuckDbMarketStore(tmp_path / "market.duckdb")

    stock_bars = [
        _bar(stock_id, "EXM", date(2026, 8, 5), "99", "100", "101", "98"),
        _bar(stock_id, "EXM", date(2026, 8, 6), "100", "101", "102", "99"),
        _bar(stock_id, "EXM", date(2026, 8, 7), "101", "150", "151", "100"),
        _bar(stock_id, "EXM", date(2026, 8, 10), "102", "104", "105", "101"),
        _bar(stock_id, "EXM", date(2026, 8, 11), "104", "106", "107", "103"),
    ]
    benchmark_bars = [
        _bar(benchmark_id, "SPY", date(2026, 8, 5), "100", "100", "101", "99"),
        _bar(benchmark_id, "SPY", date(2026, 8, 6), "100", "100", "101", "99"),
        _bar(benchmark_id, "SPY", date(2026, 8, 7), "100", "120", "121", "99"),
        _bar(benchmark_id, "SPY", date(2026, 8, 10), "100", "101", "102", "99"),
        _bar(benchmark_id, "SPY", date(2026, 8, 11), "101", "102", "103", "100"),
    ]
    store.put_many(stock_bars + benchmark_bars)

    public_time = datetime(2026, 8, 7, 14, 0, tzinfo=timezone.utc)
    event = _insider_event(stock_id, public_time)
    builder = CandidateSnapshotBuilder(
        store,
        benchmark_company_id=benchmark_id,
        feature_lookback_bars=260,
        label_horizons=(1, 2),
    )

    snapshot = builder.build(event)
    assert snapshot.execution_date == date(2026, 8, 10)
    assert snapshot.first_tradable_time == datetime(2026, 8, 10, 13, 30, tzinfo=timezone.utc)
    assert snapshot.market_features["market.return_1d"] == pytest.approx(0.01)

    labeled = builder.label(snapshot)
    labels = {label.horizon: label for label in labeled.labels}
    assert labels[1].stock_return == pytest.approx(104 / 102 - 1)
    assert labels[2].stock_return == pytest.approx(106 / 102 - 1)
    assert labels[2].benchmark_return == pytest.approx(102 / 100 - 1)
