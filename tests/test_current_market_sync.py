from __future__ import annotations

import json
from datetime import date, datetime, timezone

from stock_trading.core import RawRecord, Source, content_sha256
from stock_trading.entities import company_id_from_sec_cik
from stock_trading.live.current_market import sync_pending_current_market
from stock_trading.live.event_intake import PendingTrigger
from stock_trading.live.session_calendar import XnysExecutionSessionResolver
from stock_trading.market import DuckDbMarketStore
from stock_trading.sec import Form4XmlParser
from stock_trading.storage import FileRawStore


UTC = timezone.utc


def _raw(source, source_record_id, content_type, content, *, fetched_at=None):
    return RawRecord(
        source=source,
        source_record_id=source_record_id,
        fetched_at=fetched_at or datetime(2026, 8, 18, 12, 0, tzinfo=UTC),
        content_type=content_type,
        content=content,
        sha256=content_sha256(content),
    )


def _price_payload(day: str, close: float) -> bytes:
    return json.dumps(
        [
            {
                "date": f"{day}T00:00:00.000Z",
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1_000_000,
                "adjOpen": close,
                "adjHigh": close,
                "adjLow": close,
                "adjClose": close,
                "adjVolume": 1_000_000,
                "divCash": 0,
                "splitFactor": 1,
            }
        ]
    ).encode("utf-8")


class _Tiingo:
    def __init__(self):
        self.metadata_tickers = []
        self.price_tickers = []

    def fetch_metadata(self, ticker):
        self.metadata_tickers.append(ticker)
        names = {"FAST": "Fastenal Company"}
        content = json.dumps(
            {
                "ticker": ticker,
                "name": names[ticker],
                "exchangeCode": "NASDAQ",
                "startDate": "1987-08-20",
                "endDate": "2026-08-17",
            }
        ).encode("utf-8")
        return _raw(Source.TIINGO, f"metadata:{ticker}", "application/json", content)

    def fetch_prices(self, ticker, start, end):
        del start, end
        self.price_tickers.append(ticker)
        close = 45.0 if ticker == "FAST" else 650.0
        return _raw(
            Source.TIINGO,
            f"prices:{ticker}:test",
            "application/json",
            _price_payload("2026-08-17", close),
        )


def test_form4_parser_exposes_issuer_identity() -> None:
    content = b"""<ownershipDocument>
      <documentType>4</documentType>
      <issuer>
        <issuerCik>815556</issuerCik>
        <issuerName>FASTENAL CO</issuerName>
        <issuerTradingSymbol>FAST</issuerTradingSymbol>
      </issuer>
    </ownershipDocument>"""
    identity = Form4XmlParser().issuer_identity(
        _raw(Source.SEC_EDGAR, "0001454708-26-000010", "application/xml", content)
    )
    assert identity.cik == "0000815556"
    assert identity.name == "FASTENAL CO"
    assert identity.ticker == "FAST"


def test_targeted_current_market_sync_refreshes_mapping_and_completed_bars(tmp_path) -> None:
    data_root = tmp_path / "data"
    raw_store = FileRawStore(data_root / "raw")
    accession = "0001454708-26-000010"
    form4 = b"""<ownershipDocument>
      <documentType>4</documentType>
      <issuer>
        <issuerCik>0000815556</issuerCik>
        <issuerName>FASTENAL CO</issuerName>
        <issuerTradingSymbol>FAST</issuerTradingSymbol>
      </issuer>
    </ownershipDocument>"""
    raw_store.put(_raw(Source.SEC_EDGAR, accession, "application/xml", form4))

    company_id = company_id_from_sec_cik("0000815556")
    pending = (
        PendingTrigger(
            event_id="evt-current",
            company_id=company_id,
            public_time=datetime(2026, 8, 17, 15, 16, tzinfo=UTC),
            cik="0000815556",
            accession_number=accession,
        ),
    )
    market_store = DuckDbMarketStore(data_root / "normalized" / "market.duckdb")
    tiingo = _Tiingo()

    result = sync_pending_current_market(
        pending,
        data_root=data_root,
        market_store=market_store,
        benchmark_security_id="benchmark_spy",
        tiingo_client=tiingo,  # type: ignore[arg-type]
        sync_end_date=date(2026, 8, 17),
    )

    assert result.ready is True
    assert result.company_count == 1
    assert result.resolved_companies == 1
    assert result.unresolved_companies == 0
    assert result.tickers == ("FAST",)
    assert tiingo.metadata_tickers == ["FAST"]
    assert set(tiingo.price_tickers) == {"FAST", "SPY"}
    security_id = market_store.security_for_company(company_id, date(2026, 8, 17))
    assert security_id is not None
    assert market_store.bar_on(security_id, date(2026, 8, 17)) is not None
    assert market_store.bar_on("benchmark_spy", date(2026, 8, 17)) is not None


def test_last_completed_session_does_not_use_preopen_today() -> None:
    resolver = XnysExecutionSessionResolver()
    assert resolver.last_completed_session(
        datetime(2026, 8, 18, 12, 0, tzinfo=UTC)
    ) == date(2026, 8, 17)
    assert resolver.last_completed_session(
        datetime(2026, 8, 18, 21, 0, tzinfo=UTC)
    ) == date(2026, 8, 18)
