import argparse
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

from stock_trading.market import (
    DuckDbMarketStore,
    IssuerObservation,
    MarketBackfillService,
    TiingoClient,
    TiingoNormalizer,
)
from stock_trading.sec import QuarterlyArchiveParser, SecClient
from stock_trading.storage import DuckDbEventStore, FileRawStore


BENCHMARK_SPY_COMPANY_ID = "benchmark_spy"


@dataclass(frozen=True, slots=True)
class SecMarketPopulationConfig:
    data_root: Path
    start_year: int = 2012
    start_quarter: int = 1
    end_year: int | None = None
    end_quarter: int | None = None
    market_start: date | None = None
    market_end: date | None = None
    max_unique_tickers: int | None = None


@dataclass(frozen=True, slots=True)
class SecMarketPopulationResult:
    quarters_downloaded: int
    insider_events: int
    issuer_observations: int
    resolved_companies: int
    unresolved_observations: int
    market_bars: int
    failed_metadata_requests: int
    failed_price_series: int
    benchmark_bars: int
    events_db: Path
    market_db: Path
    benchmark_company_id: str


def populate_sec_and_market(
    config: SecMarketPopulationConfig,
    *,
    sec_client: SecClient,
    tiingo_client: TiingoClient,
) -> SecMarketPopulationResult:
    if config.start_year < 2006:
        raise ValueError("SEC quarterly insider history starts in 2006")
    if config.start_quarter not in {1, 2, 3, 4}:
        raise ValueError("start_quarter must be 1..4")
    if config.max_unique_tickers is not None and config.max_unique_tickers <= 0:
        raise ValueError("max_unique_tickers must be > 0")

    today = date.today()
    default_end_year, default_end_quarter = latest_completed_quarter(today)
    end_year = config.end_year if config.end_year is not None else default_end_year
    end_quarter = config.end_quarter if config.end_quarter is not None else default_end_quarter
    if end_quarter not in {1, 2, 3, 4}:
        raise ValueError("end_quarter must be 1..4")
    if (end_year, end_quarter) < (config.start_year, config.start_quarter):
        raise ValueError("end quarter must not precede start quarter")

    data_root = config.data_root
    raw_store = FileRawStore(data_root / "raw")
    events_db = data_root / "normalized" / "events.duckdb"
    market_db = data_root / "normalized" / "market.duckdb"
    event_store = DuckDbEventStore(events_db)
    market_store = DuckDbMarketStore(market_db)
    parser = QuarterlyArchiveParser()

    observations_by_key: dict[tuple[str, str, str], IssuerObservation] = {}
    quarters_downloaded = 0
    insider_events = 0

    for year, quarter in quarter_range(
        config.start_year,
        config.start_quarter,
        end_year,
        end_quarter,
    ):
        raw = sec_client.fetch_quarterly_archive(year, quarter)
        raw_store.put(raw)
        transactions = parser.parse(
            raw.content if isinstance(raw.content, bytes) else raw.content.encode("utf-8")
        )
        events = parser.to_events(raw, ingested_at=raw.fetched_at)
        event_store.put_many(list(events))
        quarters_downloaded += 1
        insider_events += len(events)

        for transaction in transactions:
            if not transaction.issuer_symbol:
                continue
            observation = IssuerObservation(
                sec_cik=transaction.issuer_cik,
                issuer_name=transaction.issuer_name,
                ticker=transaction.issuer_symbol,
                observed_date=transaction.filing_date,
            )
            key = (
                observation.sec_cik,
                observation.ticker.strip().upper().replace(".", "-"),
                observation.issuer_name.strip().upper(),
            )
            existing = observations_by_key.get(key)
            if existing is None or observation.observed_date < existing.observed_date:
                observations_by_key[key] = observation

    observations = tuple(observations_by_key.values())
    if config.max_unique_tickers is not None:
        allowed_tickers = set(
            sorted({observation.ticker for observation in observations})[
                : config.max_unique_tickers
            ]
        )
        observations = tuple(
            observation for observation in observations if observation.ticker in allowed_tickers
        )

    market_start = config.market_start or date(config.start_year, 1, 1) - timedelta(days=400)
    market_end = config.market_end or today
    if market_end < market_start:
        raise ValueError("market_end must not precede market_start")

    market_result = MarketBackfillService(
        client=tiingo_client,
        raw_store=raw_store,
        market_store=market_store,
    ).backfill(
        observations,
        start=market_start,
        end=market_end,
    )

    benchmark_raw = tiingo_client.fetch_prices("SPY", market_start, market_end)
    raw_store.put(benchmark_raw)
    benchmark_bars = TiingoNormalizer().parse_prices(
        benchmark_raw,
        company_id=BENCHMARK_SPY_COMPANY_ID,
        ticker="SPY",
    )
    market_store.put_many(benchmark_bars)

    unresolved_path = data_root / "manifests" / "unresolved_tiingo.jsonl"
    unresolved_path.parent.mkdir(parents=True, exist_ok=True)
    with unresolved_path.open("w", encoding="utf-8") as handle:
        for resolution in market_result.resolutions:
            if resolution.resolved:
                continue
            handle.write(
                json.dumps(
                    {
                        "reason": resolution.reason,
                        "observation": {
                            "sec_cik": resolution.observation.sec_cik,
                            "issuer_name": resolution.observation.issuer_name,
                            "ticker": resolution.observation.ticker,
                            "observed_date": resolution.observation.observed_date.isoformat(),
                        },
                    },
                    sort_keys=True,
                )
            )
            handle.write("\n")

    reason_counts = Counter(
        resolution.reason
        for resolution in market_result.resolutions
        if not resolution.resolved
    )
    result = SecMarketPopulationResult(
        quarters_downloaded=quarters_downloaded,
        insider_events=insider_events,
        issuer_observations=len(observations),
        resolved_companies=market_result.resolved_companies,
        unresolved_observations=market_result.unresolved_observations,
        market_bars=market_result.normalized_bars,
        failed_metadata_requests=market_result.failed_metadata_requests,
        failed_price_series=market_result.failed_price_series,
        benchmark_bars=len(benchmark_bars),
        events_db=events_db,
        market_db=market_db,
        benchmark_company_id=BENCHMARK_SPY_COMPANY_ID,
    )
    manifest = {
        **_jsonable(asdict(result)),
        "start_quarter": f"{config.start_year}Q{config.start_quarter}",
        "end_quarter": f"{end_year}Q{end_quarter}",
        "market_start": market_start.isoformat(),
        "market_end": market_end.isoformat(),
        "unresolved_reason_counts": dict(sorted(reason_counts.items())),
    }
    (data_root / "manifests" / "sec_market.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def latest_completed_quarter(day: date) -> tuple[int, int]:
    quarter = (day.month - 1) // 3 + 1
    if quarter == 1:
        return day.year - 1, 4
    return day.year, quarter - 1


def quarter_range(
    start_year: int,
    start_quarter: int,
    end_year: int,
    end_quarter: int,
) -> tuple[tuple[int, int], ...]:
    values: list[tuple[int, int]] = []
    year, quarter = start_year, start_quarter
    while (year, quarter) <= (end_year, end_quarter):
        values.append((year, quarter))
        quarter += 1
        if quarter == 5:
            quarter = 1
            year += 1
    return tuple(values)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (Path, date)):
        return str(value)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Populate SEC insider events and verified Tiingo EOD history."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--start-quarter", type=int, default=1)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--end-quarter", type=int)
    parser.add_argument("--market-start", type=date.fromisoformat)
    parser.add_argument("--market-end", type=date.fromisoformat)
    parser.add_argument("--max-unique-tickers", type=int)
    parser.add_argument(
        "--sec-user-agent",
        required=True,
        help="SEC-required application/contact identity, e.g. 'Stock-traiding name@email'.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    token = os.environ.get("TIINGO_API_TOKEN", "").strip()
    if not token:
        raise SystemExit("TIINGO_API_TOKEN environment variable is required")

    config = SecMarketPopulationConfig(
        data_root=args.data_root,
        start_year=args.start_year,
        start_quarter=args.start_quarter,
        end_year=args.end_year,
        end_quarter=args.end_quarter,
        market_start=args.market_start,
        market_end=args.market_end,
        max_unique_tickers=args.max_unique_tickers,
    )
    with SecClient(args.sec_user_agent) as sec_client, TiingoClient(token) as tiingo_client:
        result = populate_sec_and_market(
            config,
            sec_client=sec_client,
            tiingo_client=tiingo_client,
        )
    print(json.dumps(_jsonable(asdict(result)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
