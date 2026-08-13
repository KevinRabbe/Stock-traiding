import argparse
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter
from typing import Callable

from stock_trading.core import Source
from stock_trading.entities import company_id_from_sec_cik
from stock_trading.market import (
    DuckDbMarketStore,
    IssuerObservation,
    MarketBackfillService,
    TiingoClient,
    TiingoNormalizer,
    normalize_tiingo_ticker,
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
    tickers: tuple[str, ...] | None = None
    sec_only: bool = False
    refresh_sec_raw: bool = False


@dataclass(frozen=True, slots=True)
class SecMarketPopulationResult:
    quarters_downloaded: int
    quarters_reused: int
    insider_events: int
    temporal_anomalies_skipped: int
    issuer_observations: int
    unique_tickers: int
    sec_companies: int
    resolved_companies: int
    unresolved_observations: int
    market_bars: int
    downloaded_price_series: int
    failed_metadata_requests: int
    failed_price_series: int
    reused_metadata_responses: int
    reused_price_responses: int
    skipped_complete_price_series: int
    benchmark_bars: int
    events_db: Path
    market_db: Path
    benchmark_company_id: str


def populate_sec_and_market(
    config: SecMarketPopulationConfig,
    *,
    sec_client: SecClient,
    tiingo_client: TiingoClient | None,
    progress: Callable[[str], None] | None = None,
) -> SecMarketPopulationResult:
    if config.start_year < 2006:
        raise ValueError("SEC quarterly insider history starts in 2006")
    if config.start_quarter not in {1, 2, 3, 4}:
        raise ValueError("start_quarter must be 1..4")
    if config.max_unique_tickers is not None and config.max_unique_tickers <= 0:
        raise ValueError("max_unique_tickers must be > 0")
    if config.max_unique_tickers is not None and config.tickers:
        raise ValueError("tickers and max_unique_tickers are mutually exclusive")

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
    parser = QuarterlyArchiveParser()

    observations_by_key: dict[tuple[str, str, str], IssuerObservation] = {}
    company_manifest: dict[str, dict[str, object]] = {}
    quarters_downloaded = 0
    quarters_reused = 0
    insider_events = 0
    temporal_anomalies_skipped = 0

    quarters = quarter_range(
        config.start_year,
        config.start_quarter,
        end_year,
        end_quarter,
    )
    total_quarters = len(quarters)

    for position, (year, quarter) in enumerate(quarters, start=1):
        quarter_started = perf_counter()
        source_record_id = f"{year}Q{quarter}"
        raw = None
        if not config.refresh_sec_raw:
            raw = raw_store.latest(Source.SEC_QUARTERLY, source_record_id)
        if raw is None:
            raw = sec_client.fetch_quarterly_archive(year, quarter)
            raw_store.put(raw)
            quarters_downloaded += 1
            raw_mode = "downloaded"
        else:
            quarters_reused += 1
            raw_mode = "reused"

        transactions = parser.parse(
            raw.content if isinstance(raw.content, bytes) else raw.content.encode("utf-8")
        )
        quarter_anomalies = sum(
            1 for transaction in transactions if parser.has_temporal_anomaly(transaction)
        )
        temporal_anomalies_skipped += quarter_anomalies
        events = parser.to_events(
            raw,
            ingested_at=raw.fetched_at,
            transactions=transactions,
        )
        event_store.put_many(list(events))
        insider_events += len(events)

        for transaction in transactions:
            company_id = company_id_from_sec_cik(transaction.issuer_cik)
            company = company_manifest.setdefault(
                company_id,
                {
                    "company_id": company_id,
                    "sec_cik": transaction.issuer_cik,
                    "issuer_names": set(),
                    "tickers": set(),
                },
            )
            company["issuer_names"].add(transaction.issuer_name)
            if transaction.issuer_symbol:
                company["tickers"].add(transaction.issuer_symbol)

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

        if progress is not None:
            elapsed = perf_counter() - quarter_started
            progress(
                f"[{position:02d}/{total_quarters:02d}] {year}Q{quarter} | {raw_mode} | "
                f"transactions={len(transactions):,} events={len(events):,} "
                f"anomalies={quarter_anomalies:,} | {elapsed:.1f}s"
            )

    manifests_dir = data_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    with (manifests_dir / "sec_companies.jsonl").open("w", encoding="utf-8") as handle:
        for company_id in sorted(company_manifest):
            company = company_manifest[company_id]
            handle.write(
                json.dumps(
                    {
                        "company_id": company["company_id"],
                        "sec_cik": company["sec_cik"],
                        "issuer_names": sorted(company["issuer_names"]),
                        "tickers": sorted(company["tickers"]),
                    },
                    sort_keys=True,
                )
            )
            handle.write("\n")

    all_observations = tuple(observations_by_key.values())
    total_unique_tickers = len({_normalized_ticker(item.ticker) for item in all_observations})
    (
        observations,
        selected_unique_tickers,
        requested_tickers,
        missing_requested_tickers,
    ) = _select_observations(
        all_observations,
        max_unique_tickers=config.max_unique_tickers,
        requested_tickers=config.tickers,
    )

    market_start = config.market_start or date(config.start_year, 1, 1) - timedelta(days=400)
    market_end = config.market_end or today
    if market_end < market_start:
        raise ValueError("market_end must not precede market_start")

    if config.sec_only:
        result = SecMarketPopulationResult(
            quarters_downloaded=quarters_downloaded,
            quarters_reused=quarters_reused,
            insider_events=insider_events,
            temporal_anomalies_skipped=temporal_anomalies_skipped,
            issuer_observations=len(observations),
            unique_tickers=selected_unique_tickers,
            sec_companies=len(company_manifest),
            resolved_companies=0,
            unresolved_observations=0,
            market_bars=0,
            downloaded_price_series=0,
            failed_metadata_requests=0,
            failed_price_series=0,
            reused_metadata_responses=0,
            reused_price_responses=0,
            skipped_complete_price_series=0,
            benchmark_bars=0,
            events_db=events_db,
            market_db=market_db,
            benchmark_company_id=BENCHMARK_SPY_COMPANY_ID,
        )
        manifest = {
            **_jsonable(asdict(result)),
            "mode": "sec_only",
            "start_quarter": f"{config.start_year}Q{config.start_quarter}",
            "end_quarter": f"{end_year}Q{end_quarter}",
            "total_unique_tickers_before_limit": total_unique_tickers,
            "requested_tickers": list(requested_tickers),
            "missing_requested_tickers": list(missing_requested_tickers),
            "estimated_tiingo_requests": estimate_tiingo_requests(selected_unique_tickers),
        }
        (manifests_dir / "sec_universe.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return result

    if tiingo_client is None:
        raise ValueError("tiingo_client is required unless sec_only=True")

    market_store = DuckDbMarketStore(market_db)
    market_result = MarketBackfillService(
        client=tiingo_client,
        raw_store=raw_store,
        market_store=market_store,
    ).backfill(
        observations,
        start=market_start,
        end=market_end,
    )

    benchmark_record_id = f"prices:SPY:{market_start.isoformat()}:{market_end.isoformat()}"
    benchmark_raw = raw_store.latest(Source.TIINGO, benchmark_record_id)
    if benchmark_raw is None:
        benchmark_raw = tiingo_client.fetch_prices("SPY", market_start, market_end)
        raw_store.put(benchmark_raw)
    benchmark_bars = TiingoNormalizer().parse_prices(
        benchmark_raw,
        company_id=BENCHMARK_SPY_COMPANY_ID,
        ticker="SPY",
    )
    market_store.put_many(benchmark_bars)

    unresolved_path = manifests_dir / "unresolved_tiingo.jsonl"
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
        quarters_reused=quarters_reused,
        insider_events=insider_events,
        temporal_anomalies_skipped=temporal_anomalies_skipped,
        issuer_observations=len(observations),
        unique_tickers=selected_unique_tickers,
        sec_companies=len(company_manifest),
        resolved_companies=market_result.resolved_companies,
        unresolved_observations=market_result.unresolved_observations,
        market_bars=market_result.normalized_bars,
        downloaded_price_series=market_result.downloaded_price_series,
        failed_metadata_requests=market_result.failed_metadata_requests,
        failed_price_series=market_result.failed_price_series,
        reused_metadata_responses=market_result.reused_metadata_responses,
        reused_price_responses=market_result.reused_price_responses,
        skipped_complete_price_series=market_result.skipped_complete_price_series,
        benchmark_bars=len(benchmark_bars),
        events_db=events_db,
        market_db=market_db,
        benchmark_company_id=BENCHMARK_SPY_COMPANY_ID,
    )
    manifest = {
        **_jsonable(asdict(result)),
        "mode": "sec_market",
        "start_quarter": f"{config.start_year}Q{config.start_quarter}",
        "end_quarter": f"{end_year}Q{end_quarter}",
        "market_start": market_start.isoformat(),
        "market_end": market_end.isoformat(),
        "total_unique_tickers_before_limit": total_unique_tickers,
        "requested_tickers": list(requested_tickers),
        "missing_requested_tickers": list(missing_requested_tickers),
        "unresolved_reason_counts": dict(sorted(reason_counts.items())),
    }
    (manifests_dir / "sec_market.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def _select_observations(
    observations: tuple[IssuerObservation, ...],
    *,
    max_unique_tickers: int | None,
    requested_tickers: tuple[str, ...] | None,
) -> tuple[tuple[IssuerObservation, ...], int, tuple[str, ...], tuple[str, ...]]:
    if max_unique_tickers is not None and requested_tickers:
        raise ValueError("tickers and max_unique_tickers are mutually exclusive")

    if requested_tickers:
        requested = tuple(
            dict.fromkeys(normalize_tiingo_ticker(ticker) for ticker in requested_tickers)
        )
        requested_set = set(requested)
        selected: list[IssuerObservation] = []
        present: set[str] = set()
        for observation in observations:
            try:
                ticker = normalize_tiingo_ticker(observation.ticker)
            except ValueError:
                continue
            if ticker in requested_set:
                selected.append(observation)
                present.add(ticker)
        missing = tuple(ticker for ticker in requested if ticker not in present)
        return tuple(selected), len(present), requested, missing

    selected = observations
    if max_unique_tickers is not None:
        allowed_tickers = set(
            sorted({_normalized_ticker(observation.ticker) for observation in observations})[
                :max_unique_tickers
            ]
        )
        selected = tuple(
            observation
            for observation in observations
            if _normalized_ticker(observation.ticker) in allowed_tickers
        )

    selected_unique_tickers = len({_normalized_ticker(item.ticker) for item in selected})
    return selected, selected_unique_tickers, (), ()


def estimate_tiingo_requests(unique_tickers: int) -> dict[str, int]:
    """Conservative lower-bound request estimate for a cold market backfill."""
    if unique_tickers < 0:
        raise ValueError("unique_tickers must be >= 0")
    metadata = unique_tickers
    price_series = unique_tickers
    benchmark = 1
    return {
        "metadata": metadata,
        "price_series": price_series,
        "benchmark": benchmark,
        "minimum_total": metadata + price_series + benchmark,
    }


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


def _normalized_ticker(value: str) -> str:
    return value.strip().upper().replace(".", "-")


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
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--max-unique-tickers", type=int)
    selection.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        help="Populate only these explicit SEC-observed tickers, e.g. AAPL MSFT NVDA.",
    )
    parser.add_argument(
        "--sec-only",
        action="store_true",
        help="Populate SEC events/company universe only; do not require or call Tiingo.",
    )
    parser.add_argument(
        "--refresh-sec-raw",
        action="store_true",
        help="Refetch SEC quarterly ZIPs even when immutable raw snapshots are already cached.",
    )
    parser.add_argument(
        "--sec-user-agent",
        required=True,
        help="SEC-required application/contact identity, e.g. 'Stock-traiding name@email'.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = SecMarketPopulationConfig(
        data_root=args.data_root,
        start_year=args.start_year,
        start_quarter=args.start_quarter,
        end_year=args.end_year,
        end_quarter=args.end_quarter,
        market_start=args.market_start,
        market_end=args.market_end,
        max_unique_tickers=args.max_unique_tickers,
        tickers=tuple(args.tickers) if args.tickers else None,
        sec_only=args.sec_only,
        refresh_sec_raw=args.refresh_sec_raw,
    )

    if args.sec_only:
        with SecClient(args.sec_user_agent) as sec_client:
            result = populate_sec_and_market(
                config,
                sec_client=sec_client,
                tiingo_client=None,
                progress=print,
            )
    else:
        token = os.environ.get("TIINGO_API_TOKEN", "").strip()
        if not token:
            raise SystemExit(
                "TIINGO_API_TOKEN environment variable is required unless --sec-only is used"
            )
        with SecClient(args.sec_user_agent) as sec_client, TiingoClient(token) as tiingo_client:
            result = populate_sec_and_market(
                config,
                sec_client=sec_client,
                tiingo_client=tiingo_client,
                progress=print,
            )
    print(json.dumps(_jsonable(asdict(result)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
