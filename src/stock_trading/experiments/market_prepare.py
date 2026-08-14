from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

from stock_trading.core import Source
from stock_trading.market import (
    DuckDbMarketStore,
    IssuerObservation,
    MarketBackfillService,
    TiingoClient,
    TiingoNormalizer,
    normalize_tiingo_ticker,
)
from stock_trading.storage import FileRawStore

from .sec_snapshot import load_sec_universe_snapshot


BENCHMARK_SPY_SECURITY_ID = "benchmark_spy"


@dataclass(frozen=True, slots=True)
class MarketOnlyResult:
    snapshot_start_quarter: str
    snapshot_end_quarter: str
    snapshot_issuer_observations: int
    snapshot_unique_tickers: int
    snapshot_sec_companies: int
    issuer_observations: int
    unique_tickers: int
    resolved_companies: int
    resolved_securities: int
    unresolved_observations: int
    market_bars: int
    downloaded_price_series: int
    failed_metadata_requests: int
    failed_price_series: int
    reused_metadata_responses: int
    reused_price_responses: int
    skipped_complete_price_series: int
    benchmark_bars: int
    market_db: Path
    benchmark_security_id: str


def populate_market_from_snapshot(
    data_root: Path,
    *,
    tiingo_client: TiingoClient,
    market_start: date | None = None,
    market_end: date | None = None,
    max_unique_tickers: int | None = None,
    tickers: tuple[str, ...] | None = None,
) -> MarketOnlyResult:
    if max_unique_tickers is not None and max_unique_tickers <= 0:
        raise ValueError("max_unique_tickers must be > 0")
    if max_unique_tickers is not None and tickers:
        raise ValueError("tickers and max_unique_tickers are mutually exclusive")

    snapshot, all_observations = load_sec_universe_snapshot(data_root)
    observations, selected_unique_tickers, requested, missing = _select_observations(
        all_observations,
        max_unique_tickers=max_unique_tickers,
        requested_tickers=tickers,
    )

    default_market_start = date(snapshot.start_year, 1, 1) - timedelta(days=400)
    market_start = market_start or default_market_start
    market_end = market_end or date.today()
    if market_end < market_start:
        raise ValueError("market_end must not precede market_start")

    raw_store = FileRawStore(data_root / "raw")
    market_db = data_root / "normalized" / "market.duckdb"
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
        security_id=BENCHMARK_SPY_SECURITY_ID,
        ticker="SPY",
    )
    market_store.put_many(benchmark_bars)

    manifests_dir = data_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    unresolved_path = manifests_dir / "unresolved_tiingo.jsonl"
    with unresolved_path.open("w", encoding="utf-8", newline="\n") as handle:
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
    result = MarketOnlyResult(
        snapshot_start_quarter=f"{snapshot.start_year}Q{snapshot.start_quarter}",
        snapshot_end_quarter=f"{snapshot.end_year}Q{snapshot.end_quarter}",
        snapshot_issuer_observations=snapshot.issuer_observations,
        snapshot_unique_tickers=snapshot.unique_tickers,
        snapshot_sec_companies=snapshot.sec_companies,
        issuer_observations=len(observations),
        unique_tickers=selected_unique_tickers,
        resolved_companies=market_result.resolved_companies,
        resolved_securities=market_result.resolved_securities,
        unresolved_observations=market_result.unresolved_observations,
        market_bars=market_result.normalized_bars,
        downloaded_price_series=market_result.downloaded_price_series,
        failed_metadata_requests=market_result.failed_metadata_requests,
        failed_price_series=market_result.failed_price_series,
        reused_metadata_responses=market_result.reused_metadata_responses,
        reused_price_responses=market_result.reused_price_responses,
        skipped_complete_price_series=market_result.skipped_complete_price_series,
        benchmark_bars=len(benchmark_bars),
        market_db=market_db,
        benchmark_security_id=BENCHMARK_SPY_SECURITY_ID,
    )
    manifest = {
        **_jsonable(asdict(result)),
        "mode": "market_only",
        "market_start": market_start.isoformat(),
        "market_end": market_end.isoformat(),
        "requested_tickers": list(requested),
        "missing_requested_tickers": list(missing),
        "unresolved_reason_counts": dict(sorted(reason_counts.items())),
    }
    (manifests_dir / "market_only.json").write_text(
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
        description="Populate verified Tiingo EOD history from a cached SEC issuer snapshot."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
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
    return parser


def main() -> None:
    args = _parser().parse_args()
    token = os.environ.get("TIINGO_API_TOKEN", "").strip()
    if not token:
        raise SystemExit("TIINGO_API_TOKEN environment variable is required")

    with TiingoClient(token) as tiingo_client:
        result = populate_market_from_snapshot(
            args.data_root,
            tiingo_client=tiingo_client,
            market_start=args.market_start,
            market_end=args.market_end,
            max_unique_tickers=args.max_unique_tickers,
            tickers=tuple(args.tickers) if args.tickers else None,
        )
    print(json.dumps(_jsonable(asdict(result)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
