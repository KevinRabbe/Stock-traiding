from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

from stock_trading.core import Source
from stock_trading.entities import company_id_from_sec_cik
from stock_trading.local_secrets import load_tiingo_credentials
from stock_trading.market import (
    DuckDbMarketStore,
    IssuerObservation,
    MarketBackfillService,
    TiingoClient,
    TiingoNormalizer,
    normalize_tiingo_ticker,
)
from stock_trading.storage import FileRawStore

from .historical_universe import load_historical_universe_company_ids
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
    universe_manifest: Path | None = None,
) -> MarketOnlyResult:
    if max_unique_tickers is not None and max_unique_tickers <= 0:
        raise ValueError("max_unique_tickers must be > 0")
    selection_count = sum(
        value is not None and value != ()
        for value in (max_unique_tickers, tickers, universe_manifest)
    )
    if selection_count > 1:
        raise ValueError(
            "max_unique_tickers, tickers and universe_manifest are mutually exclusive"
        )

    snapshot, all_observations = load_sec_universe_snapshot(data_root)
    requested_company_ids = (
        load_historical_universe_company_ids(universe_manifest)
        if universe_manifest is not None
        else None
    )
    (
        observations,
        selected_unique_tickers,
        requested,
        missing,
        missing_companies,
    ) = _select_observations(
        all_observations,
        max_unique_tickers=max_unique_tickers,
        requested_tickers=tickers,
        requested_company_ids=requested_company_ids,
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
        "universe_manifest": str(universe_manifest) if universe_manifest is not None else None,
        "requested_companies": len(requested_company_ids or ()),
        "missing_requested_companies": list(missing_companies),
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
    requested_company_ids: tuple[str, ...] | None = None,
) -> tuple[
    tuple[IssuerObservation, ...],
    int,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    selection_count = sum(
        value is not None and value != ()
        for value in (max_unique_tickers, requested_tickers, requested_company_ids)
    )
    if selection_count > 1:
        raise ValueError(
            "requested_company_ids, requested_tickers and max_unique_tickers are mutually exclusive"
        )

    if requested_company_ids:
        requested_company_ids = tuple(dict.fromkeys(requested_company_ids))
        requested_set = set(requested_company_ids)
        selected = tuple(
            observation
            for observation in observations
            if company_id_from_sec_cik(observation.sec_cik) in requested_set
        )
        present = {
            company_id_from_sec_cik(observation.sec_cik) for observation in selected
        }
        missing_companies = tuple(
            company_id for company_id in requested_company_ids if company_id not in present
        )
        selected_unique_tickers = len(
            {_normalized_ticker(item.ticker) for item in selected}
        )
        return selected, selected_unique_tickers, (), (), missing_companies

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
        return tuple(selected), len(present), requested, missing, ()

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
    return selected, selected_unique_tickers, (), (), ()


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
    selection.add_argument(
        "--universe-manifest",
        type=Path,
        help=(
            "Historical universe JSON produced by experiments.historical_universe; "
            "all SEC-observed tickers for those canonical companies are retained."
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        credentials = load_tiingo_credentials(args.data_root)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    with TiingoClient(credentials.token) as tiingo_client:
        result = populate_market_from_snapshot(
            args.data_root,
            tiingo_client=tiingo_client,
            market_start=args.market_start,
            market_end=args.market_end,
            max_unique_tickers=args.max_unique_tickers,
            tickers=tuple(args.tickers) if args.tickers else None,
            universe_manifest=args.universe_manifest,
        )
    print(json.dumps(_jsonable(asdict(result)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
