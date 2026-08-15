from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import duckdb

from stock_trading.entities import company_id_from_sec_cik

from .sec_snapshot import load_sec_universe_snapshot


HISTORICAL_UNIVERSE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class HistoricalUniverseResult:
    output_path: Path
    selected_companies: int
    selected_tickers: int
    excluded_existing_companies: int
    candidate_companies: int


def build_historical_universe(
    data_root: Path,
    *,
    max_companies: int,
    seed: str = "historical-holdout-v1",
    exclude_market_db: Path | None = None,
    output_path: Path | None = None,
) -> HistoricalUniverseResult:
    """Select a deterministic SEC-company sample without looking at market outcomes.

    Selection is a stable SHA-256 rank over canonical SEC company IDs. The rank
    depends only on ``seed`` and company identity, never prices, returns, survival,
    model scores, or future labels. All issuer/ticker observations for selected
    companies remain available to the market resolver so ticker changes are not
    collapsed into a current-symbol universe.

    ``exclude_market_db`` is intended for a fresh holdout universe: any company
    already mapped in that DuckDB market store is removed before sampling.
    """

    if max_companies <= 0:
        raise ValueError("max_companies must be > 0")
    seed = seed.strip()
    if not seed:
        raise ValueError("seed must not be empty")

    snapshot, observations = load_sec_universe_snapshot(data_root)
    by_company: dict[str, list] = {}
    sec_cik_by_company: dict[str, str] = {}
    for observation in observations:
        company_id = company_id_from_sec_cik(observation.sec_cik)
        by_company.setdefault(company_id, []).append(observation)
        sec_cik_by_company[company_id] = observation.sec_cik

    excluded_company_ids = _mapped_company_ids(exclude_market_db)
    candidates = [
        company_id for company_id in by_company if company_id not in excluded_company_ids
    ]
    ranked = sorted(
        candidates,
        key=lambda company_id: (
            hashlib.sha256(f"{seed}:{company_id}".encode("utf-8")).hexdigest(),
            company_id,
        ),
    )
    selected_ids = tuple(ranked[:max_companies])
    if not selected_ids:
        raise ValueError("no SEC companies remain after exclusions")

    companies: list[dict[str, object]] = []
    first_year_counts: Counter[int] = Counter()
    selected_tickers: set[str] = set()
    for company_id in selected_ids:
        company_observations = sorted(
            by_company[company_id],
            key=lambda item: (
                item.observed_date,
                item.ticker,
                item.issuer_name,
            ),
        )
        first_observed = company_observations[0].observed_date
        tickers = sorted({_normalized_ticker(item.ticker) for item in company_observations})
        issuer_names = sorted({item.issuer_name for item in company_observations})
        first_year_counts[first_observed.year] += 1
        selected_tickers.update(tickers)
        companies.append(
            {
                "company_id": company_id,
                "sec_cik": sec_cik_by_company[company_id],
                "first_observed_date": first_observed.isoformat(),
                "tickers": tickers,
                "issuer_names": issuer_names,
                "issuer_observation_count": len(company_observations),
            }
        )

    output_path = output_path or data_root / "manifests" / "historical_holdout_universe.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": HISTORICAL_UNIVERSE_SCHEMA_VERSION,
        "selection_method": "stable_sha256_rank_of_canonical_sec_company_id",
        "selection_uses_market_outcomes": False,
        "selection_uses_model_outputs": False,
        "seed": seed,
        "max_companies": max_companies,
        "snapshot_start_quarter": f"{snapshot.start_year}Q{snapshot.start_quarter}",
        "snapshot_end_quarter": f"{snapshot.end_year}Q{snapshot.end_quarter}",
        "candidate_companies": len(candidates),
        "excluded_existing_companies": len(excluded_company_ids & set(by_company)),
        "selected_companies": len(companies),
        "selected_unique_tickers": len(selected_tickers),
        "first_observed_year_counts": {
            str(year): count for year, count in sorted(first_year_counts.items())
        },
        "exclude_market_db": str(exclude_market_db) if exclude_market_db is not None else None,
        "companies": companies,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return HistoricalUniverseResult(
        output_path=output_path,
        selected_companies=len(companies),
        selected_tickers=len(selected_tickers),
        excluded_existing_companies=len(excluded_company_ids & set(by_company)),
        candidate_companies=len(candidates),
    )


def load_historical_universe_company_ids(path: str | Path) -> tuple[str, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(raw.get("schema_version", -1)) != HISTORICAL_UNIVERSE_SCHEMA_VERSION:
        raise ValueError("unsupported historical universe manifest schema")
    companies = raw.get("companies")
    if not isinstance(companies, list) or not companies:
        raise ValueError("historical universe manifest contains no companies")
    company_ids: list[str] = []
    for row in companies:
        if not isinstance(row, dict) or not isinstance(row.get("company_id"), str):
            raise ValueError("invalid company row in historical universe manifest")
        company_id = row["company_id"].strip()
        if not company_id:
            raise ValueError("empty company_id in historical universe manifest")
        company_ids.append(company_id)
    return tuple(dict.fromkeys(company_ids))


def _mapped_company_ids(market_db: Path | None) -> set[str]:
    if market_db is None or not market_db.exists():
        return set()
    with duckdb.connect(str(market_db), read_only=True) as connection:
        tables = {
            str(row[0])
            for row in connection.execute("SHOW TABLES").fetchall()
        }
        if "company_security_map" not in tables:
            return set()
        rows = connection.execute(
            "SELECT DISTINCT company_id FROM company_security_map"
        ).fetchall()
    return {str(row[0]) for row in rows}


def _normalized_ticker(value: str) -> str:
    return value.strip().upper().replace(".", "-")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic historical SEC-company universe for a fresh "
            "holdout/backfill without using prices or model outcomes."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--max-companies", type=int, required=True)
    parser.add_argument("--seed", default="historical-holdout-v1")
    parser.add_argument(
        "--exclude-market-db",
        type=Path,
        help="Exclude companies already mapped in this market DuckDB.",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = build_historical_universe(
        args.data_root,
        max_companies=args.max_companies,
        seed=args.seed,
        exclude_market_db=args.exclude_market_db,
        output_path=args.output,
    )
    print(json.dumps({**asdict(result), "output_path": str(result.output_path)}, indent=2))


if __name__ == "__main__":
    main()
