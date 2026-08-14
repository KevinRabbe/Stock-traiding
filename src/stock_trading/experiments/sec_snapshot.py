from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Callable

from stock_trading.core import Source
from stock_trading.entities import company_id_from_sec_cik
from stock_trading.market import IssuerObservation
from stock_trading.sec import QuarterlyArchiveParser
from stock_trading.storage import FileRawStore


SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_METADATA_NAME = "sec_snapshot.json"
SNAPSHOT_OBSERVATIONS_NAME = "sec_issuer_observations.jsonl"
SEC_COMPANIES_NAME = "sec_companies.jsonl"


@dataclass(frozen=True, slots=True)
class SecUniverseSnapshot:
    schema_version: int
    start_year: int
    start_quarter: int
    end_year: int
    end_quarter: int
    issuer_observations: int
    unique_tickers: int
    sec_companies: int
    source_artifacts: tuple[dict[str, str], ...]


def build_sec_universe_snapshot(
    data_root: Path,
    *,
    start_year: int = 2012,
    start_quarter: int = 1,
    end_year: int | None = None,
    end_quarter: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> SecUniverseSnapshot:
    if start_year < 2006:
        raise ValueError("SEC quarterly insider history starts in 2006")
    if start_quarter not in {1, 2, 3, 4}:
        raise ValueError("start_quarter must be 1..4")

    default_end_year, default_end_quarter = latest_completed_quarter(date.today())
    end_year = end_year if end_year is not None else default_end_year
    end_quarter = end_quarter if end_quarter is not None else default_end_quarter
    if end_quarter not in {1, 2, 3, 4}:
        raise ValueError("end_quarter must be 1..4")
    if (end_year, end_quarter) < (start_year, start_quarter):
        raise ValueError("end quarter must not precede start quarter")

    raw_store = FileRawStore(data_root / "raw")
    parser = QuarterlyArchiveParser()
    observations_by_key: dict[tuple[str, str, str], IssuerObservation] = {}
    company_manifest: dict[str, dict[str, object]] = {}
    source_artifacts: list[dict[str, str]] = []

    quarters = quarter_range(start_year, start_quarter, end_year, end_quarter)
    for position, (year, quarter) in enumerate(quarters, start=1):
        started = perf_counter()
        source_record_id = f"{year}Q{quarter}"
        raw = raw_store.latest(Source.SEC_QUARTERLY, source_record_id)
        if raw is None:
            raise FileNotFoundError(
                f"missing cached SEC quarterly archive {source_record_id}; "
                "populate the SEC cache before building a market-only snapshot"
            )

        transactions = parser.parse(
            raw.content if isinstance(raw.content, bytes) else raw.content.encode("utf-8")
        )
        source_artifacts.append(
            {
                "source_record_id": source_record_id,
                "artifact_id": raw.artifact_id,
                "sha256": raw.sha256,
            }
        )

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
            elapsed = perf_counter() - started
            progress(
                f"[{position:02d}/{len(quarters):02d}] {source_record_id} | "
                f"snapshot transactions={len(transactions):,} | {elapsed:.1f}s"
            )

    observations = tuple(observations_by_key.values())
    manifests_dir = data_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    observations_path = manifests_dir / SNAPSHOT_OBSERVATIONS_NAME
    with observations_path.open("w", encoding="utf-8", newline="\n") as handle:
        for observation in sorted(
            observations,
            key=lambda item: (item.sec_cik, item.ticker, item.issuer_name, item.observed_date),
        ):
            handle.write(
                json.dumps(
                    {
                        "sec_cik": observation.sec_cik,
                        "issuer_name": observation.issuer_name,
                        "ticker": observation.ticker,
                        "observed_date": observation.observed_date.isoformat(),
                    },
                    sort_keys=True,
                )
            )
            handle.write("\n")

    companies_path = manifests_dir / SEC_COMPANIES_NAME
    with companies_path.open("w", encoding="utf-8", newline="\n") as handle:
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

    snapshot = SecUniverseSnapshot(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        start_year=start_year,
        start_quarter=start_quarter,
        end_year=end_year,
        end_quarter=end_quarter,
        issuer_observations=len(observations),
        unique_tickers=len({_normalized_ticker(item.ticker) for item in observations}),
        sec_companies=len(company_manifest),
        source_artifacts=tuple(source_artifacts),
    )
    (manifests_dir / SNAPSHOT_METADATA_NAME).write_text(
        json.dumps(asdict(snapshot), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return snapshot


def load_sec_universe_snapshot(
    data_root: Path,
) -> tuple[SecUniverseSnapshot, tuple[IssuerObservation, ...]]:
    manifests_dir = data_root / "manifests"
    metadata_path = manifests_dir / SNAPSHOT_METADATA_NAME
    observations_path = manifests_dir / SNAPSHOT_OBSERVATIONS_NAME
    if not metadata_path.exists() or not observations_path.exists():
        raise FileNotFoundError(
            "SEC universe snapshot is missing; run "
            "python -m stock_trading.experiments.sec_snapshot first"
        )

    raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    snapshot = SecUniverseSnapshot(
        schema_version=int(raw_metadata["schema_version"]),
        start_year=int(raw_metadata["start_year"]),
        start_quarter=int(raw_metadata["start_quarter"]),
        end_year=int(raw_metadata["end_year"]),
        end_quarter=int(raw_metadata["end_quarter"]),
        issuer_observations=int(raw_metadata["issuer_observations"]),
        unique_tickers=int(raw_metadata["unique_tickers"]),
        sec_companies=int(raw_metadata["sec_companies"]),
        source_artifacts=tuple(raw_metadata["source_artifacts"]),
    )
    if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported SEC snapshot schema {snapshot.schema_version}; rebuild the snapshot"
        )

    observations: list[IssuerObservation] = []
    with observations_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                observations.append(
                    IssuerObservation(
                        sec_cik=row["sec_cik"],
                        issuer_name=row["issuer_name"],
                        ticker=row["ticker"],
                        observed_date=date.fromisoformat(row["observed_date"]),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid SEC snapshot observation at line {line_number}"
                ) from exc

    if len(observations) != snapshot.issuer_observations:
        raise ValueError(
            "SEC snapshot observation count does not match metadata; rebuild the snapshot"
        )
    return snapshot, tuple(observations)


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a reusable SEC issuer-universe snapshot from cached quarterly archives."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--start-quarter", type=int, default=1)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--end-quarter", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    snapshot = build_sec_universe_snapshot(
        args.data_root,
        start_year=args.start_year,
        start_quarter=args.start_quarter,
        end_year=args.end_year,
        end_quarter=args.end_quarter,
        progress=print,
    )
    print(json.dumps(asdict(snapshot), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
