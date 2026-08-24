from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stock_trading.ml.walk_forward import annual_walk_forward_splits
from stock_trading.research.execution_realism import (
    load_market_quality_exclusions,
    target_overlaps_exclusion,
)

from . import lightgbm_strategy_factory_executable as executable
from .lightgbm_validation_rank import _json_safe
from .v5_execution_filter_attribution import (
    _evaluate_base_subset,
    _quality_valid_rows,
    _scenario_from_base,
)
from .v5_profit_proof import (
    DEFAULT_MARKET_QUALITY_MANIFEST,
    _resolve_market_inputs,
    current_v5_profit_proof_spec,
)


SCHEMA_VERSION = "v5-quality-row-attribution-v1"


@dataclass(frozen=True, slots=True)
class V5QualityRowAttributionResult:
    output_path: Path
    removed_row_count: int


def run_v5_quality_row_attribution(
    experiment_dir: str | Path,
    *,
    runtime_dir: str | Path = "data/runtime",
    market_db: str | Path | None = None,
    benchmark_security_id: str | None = None,
    starting_capital: float = 10_000.0,
    allocation_pct: float = 0.02,
    max_open_positions: int = 15,
    round_trip_cost_bps: float = 20.0,
    market_quality_manifest: str | Path = DEFAULT_MARKET_QUALITY_MANIFEST,
    min_train_rows: int = 100,
    market_read_cache_series: int = 200,
) -> V5QualityRowAttributionResult:
    """Measure how each verified market-quality row affects fixed-spec V5.

    The strategy, 20-session maturity convention, annual-reset portfolio and
    historical backtester are held fixed. The diagnostic compares the unfiltered
    reproducible V5 result against removing each quality-invalid row individually,
    then against cumulative removals and the complete verified exclusion set.
    It also prints each removed row's 5/20/60 targets and walk-forward roles so a
    large effect can be distinguished between corrupted-label leverage and generic
    model instability.
    """

    if starting_capital <= 0:
        raise ValueError("starting_capital must be > 0")
    if not 0.0 < allocation_pct <= 1.0:
        raise ValueError("allocation_pct must be in (0, 1]")
    if max_open_positions <= 0 or min_train_rows <= 0 or market_read_cache_series <= 0:
        raise ValueError("position/train/cache limits must be > 0")
    if round_trip_cost_bps < 0:
        raise ValueError("round_trip_cost_bps must be >= 0")

    root = Path(experiment_dir)
    runtime_root = Path(runtime_dir)
    resolved_market_db, resolved_benchmark = _resolve_market_inputs(
        runtime_root,
        market_db=market_db,
        benchmark_security_id=benchmark_security_id,
    )
    quality_manifest = Path(market_quality_manifest)
    if not quality_manifest.exists():
        raise FileNotFoundError(f"missing market quality manifest: {quality_manifest}")

    spec = current_v5_profit_proof_spec()
    horizons = tuple(int(item) for item in spec.horizons)
    prepared = executable._prepare_executable_data(
        root,
        market_db=resolved_market_db,
        benchmark_security_id=resolved_benchmark,
        market_quality_manifest=quality_manifest,
        market_read_cache_series=market_read_cache_series,
    )
    source_rows = tuple(prepared.rows)
    quality_rows = _quality_valid_rows(
        source_rows,
        horizons=horizons,
        invalid_target_keys=prepared.invalid_target_keys,
    )
    quality_ids = frozenset(row.event_id for row in quality_rows)
    removed_rows = tuple(
        sorted(
            (row for row in source_rows if row.event_id not in quality_ids),
            key=lambda row: (row.execution_date, row.event_id),
        )
    )
    if not removed_rows:
        raise ValueError("no quality-invalid rows found")

    common = {
        "starting_capital": starting_capital,
        "allocation_pct": allocation_pct,
        "max_open_positions": max_open_positions,
        "round_trip_cost_bps": round_trip_cost_bps,
        "min_train_rows": min_train_rows,
    }

    result_cache: dict[frozenset[str], Any] = {}

    def evaluate_removed(removed_ids: frozenset[str]) -> Any:
        cached = result_cache.get(removed_ids)
        if cached is not None:
            return cached
        rows = tuple(row for row in source_rows if row.event_id not in removed_ids)
        result = _evaluate_base_subset(spec, prepared, rows, common)
        result_cache[removed_ids] = result
        return result

    baseline = evaluate_removed(frozenset())
    all_removed_ids = frozenset(row.event_id for row in removed_rows)
    quality_only = evaluate_removed(all_removed_ids)
    baseline_scenario = _scenario_from_base("unfiltered_20d", baseline)
    quality_scenario = _scenario_from_base("all_quality_rows_removed_20d", quality_only)

    exclusions = load_market_quality_exclusions(quality_manifest)
    split_roles = _walk_forward_roles(source_rows)
    baseline_trade_ids = frozenset(baseline.trade_candidate_ids)

    row_diagnostics: list[dict[str, Any]] = []
    leave_one_out: list[dict[str, Any]] = []
    for row in removed_rows:
        invalid_horizons = [
            horizon
            for horizon in horizons
            if (row.event_id, horizon) in prepared.invalid_target_keys
        ]
        matching_exclusions = []
        for horizon in invalid_horizons:
            target = prepared.targets[row.event_id][horizon]
            for exclusion in exclusions:
                if target_overlaps_exclusion(
                    prepared.security_ids[row.event_id],
                    row.execution_date,
                    target.exit_date,
                    (exclusion,),
                ):
                    matching_exclusions.append(
                        {
                            "ticker": exclusion.ticker,
                            "security_id": exclusion.security_id,
                            "start_date": exclusion.start_date.isoformat(),
                            "end_date": exclusion.end_date.isoformat(),
                            "reason": exclusion.reason,
                            "overlapping_horizon": horizon,
                        }
                    )
        target_payload = {
            str(horizon): {
                "exit_date": prepared.targets[row.event_id][horizon].exit_date.isoformat(),
                "stock_return": prepared.targets[row.event_id][horizon].stock_return,
                "benchmark_return": prepared.targets[row.event_id][horizon].benchmark_return,
                "alpha": prepared.targets[row.event_id][horizon].alpha,
                "downside": prepared.targets[row.event_id][horizon].downside,
                "mfe": prepared.targets[row.event_id][horizon].mfe,
                "quality_invalid": horizon in invalid_horizons,
            }
            for horizon in horizons
        }
        row_diagnostics.append(
            {
                "event_id": row.event_id,
                "company_id": row.company_id,
                "security_id": prepared.security_ids[row.event_id],
                "decision_time": row.decision_time.isoformat(),
                "execution_date": row.execution_date.isoformat(),
                "invalid_horizons": invalid_horizons,
                "was_baseline_trade": row.event_id in baseline_trade_ids,
                "walk_forward_roles": split_roles.get(row.event_id, {}),
                "targets": target_payload,
                "matching_exclusions": matching_exclusions,
            }
        )

        result = evaluate_removed(frozenset((row.event_id,)))
        scenario = _scenario_from_base(f"remove_only:{row.event_id}", result)
        leave_one_out.append(
            {
                "event_id": row.event_id,
                "execution_date": row.execution_date.isoformat(),
                "scenario": scenario,
                "delta_vs_unfiltered": _scorecard_delta(baseline_scenario, scenario),
                "trade_jaccard_overlap_vs_unfiltered": _jaccard(
                    baseline.trade_candidate_ids,
                    result.trade_candidate_ids,
                ),
            }
        )

    cumulative: list[dict[str, Any]] = []
    cumulative_ids: set[str] = set()
    for row in removed_rows:
        cumulative_ids.add(row.event_id)
        result = evaluate_removed(frozenset(cumulative_ids))
        scenario = _scenario_from_base(
            f"remove_first_{len(cumulative_ids)}_quality_rows",
            result,
        )
        cumulative.append(
            {
                "removed_event_ids": sorted(cumulative_ids),
                "scenario": scenario,
                "delta_vs_unfiltered": _scorecard_delta(baseline_scenario, scenario),
                "trade_jaccard_overlap_vs_unfiltered": _jaccard(
                    baseline.trade_candidate_ids,
                    result.trade_candidate_ids,
                ),
            }
        )

    payload = _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_dir": str(root),
            "runtime_dir": str(runtime_root),
            "market_db": str(resolved_market_db),
            "benchmark_security_id": resolved_benchmark,
            "strategy_spec": spec.as_json(),
            "purpose": {
                "parameter_search": False,
                "threshold_tuning": False,
                "paper_policy_changed": False,
                "question": (
                    "Do the five verified quality-invalid rows contain corrupted target "
                    "leverage, or is V5 generically unstable to tiny row perturbations?"
                ),
            },
            "data": {
                "source_row_count": len(source_rows),
                "quality_removed_row_count": len(removed_rows),
                "quality_removed_fraction": len(removed_rows) / len(source_rows),
                "quality_remaining_row_count": len(quality_rows),
            },
            "baseline": baseline_scenario,
            "all_quality_rows_removed": {
                "scenario": quality_scenario,
                "delta_vs_unfiltered": _scorecard_delta(baseline_scenario, quality_scenario),
                "trade_jaccard_overlap_vs_unfiltered": _jaccard(
                    baseline.trade_candidate_ids,
                    quality_only.trade_candidate_ids,
                ),
            },
            "removed_rows": row_diagnostics,
            "leave_one_row_out": leave_one_out,
            "cumulative_removal": cumulative,
        }
    )
    output_path = root / "v5_quality_row_attribution.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return V5QualityRowAttributionResult(
        output_path=output_path,
        removed_row_count=len(removed_rows),
    )


def _walk_forward_roles(rows: tuple) -> dict[str, dict[str, list[int]]]:
    result: dict[str, dict[str, list[int]]] = {}
    for split in annual_walk_forward_splits(rows):
        for role, role_rows in (
            ("train_for_test_years", split.train_rows),
            ("validation_for_test_years", split.validation_rows),
            ("test_years", split.test_rows),
        ):
            for row in role_rows:
                result.setdefault(
                    row.event_id,
                    {
                        "train_for_test_years": [],
                        "validation_for_test_years": [],
                        "test_years": [],
                    },
                )[role].append(split.test_year)
    return result


def _scorecard_delta(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    return {
        "return_delta": float(current["return"]) - float(previous["return"]),
        "profit_factor_delta": float(current["profit_factor"]) - float(previous["profit_factor"]),
        "average_trade_alpha_delta": (
            float(current["average_trade_alpha"]) - float(previous["average_trade_alpha"])
        ),
        "trade_count_delta": int(current["total_trades"]) - int(previous["total_trades"]),
    }


def _jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _compact_console(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload["schema_version"],
        "data": payload["data"],
        "baseline": payload["baseline"],
        "all_quality_rows_removed": payload["all_quality_rows_removed"],
        "removed_rows": payload["removed_rows"],
        "leave_one_row_out": payload["leave_one_row_out"],
        "cumulative_removal": payload["cumulative_removal"],
        "output_path": str(Path(payload["experiment_dir"]) / "v5_quality_row_attribution.json"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute fixed V5 performance sensitivity to each verified market-quality row."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, default=Path("data/runtime"))
    parser.add_argument("--market-db", type=Path)
    parser.add_argument("--benchmark-security-id")
    parser.add_argument("--starting-capital", type=float, default=10_000.0)
    parser.add_argument("--allocation-pct", type=float, default=0.02)
    parser.add_argument("--max-open-positions", type=int, default=15)
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    parser.add_argument(
        "--market-quality-manifest",
        type=Path,
        default=DEFAULT_MARKET_QUALITY_MANIFEST,
    )
    parser.add_argument("--min-train-rows", type=int, default=100)
    parser.add_argument("--market-read-cache-series", type=int, default=200)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_v5_quality_row_attribution(
        args.experiment_dir,
        runtime_dir=args.runtime_dir,
        market_db=args.market_db,
        benchmark_security_id=args.benchmark_security_id,
        starting_capital=args.starting_capital,
        allocation_pct=args.allocation_pct,
        max_open_positions=args.max_open_positions,
        round_trip_cost_bps=args.round_trip_cost_bps,
        market_quality_manifest=args.market_quality_manifest,
        min_train_rows=args.min_train_rows,
        market_read_cache_series=args.market_read_cache_series,
    )
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    print(json.dumps(_compact_console(payload), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
