from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from stock_trading.ml.multi_horizon import multi_horizon_maturity_dates
from stock_trading.research.strategy_factory import StrategyVariantSpec

from . import lightgbm_strategy_factory_executable as executable_factory
from . import lightgbm_strategy_factory_executable_maturity as maturity_factory
from . import lightgbm_strategy_qualify as base_qualify
from . import lightgbm_strategy_qualify_executable as legacy_qualify
from .lightgbm_validation_rank import _json_safe


SCHEMA_VERSION = "lightgbm-strategy-finalist-qualification-executable-maturity-v1"


def run_lightgbm_strategy_qualification_executable_maturity(
    experiment_dir: str | Path,
    *,
    market_db: str | Path,
    benchmark_security_id: str,
    generation_id: str = maturity_factory.DEFAULT_GENERATION_ID,
    workers: int = 4,
    threads_per_worker: int = 2,
    market_read_cache_series: int = 200,
    tolerance: float = 1e-12,
) -> Path:
    """Exactly re-train and qualify maturity-safe execution-realistic finalists."""

    if workers <= 0 or threads_per_worker <= 0:
        raise ValueError("workers and threads_per_worker must be > 0")
    if market_read_cache_series <= 0:
        raise ValueError("market_read_cache_series must be > 0")
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")

    root = Path(experiment_dir)
    generation_root = root / "strategy_factory" / generation_id
    report_path = generation_root / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"missing strategy factory report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _validate_report(report, generation_id)

    finalists = tuple(report.get("finalists") or ())
    if not finalists:
        raise ValueError("factory report contains no finalists")
    screening_by_id = {
        item["spec"]["variant_id"]: item for item in report.get("results") or ()
    }
    specs: list[StrategyVariantSpec] = []
    finalist_by_id: dict[str, Mapping[str, Any]] = {}
    for finalist in finalists:
        spec = base_qualify._spec_from_json(finalist["spec"])
        if spec.variant_id in finalist_by_id:
            raise ValueError(f"duplicate finalist {spec.variant_id}")
        if spec.variant_id not in screening_by_id:
            raise ValueError(f"finalist {spec.variant_id} missing full screening result")
        specs.append(spec)
        finalist_by_id[spec.variant_id] = finalist

    policy = report["portfolio_policy"]
    realism = report["execution_realism"]
    quality_manifest = Path(str(realism["market_quality_manifest"]))
    common = {
        "starting_capital": float(policy["starting_capital"]),
        "allocation_pct": float(policy["allocation_pct"]),
        "max_open_positions": int(policy["max_open_positions"]),
        "round_trip_cost_bps": float(policy["round_trip_cost_bps"]),
        "min_train_rows": 100,
        "max_trailing_adv_participation_pct": float(
            realism["max_trailing_adv_participation_pct"]
        ),
        "max_entry_day_participation_pct": float(
            realism["max_entry_day_participation_pct"]
        ),
    }
    prepared = executable_factory._prepare_executable_data(
        root,
        market_db=market_db,
        benchmark_security_id=benchmark_security_id,
        market_quality_manifest=quality_manifest,
        market_read_cache_series=market_read_cache_series,
    )
    if prepared.quality_exclusion_count != int(realism["verified_quality_exclusion_count"]):
        raise ValueError("market-quality exclusion count changed since screening")
    if len(prepared.invalid_target_keys) != int(realism["invalid_target_count"]):
        raise ValueError("invalid target count changed since screening")

    old_omp = os.environ.get("OMP_NUM_THREADS")
    old_openblas = os.environ.get("OPENBLAS_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)

    qualified: dict[str, dict[str, Any]] = {}
    try:
        if workers == 1:
            _initialize_worker(common, prepared)
            for spec in specs:
                qualified[spec.variant_id] = _qualify_variant(
                    spec,
                    screening_by_id[spec.variant_id],
                    tolerance,
                )
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_worker,
                initargs=(common, prepared),
            ) as executor:
                futures = {
                    executor.submit(
                        _qualify_variant,
                        spec,
                        screening_by_id[spec.variant_id],
                        tolerance,
                    ): spec
                    for spec in specs
                }
                for future in as_completed(futures):
                    spec = futures[future]
                    qualified[spec.variant_id] = future.result()
    finally:
        if old_omp is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = old_omp
        if old_openblas is None:
            os.environ.pop("OPENBLAS_NUM_THREADS", None)
        else:
            os.environ["OPENBLAS_NUM_THREADS"] = old_openblas

    ordered = []
    for spec in specs:
        item = qualified[spec.variant_id]
        selection = finalist_by_id[spec.variant_id]
        ordered.append(
            {
                "variant_id": spec.variant_id,
                "selection_score": selection.get("selection_score"),
                "maximum_overlap_with_earlier_finalist": selection.get(
                    "maximum_overlap_with_earlier_finalist"
                ),
                **item,
            }
        )

    payload = _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "generation_id": generation_id,
            "source_report": str(report_path),
            "data": {
                "market_db": str(market_db),
                "benchmark_security_id": benchmark_security_id,
                "point_in_time_model_inputs": True,
                "execution_day_liquidity_hidden_from_strategy": True,
                "prepared_source_row_count": len(prepared.rows),
                "worker_market_db_access": False,
                "screening_models_reused": False,
                "finalists_retrained_from_scratch": True,
                "full_horizon_maturity_required": True,
                "maturity_fence": "latest_requested_horizon_exit_before_test_year",
            },
            "execution_realism": {
                "market_quality_manifest": str(quality_manifest),
                "verified_quality_exclusion_count": prepared.quality_exclusion_count,
                "invalid_target_count": len(prepared.invalid_target_keys),
                "max_trailing_adv_participation_pct": common[
                    "max_trailing_adv_participation_pct"
                ],
                "max_entry_day_participation_pct": common[
                    "max_entry_day_participation_pct"
                ],
                "full_fill_required": True,
                "return_cap_applied": False,
                "full_horizon_maturity_required": True,
                "maturity_fence": "latest_requested_horizon_exit_before_test_year",
            },
            "replay_policy": {
                "float_tolerance": tolerance,
                "requires_zero_generation_failures": True,
                "requires_exact_trade_identity": True,
                "requires_exact_horizon_counts": True,
                "requires_yearly_return_identity": True,
                "requires_exact_execution_diagnostics": True,
                "requires_market_quality_manifest_identity": True,
                "requires_full_horizon_maturity": True,
            },
            "all_finalists_exactly_reproduced": all(
                item["exact_screening_identity_verified"] for item in ordered
            ),
            "finalists": ordered,
        }
    )
    output_path = generation_root / "qualification.json"
    if output_path.exists():
        raise FileExistsError(
            f"qualification already exists: {output_path}; use a fresh generation"
        )
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return output_path


def _initialize_worker(common: dict[str, Any], prepared) -> None:
    executable_factory._initialize_worker(common, prepared)


def _qualify_variant(
    spec: StrategyVariantSpec,
    screening_result: Mapping[str, Any],
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    context = executable_factory._CONTEXT
    if context is None:
        raise RuntimeError("maturity-safe qualification worker was not initialized")
    maturity_dates = multi_horizon_maturity_dates(
        context.rows,
        context.targets,
        horizons=spec.horizons,
    )
    original = context
    executable_factory._CONTEXT = replace(
        context,
        rows=tuple(
            replace(row, exit_date_20d=maturity_dates[row.event_id])
            for row in context.rows
        ),
    )
    try:
        return legacy_qualify._qualify_variant(
            spec,
            screening_result,
            tolerance=tolerance,
        )
    finally:
        executable_factory._CONTEXT = original


def _validate_report(report: Mapping[str, Any], generation_id: str) -> None:
    if report.get("schema_version") != maturity_factory.SCHEMA_VERSION:
        raise ValueError("qualification requires a maturity-safe executable factory report")
    generation = report.get("generation") or {}
    if generation.get("generation_id") != generation_id:
        raise ValueError("factory report generation_id mismatch")
    if int(generation.get("completed_hypotheses", 0)) <= 0:
        raise ValueError("factory report contains no completed hypotheses")
    if int(generation.get("failed_hypotheses", -1)) != 0:
        raise ValueError("qualification requires zero failed hypotheses")
    realism = report.get("execution_realism") or {}
    if realism.get("enabled") is not True:
        raise ValueError("execution realism is not enabled")
    if realism.get("full_fill_required") is not True:
        raise ValueError("qualification requires full-fill execution policy")
    if realism.get("return_cap_applied") is not False:
        raise ValueError("qualification refuses reports with a return cap")
    if realism.get("full_horizon_maturity_required") is not True:
        raise ValueError("qualification requires full-horizon maturity")
    if realism.get("maturity_fence") != "latest_requested_horizon_exit_before_test_year":
        raise ValueError("unexpected maturity fence")


def _compact_console(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload["schema_version"],
        "generation_id": payload["generation_id"],
        "all_finalists_exactly_reproduced": payload[
            "all_finalists_exactly_reproduced"
        ],
        "full_horizon_maturity_required": True,
        "execution_realism": payload["execution_realism"],
        "finalists": [
            {
                "variant_id": item["variant_id"],
                "scorecard": item["scorecard"],
                "execution_diagnostics": item["execution_diagnostics"],
                "qualification_flags": item["qualification_flags"],
                "diagnostics": {
                    "trade_count": item["diagnostics"]["trade_count"],
                    "unique_company_count": item["diagnostics"]["unique_company_count"],
                    "largest_positive_trade_pnl_fraction": item["diagnostics"][
                        "largest_positive_trade_pnl_fraction"
                    ],
                    "largest_positive_company_pnl_fraction": item["diagnostics"][
                        "largest_positive_company_pnl_fraction"
                    ],
                    "best_three_years": item["diagnostics"]["best_three_years"],
                    "compounded_return_excluding_best_three_years": item[
                        "diagnostics"
                    ]["compounded_return_excluding_best_three_years"],
                    "gross_return_distribution": item["diagnostics"][
                        "gross_return_distribution"
                    ],
                },
            }
            for item in payload["finalists"]
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exactly qualify G002m finalists under full-horizon target maturity."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--market-db", type=Path, required=True)
    parser.add_argument("--benchmark-security-id", default="benchmark_spy")
    parser.add_argument("--generation-id", default=maturity_factory.DEFAULT_GENERATION_ID)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--market-read-cache-series", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser


def main() -> None:
    args = _parser().parse_args()
    path = run_lightgbm_strategy_qualification_executable_maturity(
        args.experiment_dir,
        market_db=args.market_db,
        benchmark_security_id=args.benchmark_security_id,
        generation_id=args.generation_id,
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        market_read_cache_series=args.market_read_cache_series,
        tolerance=args.tolerance,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(_compact_console(payload), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
