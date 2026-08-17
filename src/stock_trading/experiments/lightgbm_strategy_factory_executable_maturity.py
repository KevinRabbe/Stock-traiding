from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from stock_trading.ml.multi_horizon import multi_horizon_maturity_dates
from stock_trading.research.strategy_factory import (
    PopulationGate,
    StrategyVariantSpec,
    design_space_size,
    generate_population,
    select_diverse_finalists,
)

from . import lightgbm_strategy_factory as base_factory
from . import lightgbm_strategy_factory_executable as legacy
from .lightgbm_validation_rank import _json_safe


SCHEMA_VERSION = "lightgbm-strategy-factory-executable-maturity-v1"
DEFAULT_GENERATION_ID = "g002m"


def run_lightgbm_strategy_factory_executable_maturity(
    experiment_dir: str | Path,
    *,
    market_db: str | Path,
    benchmark_security_id: str,
    generation_id: str = DEFAULT_GENERATION_ID,
    generation_seed: int = 20260816,
    population_size: int = 48,
    workers: int = 4,
    threads_per_worker: int = 2,
    starting_capital: float = 10_000.0,
    allocation_pct: float = 0.02,
    max_open_positions: int = 15,
    round_trip_cost_bps: float = 20.0,
    max_trailing_adv_participation_pct: float = 0.01,
    max_entry_day_participation_pct: float = 0.01,
    market_quality_manifest: str | Path = "data/manifests/market_quality_verified.json",
    min_train_rows: int = 100,
    finalist_count: int = 8,
    max_trade_overlap: float = 0.75,
    min_profit_factor: float = 1.05,
    min_trades: int = 75,
    max_realized_drawdown: float = 0.05,
    market_read_cache_series: int = 200,
) -> base_factory.StrategyFactoryGenerationResult:
    """Run G002 execution realism with a strict full-horizon maturity fence.

    This is intentionally a new generation schema rather than a silent rewrite of
    G002. Every variant/spec remains identical, but train/validation eligibility is
    fenced by the latest realized exit date among the horizons that variant uses.
    A 60-session target can therefore never influence a test year until that exact
    target has matured before January 1 of the test year.
    """

    if not generation_id.strip():
        raise ValueError("generation_id must not be empty")
    if population_size <= 0:
        raise ValueError("population_size must be > 0")
    if workers <= 0 or threads_per_worker <= 0:
        raise ValueError("workers and threads_per_worker must be > 0")
    if starting_capital <= 0:
        raise ValueError("starting_capital must be > 0")
    if not 0.0 < allocation_pct <= 1.0:
        raise ValueError("allocation_pct must be in (0, 1]")
    if max_open_positions <= 0 or min_train_rows <= 0:
        raise ValueError("position/train limits must be > 0")
    if round_trip_cost_bps < 0:
        raise ValueError("round_trip_cost_bps must be >= 0")
    for name, value in (
        ("max_trailing_adv_participation_pct", max_trailing_adv_participation_pct),
        ("max_entry_day_participation_pct", max_entry_day_participation_pct),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
    if market_read_cache_series <= 0:
        raise ValueError("market_read_cache_series must be > 0")

    root = Path(experiment_dir)
    if not (root / "training_rows.jsonl").exists():
        raise FileNotFoundError(f"missing training rows: {root / 'training_rows.jsonl'}")
    quality_manifest = Path(market_quality_manifest)
    if not quality_manifest.exists():
        raise FileNotFoundError(f"missing market quality manifest: {quality_manifest}")

    specs = generate_population(
        generation_seed=generation_seed,
        population_size=population_size,
    )
    generation_root = root / "strategy_factory" / generation_id
    generation_root.mkdir(parents=True, exist_ok=True)
    output_path = generation_root / "report.json"
    if output_path.exists():
        raise FileExistsError(
            f"generation report already exists: {output_path}; use a new generation_id"
        )

    prepared = legacy._prepare_executable_data(
        root,
        market_db=market_db,
        benchmark_security_id=benchmark_security_id,
        market_quality_manifest=quality_manifest,
        market_read_cache_series=market_read_cache_series,
    )
    common = {
        "starting_capital": starting_capital,
        "allocation_pct": allocation_pct,
        "max_open_positions": max_open_positions,
        "round_trip_cost_bps": round_trip_cost_bps,
        "min_train_rows": min_train_rows,
        "max_trailing_adv_participation_pct": max_trailing_adv_participation_pct,
        "max_entry_day_participation_pct": max_entry_day_participation_pct,
    }

    old_omp = os.environ.get("OMP_NUM_THREADS")
    old_openblas = os.environ.get("OPENBLAS_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)

    completed: list[Any] = []
    failures: list[dict[str, str]] = []
    try:
        if workers == 1:
            _initialize_worker(common, prepared)
            for spec in specs:
                try:
                    completed.append(_evaluate_variant(spec))
                except Exception as exc:
                    failures.append(
                        {
                            "variant_id": spec.variant_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_worker,
                initargs=(common, prepared),
            ) as executor:
                futures = {
                    executor.submit(_evaluate_variant, spec): spec for spec in specs
                }
                for future in as_completed(futures):
                    spec = futures[future]
                    try:
                        completed.append(future.result())
                    except Exception as exc:
                        failures.append(
                            {
                                "variant_id": spec.variant_id,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
    finally:
        if old_omp is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = old_omp
        if old_openblas is None:
            os.environ.pop("OPENBLAS_NUM_THREADS", None)
        else:
            os.environ["OPENBLAS_NUM_THREADS"] = old_openblas

    completed.sort(key=lambda item: item.result.spec.variant_id)
    failures.sort(key=lambda item: item["variant_id"])
    results = [item.result for item in completed]
    selection = select_diverse_finalists(
        results,
        gate=PopulationGate(
            min_compounded_return=0.0,
            min_profit_factor=min_profit_factor,
            min_trades=min_trades,
            max_realized_drawdown=max_realized_drawdown,
        ),
        finalist_count=finalist_count,
        max_trade_overlap=max_trade_overlap,
    )
    evaluation_by_id = {item.result.spec.variant_id: item for item in completed}

    finalist_payload = []
    for finalist in selection.finalists:
        evaluation = evaluation_by_id[finalist.variant_id]
        finalist_payload.append(
            {
                **asdict(finalist),
                "scorecard": evaluation.result.as_json()["scorecard"],
                "spec": evaluation.result.spec.as_json(),
                "execution_diagnostics": legacy._execution_diagnostics(evaluation),
            }
        )

    payload = _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "generation": {
                "generation_id": generation_id,
                "generation_seed": generation_seed,
                "design_space_size": design_space_size(),
                "population_size_requested": population_size,
                "attempted_hypotheses": len(specs),
                "completed_hypotheses": len(completed),
                "failed_hypotheses": len(failures),
                "models_persisted_during_screening": False,
                "retrain_from_scratch": True,
                "parent_generation_design": "g002",
            },
            "compute": {
                "workers": workers,
                "threads_per_worker": threads_per_worker,
                "market_preparation_processes": 1,
                "worker_market_db_access": False,
            },
            "data": {
                "experiment_dir": str(root),
                "market_db": str(market_db),
                "benchmark_security_id": benchmark_security_id,
                "point_in_time_model_inputs": True,
                "execution_day_liquidity_hidden_from_strategy": True,
                "prepared_row_count": len(prepared.rows),
                "prepared_security_count": len(set(prepared.security_ids.values())),
                "parent_market_cache_stats": prepared.market_cache_stats,
                "full_horizon_maturity_required": True,
                "maturity_fence": "latest_requested_horizon_exit_before_test_year",
            },
            "execution_realism": {
                "enabled": True,
                "reference_order_capital": starting_capital * allocation_pct,
                "trailing_adv_feature": "market.avg_dollar_volume_20d",
                "max_trailing_adv_participation_pct": max_trailing_adv_participation_pct,
                "entry_day_notional": "raw_open_x_raw_daily_volume",
                "max_entry_day_participation_pct": max_entry_day_participation_pct,
                "full_fill_required": True,
                "market_quality_manifest": str(quality_manifest),
                "verified_quality_exclusion_count": prepared.quality_exclusion_count,
                "invalid_target_count": len(prepared.invalid_target_keys),
                "return_cap_applied": False,
                "full_horizon_maturity_required": True,
                "maturity_fence": "latest_requested_horizon_exit_before_test_year",
            },
            "portfolio_policy": {
                "starting_capital": starting_capital,
                "allocation_pct": allocation_pct,
                "max_open_positions": max_open_positions,
                "round_trip_cost_bps": round_trip_cost_bps,
                "one_active_position_per_company": True,
            },
            "selection_policy": {
                "min_compounded_return": 0.0,
                "min_profit_factor": min_profit_factor,
                "min_trades": min_trades,
                "max_realized_drawdown": max_realized_drawdown,
                "max_trade_jaccard_overlap": max_trade_overlap,
                "finalist_count_target": finalist_count,
                "eligible_count": selection.eligible_count,
                "composite_rank_weights": {
                    "compounded_return": 0.30,
                    "profit_factor": 0.20,
                    "profitable_year_rate": 0.15,
                    "inverse_drawdown": 0.15,
                    "return_excluding_best_year": 0.20,
                },
            },
            "finalists": finalist_payload,
            "rejected_by_profitability_gate": list(selection.rejected_gate),
            "failures": failures,
            "results": [
                {
                    **item.result.as_json(),
                    "execution_diagnostics": legacy._execution_diagnostics(item),
                }
                for item in completed
            ],
            "tested_specs": [spec.as_json() for spec in specs],
        }
    )
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return base_factory.StrategyFactoryGenerationResult(
        generation_id=generation_id,
        output_path=output_path,
        attempted=len(specs),
        completed=len(completed),
        failed=len(failures),
    )


def _initialize_worker(common: dict[str, Any], prepared) -> None:
    legacy._initialize_worker(common, prepared)


def _evaluate_variant(spec: StrategyVariantSpec):
    context = legacy._CONTEXT
    if context is None:
        raise RuntimeError("maturity-safe factory worker was not initialized")
    maturity_dates = multi_horizon_maturity_dates(
        context.rows,
        context.targets,
        horizons=spec.horizons,
    )
    original = context
    legacy._CONTEXT = replace(
        context,
        rows=tuple(
            replace(row, exit_date_20d=maturity_dates[row.event_id])
            for row in context.rows
        ),
    )
    try:
        return legacy._evaluate_variant(spec)
    finally:
        legacy._CONTEXT = original


def _compact_console(payload: dict[str, Any]) -> dict[str, Any]:
    compact = legacy._compact_console(payload)
    compact["full_horizon_maturity_required"] = True
    compact["maturity_fence"] = "latest_requested_horizon_exit_before_test_year"
    return compact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and walk-forward test the G002 LightGBM population with "
            "execution realism plus strict full-horizon target maturity."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--market-db", type=Path, required=True)
    parser.add_argument("--benchmark-security-id", default="benchmark_spy")
    parser.add_argument("--generation-id", default=DEFAULT_GENERATION_ID)
    parser.add_argument("--generation-seed", type=int, default=20260816)
    parser.add_argument("--population-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--starting-capital", type=float, default=10_000.0)
    parser.add_argument("--allocation-pct", type=float, default=0.02)
    parser.add_argument("--max-open-positions", type=int, default=15)
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    parser.add_argument("--max-trailing-adv-participation-pct", type=float, default=0.01)
    parser.add_argument("--max-entry-day-participation-pct", type=float, default=0.01)
    parser.add_argument(
        "--market-quality-manifest",
        type=Path,
        default=Path("data/manifests/market_quality_verified.json"),
    )
    parser.add_argument("--min-train-rows", type=int, default=100)
    parser.add_argument("--finalist-count", type=int, default=8)
    parser.add_argument("--max-trade-overlap", type=float, default=0.75)
    parser.add_argument("--min-profit-factor", type=float, default=1.05)
    parser.add_argument("--min-trades", type=int, default=75)
    parser.add_argument("--max-realized-drawdown", type=float, default=0.05)
    parser.add_argument("--market-read-cache-series", type=int, default=200)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_lightgbm_strategy_factory_executable_maturity(
        args.experiment_dir,
        market_db=args.market_db,
        benchmark_security_id=args.benchmark_security_id,
        generation_id=args.generation_id,
        generation_seed=args.generation_seed,
        population_size=args.population_size,
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        starting_capital=args.starting_capital,
        allocation_pct=args.allocation_pct,
        max_open_positions=args.max_open_positions,
        round_trip_cost_bps=args.round_trip_cost_bps,
        max_trailing_adv_participation_pct=args.max_trailing_adv_participation_pct,
        max_entry_day_participation_pct=args.max_entry_day_participation_pct,
        market_quality_manifest=args.market_quality_manifest,
        min_train_rows=args.min_train_rows,
        finalist_count=args.finalist_count,
        max_trade_overlap=args.max_trade_overlap,
        min_profit_factor=args.min_profit_factor,
        min_trades=args.min_trades,
        max_realized_drawdown=args.max_realized_drawdown,
        market_read_cache_series=args.market_read_cache_series,
    )
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    print(json.dumps(_compact_console(payload), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
