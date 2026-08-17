from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from stock_trading.engine import (
    FixedAllocationPortfolioPolicy,
    PassThroughOpportunityRiskPolicy,
    PassThroughPortfolioRiskPolicy,
)
from stock_trading.ml import LightGbmTrainer, ProfitLightGbmTrainer
from stock_trading.ml.multi_horizon import row_for_horizon
from stock_trading.ml.walk_forward import annual_walk_forward_splits
from stock_trading.research import HistoricalYearResult, summarize_historical_years
from stock_trading.research.execution_realism import ExecutionRealisticHistoricalBacktester
from stock_trading.research.strategy_factory import (
    StrategyVariantResult,
    StrategyVariantSpec,
    apply_feature_profile,
    training_window_rows,
)
from stock_trading.strategies.v5_adaptive_horizon import (
    V5AdaptiveHorizonStrategy,
    V5CalibrationState,
    V5HorizonModels,
    V5StrategyConfig,
)

from . import lightgbm_strategy_factory as base_factory
from . import lightgbm_strategy_factory_executable as executable_factory
from . import lightgbm_strategy_qualify as base_qualify
from .lightgbm_validation_rank import _json_safe


def run_lightgbm_strategy_qualification_executable(
    experiment_dir: str | Path,
    *,
    market_db: str | Path,
    benchmark_security_id: str,
    generation_id: str = "g002",
    workers: int = 4,
    threads_per_worker: int = 2,
    market_read_cache_series: int = 200,
    tolerance: float = 1e-12,
) -> Path:
    """Retrain and exactly qualify execution-realistic strategy finalists.

    Screening model weights are discarded by G002. This command reconstructs the
    exact market-quality, PIT-liquidity and hidden entry-day fill assumptions from
    the saved generation report, retrains every finalist from scratch, and proves
    scorecard, trade, horizon, yearly-return and execution-diagnostic identity.
    Concentration diagnostics are then recorded before a finalist may progress to
    immutable shadow artifacts.
    """

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
    if report.get("schema_version") != "lightgbm-strategy-factory-executable-v1":
        raise ValueError(
            "execution-realistic qualification requires an executable factory report"
        )

    generation = report.get("generation") or {}
    if generation.get("generation_id") != generation_id:
        raise ValueError("factory report generation_id mismatch")
    if int(generation.get("completed_hypotheses", 0)) <= 0:
        raise ValueError("factory report contains no completed hypotheses")
    if int(generation.get("failed_hypotheses", 0)) != 0:
        raise ValueError(
            "factory generation has failed hypotheses; rerun/fix the generation "
            "before qualifying finalists"
        )

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

    policy = report.get("portfolio_policy") or {}
    realism = report.get("execution_realism") or {}
    if realism.get("enabled") is not True:
        raise ValueError("factory report does not declare execution realism enabled")
    if realism.get("full_fill_required") is not True:
        raise ValueError("qualification requires the G002 full-fill execution policy")
    if realism.get("return_cap_applied") is not False:
        raise ValueError("qualification refuses reports that applied a return cap")

    quality_manifest = Path(
        realism.get("market_quality_manifest", "data/manifests/market_quality_verified.json")
    )
    common = {
        "starting_capital": float(policy.get("starting_capital", 10_000.0)),
        "allocation_pct": float(policy.get("allocation_pct", 0.02)),
        "max_open_positions": int(policy.get("max_open_positions", 15)),
        "round_trip_cost_bps": float(policy.get("round_trip_cost_bps", 20.0)),
        "min_train_rows": 100,
        "max_trailing_adv_participation_pct": float(
            realism.get("max_trailing_adv_participation_pct", 0.01)
        ),
        "max_entry_day_participation_pct": float(
            realism.get("max_entry_day_participation_pct", 0.01)
        ),
    }
    prepared = executable_factory._prepare_executable_data(
        root,
        market_db=market_db,
        benchmark_security_id=benchmark_security_id,
        market_quality_manifest=quality_manifest,
        market_read_cache_series=market_read_cache_series,
    )

    expected_quality_count = int(realism.get("verified_quality_exclusion_count", -1))
    if prepared.quality_exclusion_count != expected_quality_count:
        raise ValueError(
            "market-quality exclusion count changed since screening: "
            f"screening={expected_quality_count}, replay={prepared.quality_exclusion_count}"
        )
    expected_invalid_targets = int(realism.get("invalid_target_count", -1))
    if len(prepared.invalid_target_keys) != expected_invalid_targets:
        raise ValueError(
            "invalid target count changed since screening: "
            f"screening={expected_invalid_targets}, replay={len(prepared.invalid_target_keys)}"
        )

    old_omp = os.environ.get("OMP_NUM_THREADS")
    old_openblas = os.environ.get("OPENBLAS_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)

    qualified: dict[str, dict[str, Any]] = {}
    try:
        if workers == 1:
            executable_factory._initialize_worker(common, prepared)
            for spec in specs:
                qualified[spec.variant_id] = _qualify_variant(
                    spec,
                    screening_by_id[spec.variant_id],
                    tolerance=tolerance,
                )
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=executable_factory._initialize_worker,
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
            "schema_version": "lightgbm-strategy-finalist-qualification-executable-v1",
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
            },
            "replay_policy": {
                "float_tolerance": tolerance,
                "requires_zero_generation_failures": True,
                "requires_exact_trade_identity": True,
                "requires_exact_horizon_counts": True,
                "requires_yearly_return_identity": True,
                "requires_exact_execution_diagnostics": True,
                "requires_market_quality_manifest_identity": True,
            },
            "all_finalists_exactly_reproduced": all(
                item["exact_screening_identity_verified"] for item in ordered
            ),
            "finalists": ordered,
        }
    )
    output_path = generation_root / "qualification.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return output_path


def _qualify_variant(
    spec: StrategyVariantSpec,
    screening_result: Mapping[str, Any],
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    result, year_results, execution_diagnostics = _evaluate_variant_with_trades(spec)
    base_qualify._assert_screening_identity(
        screening_result,
        result,
        tolerance=tolerance,
    )
    _assert_execution_diagnostics(
        screening_result.get("execution_diagnostics") or {},
        execution_diagnostics,
    )
    diagnostics = base_qualify._concentration_diagnostics(year_results)
    return {
        "exact_screening_identity_verified": True,
        "spec": result.spec.as_json(),
        "scorecard": result.as_json()["scorecard"],
        "execution_diagnostics": execution_diagnostics,
        "diagnostics": diagnostics,
        "qualification_flags": {
            "best_year_dependency": (
                result.compounded_return_excluding_best_year is not None
                and result.compounded_return_excluding_best_year <= 0.0
            ),
            "top_three_year_dependency": (
                diagnostics["compounded_return_excluding_best_three_years"] is not None
                and diagnostics["compounded_return_excluding_best_three_years"] <= 0.0
            ),
            "single_trade_positive_pnl_concentration_ge_25pct": (
                diagnostics["largest_positive_trade_pnl_fraction"] is not None
                and diagnostics["largest_positive_trade_pnl_fraction"] >= 0.25
            ),
            "single_company_positive_pnl_concentration_ge_25pct": (
                diagnostics["largest_positive_company_pnl_fraction"] is not None
                and diagnostics["largest_positive_company_pnl_fraction"] >= 0.25
            ),
        },
    }


def _evaluate_variant_with_trades(
    spec: StrategyVariantSpec,
) -> tuple[StrategyVariantResult, tuple[HistoricalYearResult, ...], dict[str, int]]:
    context = executable_factory._CONTEXT
    if context is None:
        raise RuntimeError("executable qualification worker was not initialized")

    executable_rows, quality_removed, liquidity_removed = executable_factory._filter_executable_rows(
        context.rows,
        spec,
        context,
    )
    rows = apply_feature_profile(executable_rows, spec.feature_profile)
    splits = annual_walk_forward_splits(rows)
    if not splits:
        raise ValueError("no annual walk-forward splits after execution filters")

    profitable_threshold = context.round_trip_cost_bps / 10_000.0
    portfolio_policy = FixedAllocationPortfolioPolicy(
        allocation_pct=context.allocation_pct,
        max_open_positions=context.max_open_positions,
        max_gross_exposure_pct=1.0,
        one_position_per_company=True,
    )
    backtester = ExecutionRealisticHistoricalBacktester(
        starting_capital=context.starting_capital,
        round_trip_cost_bps=context.round_trip_cost_bps,
    )
    profit_trainer = ProfitLightGbmTrainer(spec.training_config)
    alpha_trainer = LightGbmTrainer(spec.training_config)
    year_results: list[HistoricalYearResult] = []
    trade_ids: list[str] = []
    horizon_counts: Counter[int] = Counter()
    rejected_entry_liquidity = 0

    for split in splits:
        train_rows = training_window_rows(
            split.train_rows,
            test_year=split.test_year,
            window_years=spec.training_window_years,
        )
        if len(train_rows) < context.min_train_rows:
            raise ValueError(
                f"{spec.variant_id} has only {len(train_rows)} executable training rows for "
                f"test year {split.test_year}"
            )

        models: dict[int, V5HorizonModels] = {}
        for horizon in spec.horizons:
            train_h = tuple(
                row_for_horizon(row, context.targets[row.event_id][horizon])
                for row in train_rows
            )
            validation_h = tuple(
                row_for_horizon(row, context.targets[row.event_id][horizon])
                for row in split.validation_rows
            )
            models[horizon] = V5HorizonModels(
                profit=profit_trainer.train(
                    train_h,
                    validation_h,
                    profitable_return_threshold=profitable_threshold,
                ),
                alpha=alpha_trainer.train(train_h, validation_h),
            )

        validation_candidates = base_factory._feature_snapshots(
            split.validation_rows,
            context.security_ids,
        )
        strategy_config = V5StrategyConfig(
            strategy_id=spec.variant_id,
            horizons=spec.horizons,
            validation_top_fraction=spec.validation_top_fraction,
            alpha_rank_weight=spec.alpha_rank_weight,
            calibration_window_days=spec.calibration_window_days,
            min_expected_return=profitable_threshold,
            max_expected_downside=spec.max_expected_downside,
        )
        calibration = V5CalibrationState.from_validation(
            validation_candidates,
            models,
            strategy_config,
        )
        strategy = V5AdaptiveHorizonStrategy(models, calibration, strategy_config)
        historical = base_factory._historical_candidates(
            split.test_rows,
            context.targets,
            context.security_ids,
        )
        backtest, execution = backtester.run(
            strategy=strategy,
            candidates=historical,
            opportunity_risk=PassThroughOpportunityRiskPolicy(),
            portfolio_policy=portfolio_policy,
            portfolio_risk=PassThroughPortfolioRiskPolicy(),
            entry_liquidity=context.entry_liquidity,
            max_entry_day_participation_pct=context.max_entry_day_participation_pct,
        )
        year_results.append(HistoricalYearResult(split.test_year, backtest))
        trade_ids.extend(trade.candidate_id for trade in backtest.trades)
        horizon_counts.update(trade.horizon_sessions for trade in backtest.trades)
        rejected_entry_liquidity += execution.rejected_entry_liquidity

    year_results_tuple = tuple(year_results)
    summary = summarize_historical_years(year_results_tuple)
    scorecard = summary.scorecard
    result = StrategyVariantResult(
        spec=spec,
        compounded_return=scorecard.compounded_return,
        profit_factor=scorecard.profit_factor,
        worst_realized_drawdown=scorecard.worst_realized_drawdown,
        total_trades=scorecard.total_trades,
        profitable_year_rate=scorecard.profitable_year_rate,
        average_trade_alpha=scorecard.average_trade_alpha,
        compounded_return_excluding_best_year=summary.compounded_return_excluding_best_year,
        best_year=summary.best_year,
        yearly_returns={
            item.year: item.backtest.total_return for item in summary.year_results
        },
        trade_candidate_ids=tuple(sorted(set(trade_ids))),
        trade_horizon_counts=dict(sorted(horizon_counts.items())),
    )
    execution_diagnostics = {
        "source_row_count": len(context.rows),
        "quality_removed_row_count": quality_removed,
        "pit_liquidity_removed_row_count": liquidity_removed,
        "executable_row_count": len(executable_rows),
        "rejected_entry_liquidity": rejected_entry_liquidity,
    }
    return result, year_results_tuple, execution_diagnostics


def _assert_execution_diagnostics(
    expected: Mapping[str, Any],
    actual: Mapping[str, int],
) -> None:
    required = (
        "source_row_count",
        "quality_removed_row_count",
        "pit_liquidity_removed_row_count",
        "executable_row_count",
        "rejected_entry_liquidity",
    )
    for key in required:
        if key not in expected:
            raise ValueError(f"screening result missing execution diagnostic {key}")
        if int(expected[key]) != int(actual[key]):
            raise ValueError(
                f"execution diagnostic mismatch for {key}: "
                f"screening={expected[key]!r}, replay={actual[key]!r}"
            )


def _compact_console(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload["schema_version"],
        "generation_id": payload["generation_id"],
        "all_finalists_exactly_reproduced": payload[
            "all_finalists_exactly_reproduced"
        ],
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
            "Retrain and exactly qualify execution-realistic G002 strategy finalists, "
            "including trade/concentration and fill-replay diagnostics."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--market-db", type=Path, required=True)
    parser.add_argument("--benchmark-security-id", default="benchmark_spy")
    parser.add_argument("--generation-id", default="g002")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--market-read-cache-series", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser


def main() -> None:
    args = _parser().parse_args()
    output_path = run_lightgbm_strategy_qualification_executable(
        args.experiment_dir,
        market_db=args.market_db,
        benchmark_security_id=args.benchmark_security_id,
        generation_id=args.generation_id,
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        market_read_cache_series=args.market_read_cache_series,
        tolerance=args.tolerance,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    print(json.dumps(_compact_console(payload), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
