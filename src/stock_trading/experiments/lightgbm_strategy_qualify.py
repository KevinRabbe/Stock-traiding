from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from math import prod
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from stock_trading.engine import (
    FixedAllocationPortfolioPolicy,
    PassThroughOpportunityRiskPolicy,
    PassThroughPortfolioRiskPolicy,
)
from stock_trading.ml import LightGbmTrainer, ProfitLightGbmTrainer
from stock_trading.ml.multi_horizon import row_for_horizon
from stock_trading.ml.walk_forward import annual_walk_forward_splits
from stock_trading.research import HistoricalStrategyBacktester, HistoricalYearResult, summarize_historical_years
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

from . import lightgbm_strategy_factory as factory
from .lightgbm_validation_rank import _json_safe


def run_lightgbm_strategy_qualification(
    experiment_dir: str | Path,
    *,
    market_db: str | Path,
    benchmark_security_id: str,
    generation_id: str = "g001",
    workers: int = 4,
    threads_per_worker: int = 2,
    market_read_cache_series: int = 200,
    tolerance: float = 1e-12,
) -> Path:
    """Retrain selected factory finalists and qualify their historical identity.

    Screening weights are deliberately discarded by the strategy factory. This
    command regenerates every selected finalist from its exact recorded spec,
    proves the scorecard/trade identity matches the screening run, and records
    concentration diagnostics before any model artifact is allowed into shadow.

    Market-dependent PIT preparation is shared with the strategy factory and is
    completed once in the parent process. Spawned workers never open DuckDB.
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
        spec = _spec_from_json(finalist["spec"])
        if spec.variant_id in finalist_by_id:
            raise ValueError(f"duplicate finalist {spec.variant_id}")
        if spec.variant_id not in screening_by_id:
            raise ValueError(f"finalist {spec.variant_id} missing full screening result")
        specs.append(spec)
        finalist_by_id[spec.variant_id] = finalist

    policy = report.get("portfolio_policy") or {}
    common = {
        "starting_capital": float(policy.get("starting_capital", 10_000.0)),
        "allocation_pct": float(policy.get("allocation_pct", 0.02)),
        "max_open_positions": int(policy.get("max_open_positions", 15)),
        "round_trip_cost_bps": float(policy.get("round_trip_cost_bps", 20.0)),
        "min_train_rows": 100,
    }
    prepared = factory._prepare_factory_data(
        root,
        market_db=market_db,
        benchmark_security_id=benchmark_security_id,
        market_read_cache_series=market_read_cache_series,
    )

    old_omp = os.environ.get("OMP_NUM_THREADS")
    old_openblas = os.environ.get("OPENBLAS_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)

    qualified: dict[str, dict[str, Any]] = {}
    try:
        if workers == 1:
            factory._initialize_worker(common, prepared)
            for spec in specs:
                qualified[spec.variant_id] = _qualify_variant(
                    spec,
                    screening_by_id[spec.variant_id],
                    tolerance=tolerance,
                )
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=factory._initialize_worker,
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
            "schema_version": "lightgbm-strategy-finalist-qualification-v1",
            "generation_id": generation_id,
            "source_report": str(report_path),
            "data": {
                "market_db": str(market_db),
                "benchmark_security_id": benchmark_security_id,
                "point_in_time": True,
                "prepared_row_count": len(prepared.rows),
                "worker_market_db_access": False,
                "screening_models_reused": False,
                "finalists_retrained_from_scratch": True,
            },
            "replay_policy": {
                "float_tolerance": tolerance,
                "requires_zero_generation_failures": True,
                "requires_exact_trade_identity": True,
                "requires_exact_horizon_counts": True,
                "requires_yearly_return_identity": True,
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
    result, year_results = _evaluate_variant_with_trades(spec)
    _assert_screening_identity(
        screening_result,
        result,
        tolerance=tolerance,
    )
    diagnostics = _concentration_diagnostics(year_results)
    return {
        "exact_screening_identity_verified": True,
        "spec": result.spec.as_json(),
        "scorecard": result.as_json()["scorecard"],
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
) -> tuple[StrategyVariantResult, tuple[HistoricalYearResult, ...]]:
    context = factory._CONTEXT
    if context is None:
        raise RuntimeError("strategy qualification worker was not initialized")

    rows = apply_feature_profile(context.rows, spec.feature_profile)
    splits = annual_walk_forward_splits(rows)
    if not splits:
        raise ValueError("no annual walk-forward splits")

    profitable_threshold = context.round_trip_cost_bps / 10_000.0
    portfolio_policy = FixedAllocationPortfolioPolicy(
        allocation_pct=context.allocation_pct,
        max_open_positions=context.max_open_positions,
        max_gross_exposure_pct=1.0,
        one_position_per_company=True,
    )
    backtester = HistoricalStrategyBacktester(
        starting_capital=context.starting_capital,
        round_trip_cost_bps=context.round_trip_cost_bps,
    )
    profit_trainer = ProfitLightGbmTrainer(spec.training_config)
    alpha_trainer = LightGbmTrainer(spec.training_config)
    year_results: list[HistoricalYearResult] = []
    trade_ids: list[str] = []
    horizon_counts: Counter[int] = Counter()

    for split in splits:
        train_rows = training_window_rows(
            split.train_rows,
            test_year=split.test_year,
            window_years=spec.training_window_years,
        )
        if len(train_rows) < context.min_train_rows:
            raise ValueError(
                f"{spec.variant_id} has only {len(train_rows)} training rows for "
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

        validation_candidates = factory._feature_snapshots(
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
        historical = factory._historical_candidates(
            split.test_rows,
            context.targets,
            context.security_ids,
        )
        backtest = backtester.run(
            strategy=strategy,
            candidates=historical,
            opportunity_risk=PassThroughOpportunityRiskPolicy(),
            portfolio_policy=portfolio_policy,
            portfolio_risk=PassThroughPortfolioRiskPolicy(),
        )
        year_results.append(HistoricalYearResult(split.test_year, backtest))
        trade_ids.extend(trade.candidate_id for trade in backtest.trades)
        horizon_counts.update(trade.horizon_sessions for trade in backtest.trades)

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
    return result, year_results_tuple


def _concentration_diagnostics(
    year_results: Sequence[HistoricalYearResult],
) -> dict[str, Any]:
    trades = [
        (item.year, trade)
        for item in year_results
        for trade in item.backtest.trades
    ]
    yearly_returns = {
        item.year: item.backtest.total_return for item in year_results
    }
    yearly_trade_counts = {
        item.year: len(item.backtest.trades) for item in year_results
    }

    positive_pnl = [trade.pnl for _, trade in trades if trade.pnl > 0]
    positive_total = sum(positive_pnl)
    positive_company: defaultdict[str, float] = defaultdict(float)
    positive_year: defaultdict[int, float] = defaultdict(float)
    company_counts: Counter[str] = Counter()
    gross_returns: list[float] = []
    for year, trade in trades:
        company_counts[trade.company_id] += 1
        gross_returns.append(trade.gross_return)
        if trade.pnl > 0:
            positive_company[trade.company_id] += trade.pnl
            positive_year[year] += trade.pnl

    positive_trade_sorted = sorted(positive_pnl, reverse=True)
    positive_company_sorted = sorted(positive_company.items(), key=lambda item: item[1], reverse=True)
    positive_year_sorted = sorted(positive_year.items(), key=lambda item: item[1], reverse=True)
    company_count_sorted = company_counts.most_common(10)

    best_three_years = tuple(
        year for year, _ in sorted(yearly_returns.items(), key=lambda item: item[1], reverse=True)[:3]
    )
    remaining_returns = [
        value for year, value in yearly_returns.items() if year not in best_three_years
    ]
    excluding_best_three = (
        prod(1.0 + value for value in remaining_returns) - 1.0
        if remaining_returns
        else None
    )
    worst_year = (
        min(yearly_returns, key=yearly_returns.get) if yearly_returns else None
    )
    best_year = (
        max(yearly_returns, key=yearly_returns.get) if yearly_returns else None
    )

    return {
        "trade_count": len(trades),
        "unique_company_count": len(company_counts),
        "largest_company_trade_count": company_count_sorted[0][1] if company_count_sorted else 0,
        "largest_company_trade_fraction": (
            company_count_sorted[0][1] / len(trades) if trades and company_count_sorted else None
        ),
        "top_company_trade_counts": [
            {"company_id": company_id, "trades": count}
            for company_id, count in company_count_sorted
        ],
        "largest_positive_trade_pnl_fraction": _fraction(
            positive_trade_sorted[0] if positive_trade_sorted else None,
            positive_total,
        ),
        "top_five_positive_trade_pnl_fraction": _fraction(
            sum(positive_trade_sorted[:5]) if positive_trade_sorted else None,
            positive_total,
        ),
        "largest_positive_company_pnl_fraction": _fraction(
            positive_company_sorted[0][1] if positive_company_sorted else None,
            positive_total,
        ),
        "top_five_positive_company_pnl_fraction": _fraction(
            sum(value for _, value in positive_company_sorted[:5])
            if positive_company_sorted
            else None,
            positive_total,
        ),
        "best_year_positive_pnl_fraction": _fraction(
            positive_year.get(best_year, 0.0) if best_year is not None else None,
            positive_total,
        ),
        "top_positive_pnl_years": [
            {"year": year, "positive_pnl": value}
            for year, value in positive_year_sorted[:5]
        ],
        "gross_return_distribution": {
            "min": min(gross_returns) if gross_returns else None,
            "median": median(gross_returns) if gross_returns else None,
            "p95": _percentile(gross_returns, 0.95),
            "max": max(gross_returns) if gross_returns else None,
        },
        "best_year": best_year,
        "best_year_return": yearly_returns.get(best_year) if best_year is not None else None,
        "worst_year": worst_year,
        "worst_year_return": yearly_returns.get(worst_year) if worst_year is not None else None,
        "best_three_years": list(best_three_years),
        "compounded_return_excluding_best_three_years": excluding_best_three,
        "yearly_trade_counts": {str(year): count for year, count in sorted(yearly_trade_counts.items())},
    }


def _assert_screening_identity(
    screening: Mapping[str, Any],
    result: StrategyVariantResult,
    *,
    tolerance: float,
) -> None:
    expected_spec = screening.get("spec")
    if expected_spec != result.spec.as_json():
        raise ValueError(f"{result.spec.variant_id} spec changed since screening")

    actual = result.as_json()
    expected_scorecard = screening.get("scorecard") or {}
    for key, actual_value in actual["scorecard"].items():
        expected_value = expected_scorecard.get(key)
        if not _values_equal(expected_value, actual_value, tolerance):
            raise ValueError(
                f"{result.spec.variant_id} scorecard mismatch for {key}: "
                f"screening={expected_value!r}, replay={actual_value!r}"
            )

    expected_years = screening.get("yearly_returns") or {}
    actual_years = actual["yearly_returns"]
    if set(expected_years) != set(actual_years):
        raise ValueError(f"{result.spec.variant_id} yearly return keys changed")
    for year, actual_value in actual_years.items():
        if not _values_equal(expected_years[year], actual_value, tolerance):
            raise ValueError(
                f"{result.spec.variant_id} yearly return mismatch for {year}: "
                f"screening={expected_years[year]!r}, replay={actual_value!r}"
            )

    if screening.get("trade_horizon_counts") != actual["trade_horizon_counts"]:
        raise ValueError(f"{result.spec.variant_id} horizon trade counts changed")
    if tuple(screening.get("trade_candidate_ids") or ()) != result.trade_candidate_ids:
        raise ValueError(f"{result.spec.variant_id} trade identity changed")


def _values_equal(left: Any, right: Any, tolerance: float) -> bool:
    if left is None or right is None:
        if left is None and isinstance(right, float) and not math.isfinite(right):
            return True
        if right is None and isinstance(left, float) and not math.isfinite(left):
            return True
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isfinite(float(left)) or not math.isfinite(float(right)):
            return float(left) == float(right)
        return abs(float(left) - float(right)) <= tolerance
    return left == right


def _fraction(numerator: float | None, denominator: float) -> float | None:
    if numerator is None or denominator <= 0:
        return None
    return numerator / denominator


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _spec_from_json(value: Mapping[str, Any]) -> StrategyVariantSpec:
    return StrategyVariantSpec(
        variant_id=str(value["variant_id"]),
        feature_profile=str(value["feature_profile"]),
        training_window_years=(
            None if value.get("training_window_years") is None else int(value["training_window_years"])
        ),
        tree_profile=str(value["tree_profile"]),
        horizons=tuple(int(item) for item in value["horizons"]),
        alpha_rank_weight=float(value["alpha_rank_weight"]),
        seed=int(value["seed"]),
        validation_top_fraction=float(value.get("validation_top_fraction", 0.05)),
        calibration_window_days=int(value.get("calibration_window_days", 365)),
        max_expected_downside=float(value.get("max_expected_downside", 0.06)),
    )


def _compact_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": payload["schema_version"],
        "generation_id": payload["generation_id"],
        "all_finalists_exactly_reproduced": payload["all_finalists_exactly_reproduced"],
        "qualification_path": str(path),
        "finalists": [
            {
                "variant_id": item["variant_id"],
                "selection_score": item.get("selection_score"),
                "scorecard": item["scorecard"],
                "diagnostics": item["diagnostics"],
                "qualification_flags": item["qualification_flags"],
                "spec": item["spec"],
            }
            for item in payload["finalists"]
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain strategy-factory finalists from scratch, prove exact screening "
            "identity, and measure trade/company/year concentration before shadow."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--market-db", type=Path, required=True)
    parser.add_argument("--benchmark-security-id", default="benchmark_spy")
    parser.add_argument("--generation-id", default="g001")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--market-read-cache-series", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser


def main() -> None:
    args = _parser().parse_args()
    path = run_lightgbm_strategy_qualification(
        args.experiment_dir,
        market_db=args.market_db,
        benchmark_security_id=args.benchmark_security_id,
        generation_id=args.generation_id,
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        market_read_cache_series=args.market_read_cache_series,
        tolerance=args.tolerance,
    )
    print(json.dumps(_compact_payload(path), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
