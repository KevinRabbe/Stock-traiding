from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stock_trading.engine import (
    FixedAllocationPortfolioPolicy,
    PassThroughOpportunityRiskPolicy,
    PassThroughPortfolioRiskPolicy,
)
from stock_trading.market import DuckDbMarketStore
from stock_trading.ml import LightGbmTrainer, ProfitLightGbmTrainer
from stock_trading.ml.multi_horizon import row_for_horizon
from stock_trading.ml.walk_forward import annual_walk_forward_splits
from stock_trading.research import HistoricalYearResult, summarize_historical_years
from stock_trading.research.execution_realism import (
    ExecutionRealisticHistoricalBacktester,
    HistoricalExecutionLiquidity,
    load_market_quality_exclusions,
    target_overlaps_exclusion,
    trailing_adv_supports,
)
from stock_trading.research.strategy_factory import (
    PopulationGate,
    StrategyVariantResult,
    StrategyVariantSpec,
    apply_feature_profile,
    design_space_size,
    generate_population,
    select_diverse_finalists,
    training_window_rows,
)
from stock_trading.strategies.v5_adaptive_horizon import (
    V5AdaptiveHorizonStrategy,
    V5CalibrationState,
    V5HorizonModels,
    V5StrategyConfig,
)

from . import lightgbm_strategy_factory as base
from .lightgbm_validation_rank import _json_safe


@dataclass(frozen=True, slots=True)
class _ExecutablePreparedData:
    rows: tuple
    targets: dict
    security_ids: dict[str, str]
    entry_liquidity: dict[str, HistoricalExecutionLiquidity]
    invalid_target_keys: frozenset[tuple[str, int]]
    market_cache_stats: dict[str, int]
    quality_exclusion_count: int


@dataclass(slots=True)
class _ExecutableWorkerContext:
    rows: tuple
    targets: dict
    security_ids: dict[str, str]
    entry_liquidity: dict[str, HistoricalExecutionLiquidity]
    invalid_target_keys: frozenset[tuple[str, int]]
    starting_capital: float
    allocation_pct: float
    max_open_positions: int
    round_trip_cost_bps: float
    min_train_rows: int
    max_trailing_adv_participation_pct: float
    max_entry_day_participation_pct: float


@dataclass(frozen=True, slots=True)
class _ExecutableEvaluation:
    result: StrategyVariantResult
    source_row_count: int
    quality_removed_row_count: int
    pit_liquidity_removed_row_count: int
    executable_row_count: int
    rejected_entry_liquidity: int


_CONTEXT: _ExecutableWorkerContext | None = None


def run_lightgbm_strategy_factory_executable(
    experiment_dir: str | Path,
    *,
    market_db: str | Path,
    benchmark_security_id: str,
    generation_id: str = "g002",
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
) -> base.StrategyFactoryGenerationResult:
    """Run the strategy population with explicit data-quality and fill realism.

    G001 remains an immutable prediction/strategy screen. This runner uses the same
    deterministic strategy design space but constrains the research universe to
    opportunities executable at the reference account size. A PIT trailing-ADV
    gate is applied before model feature profiles; hidden execution-day volume is
    then used only by the historical execution adapter to require a full fill.
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

    prepared = _prepare_executable_data(
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

    completed: list[_ExecutableEvaluation] = []
    failures: list[dict[str, str]] = []
    try:
        if workers == 1:
            _initialize_worker(common, prepared)
            for spec in specs:
                try:
                    completed.append(_evaluate_variant(spec))
                except Exception as exc:
                    failures.append(
                        {"variant_id": spec.variant_id, "error": f"{type(exc).__name__}: {exc}"}
                    )
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_worker,
                initargs=(common, prepared),
            ) as executor:
                futures = {executor.submit(_evaluate_variant, spec): spec for spec in specs}
                for future in as_completed(futures):
                    spec = futures[future]
                    try:
                        completed.append(future.result())
                    except Exception as exc:
                        failures.append(
                            {"variant_id": spec.variant_id, "error": f"{type(exc).__name__}: {exc}"}
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
                "execution_diagnostics": _execution_diagnostics(evaluation),
            }
        )

    payload = _json_safe(
        {
            "schema_version": "lightgbm-strategy-factory-executable-v1",
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
                    "execution_diagnostics": _execution_diagnostics(item),
                }
                for item in completed
            ],
            "tested_specs": [spec.as_json() for spec in specs],
        }
    )
    output_path = generation_root / "report.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return base.StrategyFactoryGenerationResult(
        generation_id=generation_id,
        output_path=output_path,
        attempted=len(specs),
        completed=len(completed),
        failed=len(failures),
    )


def _prepare_executable_data(
    root: Path,
    *,
    market_db: str | Path,
    benchmark_security_id: str,
    market_quality_manifest: Path,
    market_read_cache_series: int,
) -> _ExecutablePreparedData:
    prepared = base._prepare_factory_data(
        root,
        market_db=market_db,
        benchmark_security_id=benchmark_security_id,
        market_read_cache_series=market_read_cache_series,
    )
    exclusions = load_market_quality_exclusions(market_quality_manifest)

    market_store = DuckDbMarketStore(market_db)
    market_store.enable_read_cache(max_series=market_read_cache_series)
    entry_liquidity: dict[str, HistoricalExecutionLiquidity] = {}
    invalid_target_keys: set[tuple[str, int]] = set()

    for row in prepared.rows:
        security_id = prepared.security_ids[row.event_id]
        entry_bar = market_store.bar_on(security_id, row.execution_date)
        if entry_bar is None:
            raise RuntimeError(
                f"missing execution-day market bar for {row.event_id} on {row.execution_date}"
            )
        entry_liquidity[row.event_id] = HistoricalExecutionLiquidity(
            candidate_id=row.event_id,
            entry_price=float(entry_bar.open),
            entry_volume=float(entry_bar.volume),
        )
        for horizon, target in prepared.targets[row.event_id].items():
            if target_overlaps_exclusion(
                security_id,
                row.execution_date,
                target.exit_date,
                exclusions,
            ):
                invalid_target_keys.add((row.event_id, int(horizon)))

    return _ExecutablePreparedData(
        rows=prepared.rows,
        targets=prepared.targets,
        security_ids=prepared.security_ids,
        entry_liquidity=entry_liquidity,
        invalid_target_keys=frozenset(invalid_target_keys),
        market_cache_stats=market_store.read_cache_stats(),
        quality_exclusion_count=len(exclusions),
    )


def _initialize_worker(common: dict[str, Any], prepared: _ExecutablePreparedData) -> None:
    global _CONTEXT
    _CONTEXT = _ExecutableWorkerContext(
        rows=prepared.rows,
        targets=prepared.targets,
        security_ids=prepared.security_ids,
        entry_liquidity=prepared.entry_liquidity,
        invalid_target_keys=prepared.invalid_target_keys,
        starting_capital=float(common["starting_capital"]),
        allocation_pct=float(common["allocation_pct"]),
        max_open_positions=int(common["max_open_positions"]),
        round_trip_cost_bps=float(common["round_trip_cost_bps"]),
        min_train_rows=int(common["min_train_rows"]),
        max_trailing_adv_participation_pct=float(common["max_trailing_adv_participation_pct"]),
        max_entry_day_participation_pct=float(common["max_entry_day_participation_pct"]),
    )


def _filter_executable_rows(
    rows: tuple,
    spec: StrategyVariantSpec,
    context: _ExecutableWorkerContext,
) -> tuple[tuple, int, int]:
    quality_valid = tuple(
        row
        for row in rows
        if not any((row.event_id, horizon) in context.invalid_target_keys for horizon in spec.horizons)
    )
    quality_removed = len(rows) - len(quality_valid)
    required_capital = context.starting_capital * context.allocation_pct
    executable = tuple(
        row
        for row in quality_valid
        if trailing_adv_supports(
            row.features,
            required_capital=required_capital,
            max_participation_pct=context.max_trailing_adv_participation_pct,
        )
    )
    liquidity_removed = len(quality_valid) - len(executable)
    return executable, quality_removed, liquidity_removed


def _evaluate_variant(spec: StrategyVariantSpec) -> _ExecutableEvaluation:
    if _CONTEXT is None:
        raise RuntimeError("executable strategy factory worker was not initialized")
    context = _CONTEXT

    executable_rows, quality_removed, liquidity_removed = _filter_executable_rows(
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
                row_for_horizon(row, context.targets[row.event_id][horizon]) for row in train_rows
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

        validation_candidates = base._feature_snapshots(split.validation_rows, context.security_ids)
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
        historical = base._historical_candidates(
            split.test_rows,
            context.targets,
            context.security_ids,
        )
        backtest, diagnostics = backtester.run(
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
        rejected_entry_liquidity += diagnostics.rejected_entry_liquidity

    summary = summarize_historical_years(tuple(year_results))
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
        yearly_returns={item.year: item.backtest.total_return for item in summary.year_results},
        trade_candidate_ids=tuple(sorted(set(trade_ids))),
        trade_horizon_counts=dict(sorted(horizon_counts.items())),
    )
    return _ExecutableEvaluation(
        result=result,
        source_row_count=len(context.rows),
        quality_removed_row_count=quality_removed,
        pit_liquidity_removed_row_count=liquidity_removed,
        executable_row_count=len(executable_rows),
        rejected_entry_liquidity=rejected_entry_liquidity,
    )


def _execution_diagnostics(item: _ExecutableEvaluation) -> dict[str, int]:
    return {
        "source_row_count": item.source_row_count,
        "quality_removed_row_count": item.quality_removed_row_count,
        "pit_liquidity_removed_row_count": item.pit_liquidity_removed_row_count,
        "executable_row_count": item.executable_row_count,
        "rejected_entry_liquidity": item.rejected_entry_liquidity,
    }


def _compact_console(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload["schema_version"],
        "generation": payload["generation"],
        "execution_realism": payload["execution_realism"],
        "eligible_count": payload["selection_policy"]["eligible_count"],
        "finalists": [
            {
                "variant_id": item["variant_id"],
                "selection_score": item["selection_score"],
                "scorecard": item["scorecard"],
                "execution_diagnostics": item["execution_diagnostics"],
                "spec": item["spec"],
            }
            for item in payload["finalists"]
        ],
        "failure_count": len(payload["failures"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and walk-forward test a fresh LightGBM strategy population "
            "with PIT liquidity, verified market-quality exclusions, and entry-day fill realism."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--market-db", type=Path, required=True)
    parser.add_argument("--benchmark-security-id", default="benchmark_spy")
    parser.add_argument("--generation-id", default="g002")
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
    result = run_lightgbm_strategy_factory_executable(
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
