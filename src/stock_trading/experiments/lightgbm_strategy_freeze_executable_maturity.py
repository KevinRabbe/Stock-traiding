from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from stock_trading.engine import (
    FileStrategyMetadataStore,
    FixedAllocationPortfolioPolicy,
    PassThroughOpportunityRiskPolicy,
    PassThroughPortfolioRiskPolicy,
    StrategyRecord,
    StrategyRegistrySnapshot,
    StrategyScorecard,
    StrategyStage,
    build_strategy_artifact_manifest,
    verify_strategy_artifact_manifest,
    write_strategy_artifact_manifest,
)
from stock_trading.ml import LightGbmTrainer, ProfitLightGbmTrainer
from stock_trading.ml.multi_horizon import multi_horizon_maturity_dates, row_for_horizon
from stock_trading.ml.walk_forward import annual_walk_forward_splits
from stock_trading.research.execution_realism import ExecutionRealisticHistoricalBacktester
from stock_trading.research.strategy_factory import (
    StrategyVariantSpec,
    apply_feature_profile,
    training_window_rows,
)
from stock_trading.strategies.frozen_factory import (
    load_frozen_factory_strategy,
    load_frozen_factory_strategy_from_manifest,
    write_frozen_factory_strategy,
)
from stock_trading.strategies.v5_adaptive_horizon import (
    V5CalibrationState,
    V5HorizonModels,
    V5StrategyConfig,
)

from . import lightgbm_strategy_factory as base_factory
from . import lightgbm_strategy_factory_executable as executable_factory
from . import lightgbm_strategy_factory_executable_maturity as maturity_factory
from . import lightgbm_strategy_qualify as base_qualify
from . import lightgbm_strategy_qualify_executable_maturity as maturity_qualify


FACTORY_SCHEMA = maturity_factory.SCHEMA_VERSION
QUALIFICATION_SCHEMA = maturity_qualify.SCHEMA_VERSION
DEFAULT_GENERATION_ID = maturity_factory.DEFAULT_GENERATION_ID
DEFAULT_FINALIST_COUNT = 3
MATURITY_FENCE = "latest_requested_horizon_exit_before_test_year"


@dataclass(frozen=True, slots=True)
class FrozenShadowStrategyResult:
    strategy_id: str
    model_year: int
    model_root: Path
    manifest_path: Path
    verified_test_return: float
    verified_test_trade_count: int


@dataclass(frozen=True, slots=True)
class FreezeShadowFinalistsResult:
    generation_id: str
    champion_id: str
    registry_path: Path
    summary_path: Path
    strategies: tuple[FrozenShadowStrategyResult, ...]


def freeze_maturity_safe_finalists(
    experiment_dir: str | Path,
    *,
    market_db: str | Path,
    benchmark_security_id: str,
    runtime_dir: str | Path = "data/runtime",
    generation_id: str = DEFAULT_GENERATION_ID,
    model_year: int = 2026,
    variant_ids: Sequence[str] = (),
    finalist_count: int = DEFAULT_FINALIST_COUNT,
    market_read_cache_series: int = 200,
    tolerance: float = 1e-12,
) -> FreezeShadowFinalistsResult:
    """Freeze clean G002m finalists as immutable SHADOW strategy artifacts.

    Every frozen model is the exact annual walk-forward model used for
    ``model_year`` under G002m's full-horizon maturity fence. The serialized
    artifact is loaded back from disk and must reproduce that year's exact trade
    identity and return before any artifact is published or registry state changes.
    The persisted PAPER champion is mandatory and is never changed by this command.
    """

    if not generation_id.strip():
        raise ValueError("generation_id must not be empty")
    if model_year <= 0:
        raise ValueError("model_year must be positive")
    if finalist_count <= 0:
        raise ValueError("finalist_count must be > 0")
    if market_read_cache_series <= 0:
        raise ValueError("market_read_cache_series must be > 0")
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")

    root = Path(experiment_dir)
    generation_root = root / "strategy_factory" / generation_id
    report_path = generation_root / "report.json"
    qualification_path = generation_root / "qualification.json"
    report = _read_json(report_path, "strategy factory report")
    qualification = _read_json(qualification_path, "strategy qualification")
    _validate_sources(report, qualification, generation_id)

    selected = _select_qualified_variants(
        report,
        qualification,
        variant_ids=tuple(variant_ids),
        finalist_count=finalist_count,
    )

    runtime_root = Path(runtime_dir)
    registry_path = runtime_root / "strategy_registry.json"
    metadata_store = FileStrategyMetadataStore(registry_path)
    existing = metadata_store.load()
    if existing is None or existing.champion_id is None:
        raise RuntimeError(
            "runtime has no persisted champion; bootstrap the V5 PAPER champion "
            "before registering SHADOW finalists"
        )
    champion_id = existing.champion_id
    existing_records = {item.strategy_id: item for item in existing.records}
    _preflight_registry(existing_records, selected)

    models_parent = runtime_root / "models"
    artifacts_parent = runtime_root / "artifacts"
    models_parent.mkdir(parents=True, exist_ok=True)
    artifacts_parent.mkdir(parents=True, exist_ok=True)
    for item in selected:
        strategy_id = str(item["variant_id"])
        final_root = models_parent / strategy_id
        temp_root = models_parent / f".{strategy_id}.freeze-tmp"
        manifest_path = artifacts_parent / f"{strategy_id}.json"
        if final_root.exists():
            raise FileExistsError(f"strategy model root already exists: {final_root}")
        if temp_root.exists():
            raise FileExistsError(f"stale strategy freeze directory exists: {temp_root}")
        if manifest_path.exists():
            raise FileExistsError(f"strategy artifact manifest already exists: {manifest_path}")

    realism = report["execution_realism"]
    portfolio = report["portfolio_policy"]
    quality_manifest = Path(str(realism["market_quality_manifest"]))
    common = {
        "starting_capital": float(portfolio["starting_capital"]),
        "allocation_pct": float(portfolio["allocation_pct"]),
        "max_open_positions": int(portfolio["max_open_positions"]),
        "round_trip_cost_bps": float(portfolio["round_trip_cost_bps"]),
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
    _verify_realism_identity(realism, prepared)
    executable_factory._initialize_worker(common, prepared)

    verified_temps: list[dict[str, Any]] = []
    published_roots: list[Path] = []
    published_manifests: list[Path] = []
    try:
        # Phase 1: train, serialize, reload, and exact-replay every finalist while
        # everything still lives in disposable temporary directories.
        for selected_item in selected:
            strategy_id = str(selected_item["variant_id"])
            screening = selected_item["screening"]
            qualified = selected_item["qualification"]
            spec = base_qualify._spec_from_json(screening["spec"])
            temp_root = models_parent / f".{strategy_id}.freeze-tmp"
            verification = _freeze_one(
                spec,
                screening=screening,
                qualification=qualified,
                report=report,
                model_year=model_year,
                temp_root=temp_root,
                tolerance=tolerance,
            )
            verified_temps.append(
                {
                    "strategy_id": strategy_id,
                    "temp_root": temp_root,
                    "verification": verification,
                }
            )

        # Phase 2: only after the entire cohort verified do immutable roots and
        # manifests become visible. Registry state still remains unchanged.
        frozen: list[FrozenShadowStrategyResult] = []
        for item in verified_temps:
            strategy_id = str(item["strategy_id"])
            temp_root = Path(item["temp_root"])
            final_root = models_parent / strategy_id
            temp_root.replace(final_root)
            published_roots.append(final_root)

            manifest = build_strategy_artifact_manifest(strategy_id, final_root)
            manifest_path = artifacts_parent / f"{strategy_id}.json"
            write_strategy_artifact_manifest(manifest, manifest_path)
            published_manifests.append(manifest_path)
            verify_strategy_artifact_manifest(manifest)
            loaded = load_frozen_factory_strategy_from_manifest(manifest_path)
            if loaded.strategy_id != strategy_id:
                raise RuntimeError("verified frozen artifact loaded a foreign strategy_id")

            verification = item["verification"]
            frozen.append(
                FrozenShadowStrategyResult(
                    strategy_id=strategy_id,
                    model_year=model_year,
                    model_root=final_root,
                    manifest_path=manifest_path,
                    verified_test_return=float(verification["verified_test_return"]),
                    verified_test_trade_count=int(
                        verification["verified_test_trade_count"]
                    ),
                )
            )

        records = dict(existing_records)
        selected_by_id = {str(item["variant_id"]): item for item in selected}
        for item in frozen:
            source = selected_by_id[item.strategy_id]
            qualified = source["qualification"]
            records[item.strategy_id] = StrategyRecord(
                strategy_id=item.strategy_id,
                stage=StrategyStage.SHADOW,
                artifact_ref=str(item.manifest_path),
                scorecard=_scorecard(qualified["scorecard"]),
                selection_score=(
                    float(source["finalist"].get("selection_score"))
                    if source["finalist"].get("selection_score") is not None
                    else None
                ),
                notes=(
                    f"Frozen from {generation_id} exact maturity-safe executable "
                    f"qualification; annual walk-forward model_year={model_year}"
                ),
            )
        metadata_store.save(
            StrategyRegistrySnapshot(
                champion_id=champion_id,
                records=tuple(records[key] for key in sorted(records)),
            )
        )

        summary_path = runtime_root / f"shadow_finalists_{generation_id}.json"
        summary = {
            "schema_version": "frozen-shadow-finalists-maturity-v1",
            "generation_id": generation_id,
            "champion_id": champion_id,
            "champion_unchanged": True,
            "stage": StrategyStage.SHADOW.value,
            "model_year": model_year,
            "full_horizon_maturity_required": True,
            "maturity_fence": MATURITY_FENCE,
            "source_report": str(report_path),
            "source_qualification": str(qualification_path),
            "strategies": [
                {
                    "strategy_id": item.strategy_id,
                    "model_root": str(item.model_root),
                    "manifest_path": str(item.manifest_path),
                    "verified_test_return": item.verified_test_return,
                    "verified_test_trade_count": item.verified_test_trade_count,
                }
                for item in frozen
            ],
        }
        _write_json(summary_path, summary)
        return FreezeShadowFinalistsResult(
            generation_id=generation_id,
            champion_id=champion_id,
            registry_path=registry_path,
            summary_path=summary_path,
            strategies=tuple(frozen),
        )
    except Exception:
        # Registry is written only after all artifacts are verified. Clean every
        # path created by this invocation so a failed freeze can be retried safely.
        for item in selected:
            strategy_id = str(item["variant_id"])
            temp_root = models_parent / f".{strategy_id}.freeze-tmp"
            if temp_root.exists():
                shutil.rmtree(temp_root)
        for path in reversed(published_manifests):
            if path.exists():
                path.unlink()
        for path in reversed(published_roots):
            if path.exists():
                shutil.rmtree(path)
        raise


def _freeze_one(
    spec: StrategyVariantSpec,
    *,
    screening: Mapping[str, Any],
    qualification: Mapping[str, Any],
    report: Mapping[str, Any],
    model_year: int,
    temp_root: Path,
    tolerance: float,
) -> dict[str, Any]:
    context = executable_factory._CONTEXT
    if context is None:
        raise RuntimeError("executable factory context was not initialized")

    executable_rows, quality_removed, liquidity_removed = (
        executable_factory._filter_executable_rows(context.rows, spec, context)
    )
    maturity_dates = multi_horizon_maturity_dates(
        executable_rows,
        context.targets,
        horizons=spec.horizons,
    )
    maturity_rows = tuple(
        replace(row, exit_date_20d=maturity_dates[row.event_id])
        for row in executable_rows
    )
    rows = apply_feature_profile(maturity_rows, spec.feature_profile)
    splits = annual_walk_forward_splits(rows)
    split = next((item for item in splits if item.test_year == model_year), None)
    if split is None:
        raise ValueError(
            f"{spec.variant_id} has no maturity-safe annual split for model_year {model_year}"
        )
    train_rows = training_window_rows(
        split.train_rows,
        test_year=split.test_year,
        window_years=spec.training_window_years,
    )
    if len(train_rows) < context.min_train_rows:
        raise ValueError(
            f"{spec.variant_id} has only {len(train_rows)} deployment training rows"
        )

    profitable_threshold = context.round_trip_cost_bps / 10_000.0
    profit_trainer = ProfitLightGbmTrainer(spec.training_config)
    alpha_trainer = LightGbmTrainer(spec.training_config)
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

    config = V5StrategyConfig(
        strategy_id=spec.variant_id,
        horizons=spec.horizons,
        validation_top_fraction=spec.validation_top_fraction,
        alpha_rank_weight=spec.alpha_rank_weight,
        calibration_window_days=spec.calibration_window_days,
        min_expected_return=profitable_threshold,
        max_expected_downside=spec.max_expected_downside,
    )
    validation_candidates = base_factory._feature_snapshots(
        split.validation_rows,
        context.security_ids,
    )
    calibration = V5CalibrationState.from_validation(
        validation_candidates,
        models,
        config,
    )

    expected_year_return = float(screening["yearly_returns"][str(model_year)])
    test_ids = {row.event_id for row in split.test_rows}
    expected_trade_ids = tuple(
        sorted(
            candidate_id
            for candidate_id in screening.get("trade_candidate_ids") or ()
            if candidate_id in test_ids
        )
    )
    source = {
        "generation_id": report["generation"]["generation_id"],
        "factory_schema": report["schema_version"],
        "qualification_schema": QUALIFICATION_SCHEMA,
        "variant_spec": spec.as_json(),
        "qualification_scorecard": qualification["scorecard"],
        "qualification_flags": qualification["qualification_flags"],
        "qualification_diagnostics": qualification["diagnostics"],
        "execution_realism": report["execution_realism"],
        "full_horizon_maturity_required": True,
        "maturity_fence": MATURITY_FENCE,
    }
    training = {
        "annual_walk_forward_identity": True,
        "model_year": model_year,
        "train_count": len(train_rows),
        "validation_count": len(split.validation_rows),
        "test_count": len(split.test_rows),
        "quality_removed_row_count": quality_removed,
        "pit_liquidity_removed_row_count": liquidity_removed,
        "expected_test_return": expected_year_return,
        "expected_test_trade_count": len(expected_trade_ids),
        "full_horizon_maturity_required": True,
        "maturity_fence": MATURITY_FENCE,
    }
    write_frozen_factory_strategy(
        temp_root,
        strategy_id=spec.variant_id,
        model_year=model_year,
        models=models,
        calibration=calibration,
        config=config,
        source=source,
        training=training,
    )

    serialized_strategy = load_frozen_factory_strategy(temp_root)
    portfolio_policy = FixedAllocationPortfolioPolicy(
        allocation_pct=context.allocation_pct,
        max_open_positions=context.max_open_positions,
        max_gross_exposure_pct=1.0,
        one_position_per_company=True,
    )
    historical = base_factory._historical_candidates(
        split.test_rows,
        context.targets,
        context.security_ids,
    )
    backtester = ExecutionRealisticHistoricalBacktester(
        starting_capital=context.starting_capital,
        round_trip_cost_bps=context.round_trip_cost_bps,
    )
    replay, execution = backtester.run(
        strategy=serialized_strategy,
        candidates=historical,
        opportunity_risk=PassThroughOpportunityRiskPolicy(),
        portfolio_policy=portfolio_policy,
        portfolio_risk=PassThroughPortfolioRiskPolicy(),
        entry_liquidity=context.entry_liquidity,
        max_entry_day_participation_pct=context.max_entry_day_participation_pct,
    )
    actual_trade_ids = tuple(sorted(trade.candidate_id for trade in replay.trades))
    if actual_trade_ids != expected_trade_ids:
        raise ValueError(
            f"{spec.variant_id} frozen {model_year} trade identity diverged from G002m"
        )
    if abs(float(replay.total_return) - expected_year_return) > tolerance:
        raise ValueError(
            f"{spec.variant_id} frozen {model_year} return diverged from G002m: "
            f"{replay.total_return} != {expected_year_return}"
        )

    screening_execution = screening.get("execution_diagnostics") or {}
    expected_rejected_total = int(screening_execution.get("rejected_entry_liquidity", 0))
    # The screening count is generation-wide; exact annual replay identity is
    # already enforced through candidate IDs/return. Keep the annual rejection
    # count as artifact evidence without comparing unlike scopes.
    return {
        "verified_test_return": float(replay.total_return),
        "verified_test_trade_count": len(replay.trades),
        "verified_test_rejected_entry_liquidity": int(
            execution.rejected_entry_liquidity
        ),
        "screening_total_rejected_entry_liquidity": expected_rejected_total,
    }


def _validate_sources(
    report: Mapping[str, Any],
    qualification: Mapping[str, Any],
    generation_id: str,
) -> None:
    if report.get("schema_version") != FACTORY_SCHEMA:
        raise ValueError("freeze requires a G002m maturity-safe factory report")
    generation = report.get("generation") or {}
    if generation.get("generation_id") != generation_id:
        raise ValueError("factory report generation_id mismatch")
    if int(generation.get("failed_hypotheses", -1)) != 0:
        raise ValueError("freeze requires a generation with zero failed hypotheses")
    realism = report.get("execution_realism") or {}
    if realism.get("enabled") is not True or realism.get("full_fill_required") is not True:
        raise ValueError("freeze requires execution realism/full-fill policy")
    if realism.get("return_cap_applied") is not False:
        raise ValueError("freeze refuses factory reports with a return cap")
    if realism.get("full_horizon_maturity_required") is not True:
        raise ValueError("freeze requires full-horizon target maturity")
    if realism.get("maturity_fence") != MATURITY_FENCE:
        raise ValueError("freeze requires the G002m maturity fence")

    if qualification.get("schema_version") != QUALIFICATION_SCHEMA:
        raise ValueError("freeze requires maturity-safe finalist qualification")
    if qualification.get("generation_id") != generation_id:
        raise ValueError("qualification generation_id mismatch")
    if qualification.get("all_finalists_exactly_reproduced") is not True:
        raise ValueError("not all finalists reproduced exactly")
    if qualification.get("full_horizon_maturity_required") is not True:
        raise ValueError("qualification does not require full-horizon maturity")
    qualified_realism = qualification.get("execution_realism") or {}
    for key in (
        "full_fill_required",
        "full_horizon_maturity_required",
        "invalid_target_count",
        "market_quality_manifest",
        "maturity_fence",
        "max_entry_day_participation_pct",
        "max_trailing_adv_participation_pct",
        "return_cap_applied",
        "verified_quality_exclusion_count",
    ):
        if qualified_realism.get(key) != realism.get(key):
            raise ValueError(f"qualification execution realism differs for {key}")


def _select_qualified_variants(
    report: Mapping[str, Any],
    qualification: Mapping[str, Any],
    *,
    variant_ids: tuple[str, ...],
    finalist_count: int,
) -> tuple[dict[str, Any], ...]:
    report_finalists = {
        str(item["variant_id"]): item for item in report.get("finalists") or ()
    }
    screening_by_id = {
        str(item["spec"]["variant_id"]): item for item in report.get("results") or ()
    }
    qualified_order = list(qualification.get("finalists") or ())
    qualified_by_id = {
        str(item["variant_id"]): item for item in qualified_order
    }

    if variant_ids:
        if len(set(variant_ids)) != len(variant_ids):
            raise ValueError("variant_ids must be unique")
        selected_ids = list(variant_ids)
    else:
        selected_ids = []
        for item in qualified_order:
            if _qualification_clean(item):
                selected_ids.append(str(item["variant_id"]))
            if len(selected_ids) >= finalist_count:
                break
    if not selected_ids:
        raise ValueError("no clean qualified finalists selected")

    selected: list[dict[str, Any]] = []
    for strategy_id in selected_ids:
        finalist = report_finalists.get(strategy_id)
        screening = screening_by_id.get(strategy_id)
        qualified = qualified_by_id.get(strategy_id)
        if finalist is None or screening is None or qualified is None:
            raise ValueError(
                f"selected strategy {strategy_id} is not a G002m qualified finalist"
            )
        if not _qualification_clean(qualified):
            raise ValueError(f"selected strategy {strategy_id} did not pass clean qualification")
        selected.append(
            {
                "variant_id": strategy_id,
                "finalist": finalist,
                "screening": screening,
                "qualification": qualified,
            }
        )
    return tuple(selected)


def _qualification_clean(item: Mapping[str, Any]) -> bool:
    if item.get("exact_screening_identity_verified") is not True:
        return False
    flags = item.get("qualification_flags") or {}
    if any(bool(value) for value in flags.values()):
        return False
    diagnostics = item.get("diagnostics") or {}
    ex_best_three = diagnostics.get("compounded_return_excluding_best_three_years")
    return ex_best_three is not None and float(ex_best_three) > 0.0


def _preflight_registry(
    existing_records: Mapping[str, StrategyRecord],
    selected: Sequence[Mapping[str, Any]],
) -> None:
    for item in selected:
        strategy_id = str(item["variant_id"])
        if strategy_id in existing_records:
            raise RuntimeError(
                f"runtime registry already contains {strategy_id}; freeze refuses overwrite"
            )


def _verify_realism_identity(realism: Mapping[str, Any], prepared) -> None:
    expected_quality = int(realism["verified_quality_exclusion_count"])
    if prepared.quality_exclusion_count != expected_quality:
        raise ValueError("market-quality exclusion count changed since G002m")
    expected_invalid = int(realism["invalid_target_count"])
    if len(prepared.invalid_target_keys) != expected_invalid:
        raise ValueError("invalid target count changed since G002m")


def _scorecard(value: Mapping[str, Any]) -> StrategyScorecard:
    return StrategyScorecard(
        compounded_return=float(value["compounded_return"]),
        profit_factor=float(value["profit_factor"]),
        worst_realized_drawdown=float(value["worst_realized_drawdown"]),
        total_trades=int(value["total_trades"]),
        profitable_year_rate=float(value["profitable_year_rate"]),
        average_trade_alpha=(
            float(value["average_trade_alpha"])
            if value.get("average_trade_alpha") is not None
            else None
        ),
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"missing {label}: {path}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze exactly-qualified G002m finalists as immutable maturity-safe "
            "annual-model SHADOW artifacts while preserving the PAPER champion."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--market-db", type=Path, required=True)
    parser.add_argument("--benchmark-security-id", default="benchmark_spy")
    parser.add_argument("--runtime-dir", type=Path, default=Path("data/runtime"))
    parser.add_argument("--generation-id", default=DEFAULT_GENERATION_ID)
    parser.add_argument("--model-year", type=int, default=2026)
    parser.add_argument("--variant-id", action="append", default=[])
    parser.add_argument("--finalist-count", type=int, default=DEFAULT_FINALIST_COUNT)
    parser.add_argument("--market-read-cache-series", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = freeze_maturity_safe_finalists(
        args.experiment_dir,
        market_db=args.market_db,
        benchmark_security_id=args.benchmark_security_id,
        runtime_dir=args.runtime_dir,
        generation_id=args.generation_id,
        model_year=args.model_year,
        variant_ids=tuple(args.variant_id),
        finalist_count=args.finalist_count,
        market_read_cache_series=args.market_read_cache_series,
        tolerance=args.tolerance,
    )
    print(
        json.dumps(
            {
                "generation_id": result.generation_id,
                "champion_id": result.champion_id,
                "champion_unchanged": True,
                "registry_path": str(result.registry_path),
                "summary_path": str(result.summary_path),
                "full_horizon_maturity_required": True,
                "maturity_fence": MATURITY_FENCE,
                "strategies": [
                    {
                        "strategy_id": item.strategy_id,
                        "stage": StrategyStage.SHADOW.value,
                        "model_year": item.model_year,
                        "model_root": str(item.model_root),
                        "manifest_path": str(item.manifest_path),
                        "verified_test_return": item.verified_test_return,
                        "verified_test_trade_count": item.verified_test_trade_count,
                    }
                    for item in result.strategies
                ],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
