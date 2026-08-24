from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from stock_trading.engine import (
    BasicOpportunityRiskPolicy,
    FixedAllocationPortfolioPolicy,
    PassThroughPortfolioRiskPolicy,
)
from stock_trading.ml import LightGbmTrainer, ProfitLightGbmTrainer
from stock_trading.ml.multi_horizon import row_for_horizon
from stock_trading.research.execution_realism import (
    ExecutionRealisticHistoricalBacktester,
    trailing_adv_supports,
)
from stock_trading.research.strategy_factory import (
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
from . import lightgbm_strategy_factory_executable as executable
from .lightgbm_validation_rank import _json_safe
from .v5_profit_proof import (
    DEFAULT_MARKET_QUALITY_MANIFEST,
    DEFAULT_MAX_GROSS_EXPOSURE_PCT,
    _AnnualV5StrategyRouter,
    _resolve_market_inputs,
    current_v5_profit_proof_spec,
    minimum_viability,
)


SCHEMA_VERSION = "v5-horizon-quality-profit-proof-v1"


@dataclass(frozen=True, slots=True)
class V5HorizonQualityProfitProofResult:
    output_path: Path
    best_scenario: str
    best_return: float


@dataclass(frozen=True, slots=True)
class _ScenarioReplay:
    name: str
    strategies: dict[int, V5AdaptiveHorizonStrategy]
    candidates: tuple
    split_diagnostics: tuple[dict[str, Any], ...]
    salvaged_label_count: int
    quality_unscorable_candidate_count: int
    adv_candidate_removed_count: int


def run_v5_horizon_quality_profit_proof(
    experiment_dir: str | Path,
    *,
    runtime_dir: str | Path = "data/runtime",
    market_db: str | Path | None = None,
    benchmark_security_id: str | None = None,
    starting_capital: float = 10_000.0,
    allocation_pct: float = 0.02,
    max_open_positions: int = 15,
    max_gross_exposure_pct: float = DEFAULT_MAX_GROSS_EXPOSURE_PCT,
    round_trip_cost_bps: float = 20.0,
    max_trailing_adv_participation_pct: float = 0.01,
    max_entry_day_participation_pct: float = 0.01,
    market_quality_manifest: str | Path = DEFAULT_MARKET_QUALITY_MANIFEST,
    min_train_rows: int = 100,
    market_read_cache_series: int = 200,
) -> V5HorizonQualityProfitProofResult:
    """Test V5 with horizon-local quality/maturity instead of row-wide deletion.

    This remains a fixed-spec falsification test. It does not search parameters or
    alter PAPER thresholds. The only controlled changes are data-eligibility rules:

    - a label is excluded only from the horizon whose target overlaps a verified
      market-data defect;
    - each horizon's training/validation labels need only its own outcome to have
      matured before the test year;
    - one scenario keeps the legacy trailing-ADV training prefilter;
    - one scenario uses broad training data and applies trailing ADV only to
      calibration/test opportunities that could actually be traded.

    Test candidates with any corrupted requested horizon are omitted because their
    selected-horizon realized PnL cannot be evaluated honestly from the local data.
    """
    if starting_capital <= 0:
        raise ValueError("starting_capital must be > 0")
    if not 0.0 < allocation_pct <= 1.0:
        raise ValueError("allocation_pct must be in (0, 1]")
    if max_open_positions <= 0 or min_train_rows <= 0 or market_read_cache_series <= 0:
        raise ValueError("position/train/cache limits must be > 0")
    if not 0.0 < max_gross_exposure_pct <= 1.0:
        raise ValueError("max_gross_exposure_pct must be in (0, 1]")
    if round_trip_cost_bps < 0:
        raise ValueError("round_trip_cost_bps must be >= 0")
    for name, value in (
        ("max_trailing_adv_participation_pct", max_trailing_adv_participation_pct),
        ("max_entry_day_participation_pct", max_entry_day_participation_pct),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")

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
    profitable_threshold = round_trip_cost_bps / 10_000.0
    if abs(profitable_threshold - 0.002) > 1e-12:
        raise ValueError(
            "horizon-quality proof requires the current 20 bps round-trip cost floor"
        )

    prepared = executable._prepare_executable_data(
        root,
        market_db=resolved_market_db,
        benchmark_security_id=resolved_benchmark,
        market_quality_manifest=quality_manifest,
        market_read_cache_series=market_read_cache_series,
    )
    profiled_rows = apply_feature_profile(tuple(prepared.rows), spec.feature_profile)
    required_capital = starting_capital * allocation_pct

    scenarios: list[dict[str, Any]] = []
    for name, training_adv_prefilter in (
        ("horizon_quality_adv_prefilter", True),
        ("horizon_quality_adv_execution_only", False),
    ):
        replay = _prepare_horizon_aware_replay(
            name=name,
            rows=profiled_rows,
            targets=prepared.targets,
            security_ids=prepared.security_ids,
            invalid_target_keys=prepared.invalid_target_keys,
            spec=spec,
            min_train_rows=min_train_rows,
            profitable_threshold=profitable_threshold,
            required_capital=required_capital,
            max_trailing_adv_participation_pct=max_trailing_adv_participation_pct,
            training_adv_prefilter=training_adv_prefilter,
        )
        scorecard, execution_diag = _run_continuous_replay(
            replay,
            entry_liquidity=prepared.entry_liquidity,
            starting_capital=starting_capital,
            allocation_pct=allocation_pct,
            max_open_positions=max_open_positions,
            max_gross_exposure_pct=max_gross_exposure_pct,
            round_trip_cost_bps=round_trip_cost_bps,
            max_entry_day_participation_pct=max_entry_day_participation_pct,
            max_expected_downside=spec.max_expected_downside,
        )
        scenarios.append(
            {
                "scenario": name,
                "training_adv_prefilter": training_adv_prefilter,
                "quality_mode": "per_horizon_target",
                "maturity_mode": "per_horizon_target_exit_before_test_year",
                "calibration_candidate_adv_required": True,
                "test_candidate_adv_required": True,
                "test_candidate_all_horizons_quality_required": True,
                "continuous_portfolio": True,
                "scorecard": scorecard,
                "viability": minimum_viability(scorecard),
                "model_years": sorted(replay.strategies),
                "candidate_count": len(replay.candidates),
                "salvaged_valid_label_count": replay.salvaged_label_count,
                "quality_unscorable_candidate_count": replay.quality_unscorable_candidate_count,
                "adv_candidate_removed_count": replay.adv_candidate_removed_count,
                "execution_diagnostics": execution_diag,
                "walk_forward_splits": list(replay.split_diagnostics),
            }
        )

    strict_reference = _load_strict_reference(root)
    for scenario in scenarios:
        scenario["delta_vs_current_strict"] = _scorecard_delta(
            strict_reference.get("scorecard"),
            scenario["scorecard"],
        )

    best = max(
        scenarios,
        key=lambda item: float(item["scorecard"]["total_return"]),
    )
    payload = _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_dir": str(root),
            "runtime_dir": str(runtime_root),
            "question": (
                "Does current V5 recover historical viability when verified market "
                "quality and label maturity are handled per horizon instead of "
                "deleting an entire opportunity row?"
            ),
            "parameter_search": False,
            "strategy_thresholds_changed": False,
            "strategy_spec": spec.as_json(),
            "portfolio_policy": {
                "starting_capital": starting_capital,
                "allocation_pct": allocation_pct,
                "max_open_positions": max_open_positions,
                "max_gross_exposure_pct": max_gross_exposure_pct,
            },
            "execution_policy": {
                "round_trip_cost_bps": round_trip_cost_bps,
                "max_trailing_adv_participation_pct": max_trailing_adv_participation_pct,
                "max_entry_day_participation_pct": max_entry_day_participation_pct,
                "full_fill_required": True,
            },
            "data": {
                "source_row_count": len(profiled_rows),
                "invalid_target_count": len(prepared.invalid_target_keys),
                "invalid_target_keys_by_horizon": _invalid_counts_by_horizon(
                    prepared.invalid_target_keys
                ),
                "verified_quality_exclusion_count": prepared.quality_exclusion_count,
                "market_cache_stats": prepared.market_cache_stats,
            },
            "current_strict_reference": strict_reference,
            "scenarios": scenarios,
            "best_scenario": best["scenario"],
            "best_scenario_viability": best["viability"],
            "interpretation": _interpret(scenarios),
        }
    )
    output_path = root / "v5_horizon_quality_profit_proof.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return V5HorizonQualityProfitProofResult(
        output_path=output_path,
        best_scenario=str(best["scenario"]),
        best_return=float(best["scorecard"]["total_return"]),
    )


def _prepare_horizon_aware_replay(
    *,
    name: str,
    rows: tuple,
    targets: dict,
    security_ids: dict[str, str],
    invalid_target_keys: frozenset[tuple[str, int]],
    spec: Any,
    min_train_rows: int,
    profitable_threshold: float,
    required_capital: float,
    max_trailing_adv_participation_pct: float,
    training_adv_prefilter: bool,
) -> _ScenarioReplay:
    years = sorted({row.decision_time.year for row in rows})
    profit_trainer = ProfitLightGbmTrainer(spec.training_config)
    alpha_trainer = LightGbmTrainer(spec.training_config)
    strategies: dict[int, V5AdaptiveHorizonStrategy] = {}
    candidates: list = []
    split_diagnostics: list[dict[str, Any]] = []

    invalid_event_ids = {event_id for event_id, _ in invalid_target_keys}
    salvaged_label_count = sum(
        1
        for row in rows
        if row.event_id in invalid_event_ids
        for horizon in spec.horizons
        if (row.event_id, int(horizon)) not in invalid_target_keys
    )
    quality_unscorable_candidate_count = 0
    adv_candidate_removed_count = 0

    for test_year in years:
        validation_year = test_year - 1
        test_start = date(test_year, 1, 1)

        calibration_pool = tuple(
            row
            for row in rows
            if row.decision_time.year == validation_year
            and row.execution_date < test_start
        )
        calibration_rows, calibration_adv_removed = _adv_filter_rows(
            calibration_pool,
            required_capital=required_capital,
            max_participation_pct=max_trailing_adv_participation_pct,
        )

        test_pool = tuple(row for row in rows if row.decision_time.year == test_year)
        quality_test_rows = tuple(
            row
            for row in test_pool
            if _candidate_all_horizons_quality_valid(
                row.event_id,
                spec.horizons,
                invalid_target_keys,
            )
        )
        quality_unscorable_candidate_count += len(test_pool) - len(quality_test_rows)
        test_rows, test_adv_removed = _adv_filter_rows(
            quality_test_rows,
            required_capital=required_capital,
            max_participation_pct=max_trailing_adv_participation_pct,
        )
        adv_candidate_removed_count += calibration_adv_removed + test_adv_removed

        if not calibration_rows or not test_rows:
            continue

        models: dict[int, V5HorizonModels] = {}
        horizon_diagnostics: dict[str, Any] = {}
        viable_year = True

        for horizon in spec.horizons:
            train_rows = tuple(
                row
                for row in rows
                if row.decision_time.year < validation_year
                and _label_is_eligible(
                    row,
                    targets[row.event_id][horizon],
                    horizon=int(horizon),
                    test_start=test_start,
                    invalid_target_keys=invalid_target_keys,
                )
            )
            validation_h = tuple(
                row
                for row in rows
                if row.decision_time.year == validation_year
                and _label_is_eligible(
                    row,
                    targets[row.event_id][horizon],
                    horizon=int(horizon),
                    test_start=test_start,
                    invalid_target_keys=invalid_target_keys,
                )
            )

            if training_adv_prefilter:
                train_rows, _ = _adv_filter_rows(
                    train_rows,
                    required_capital=required_capital,
                    max_participation_pct=max_trailing_adv_participation_pct,
                )
                validation_h, _ = _adv_filter_rows(
                    validation_h,
                    required_capital=required_capital,
                    max_participation_pct=max_trailing_adv_participation_pct,
                )

            train_rows = training_window_rows(
                train_rows,
                test_year=test_year,
                window_years=spec.training_window_years,
            )
            if len(train_rows) < min_train_rows or not validation_h:
                viable_year = False
                break

            train_projected = tuple(
                row_for_horizon(row, targets[row.event_id][horizon])
                for row in train_rows
            )
            validation_projected = tuple(
                row_for_horizon(row, targets[row.event_id][horizon])
                for row in validation_h
            )
            models[horizon] = V5HorizonModels(
                profit=profit_trainer.train(
                    train_projected,
                    validation_projected,
                    profitable_return_threshold=profitable_threshold,
                ),
                alpha=alpha_trainer.train(
                    train_projected,
                    validation_projected,
                ),
            )
            horizon_diagnostics[str(horizon)] = {
                "train_count": len(train_rows),
                "validation_label_count": len(validation_h),
                "latest_label_exit_before_test_year": True,
                "invalid_labels_excluded_only_for_this_horizon": True,
            }

        if not viable_year:
            continue

        strategy_config = V5StrategyConfig(
            strategy_id=spec.variant_id,
            horizons=spec.horizons,
            validation_top_fraction=spec.validation_top_fraction,
            alpha_rank_weight=spec.alpha_rank_weight,
            calibration_window_days=spec.calibration_window_days,
            min_expected_return=profitable_threshold,
            max_expected_downside=spec.max_expected_downside,
        )
        validation_candidates = base_factory._feature_snapshots(
            calibration_rows,
            security_ids,
        )
        calibration = V5CalibrationState.from_validation(
            validation_candidates,
            models,
            strategy_config,
        )
        strategies[test_year] = V5AdaptiveHorizonStrategy(
            models,
            calibration,
            strategy_config,
        )
        candidates.extend(
            base_factory._historical_candidates(
                test_rows,
                targets,
                security_ids,
            )
        )
        split_diagnostics.append(
            {
                "test_year": test_year,
                "calibration_candidate_count": len(calibration_rows),
                "test_candidate_count": len(test_rows),
                "horizons": horizon_diagnostics,
                "training_adv_prefilter": training_adv_prefilter,
            }
        )

    materialized = tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.snapshot.execution_date,
                item.snapshot.decision_time,
                item.snapshot.candidate_id,
            ),
        )
    )
    if not strategies:
        raise ValueError(f"{name} produced no trainable V5 years")
    if len({item.snapshot.candidate_id for item in materialized}) != len(materialized):
        raise RuntimeError(f"{name} produced duplicate candidate identities")

    return _ScenarioReplay(
        name=name,
        strategies=strategies,
        candidates=materialized,
        split_diagnostics=tuple(split_diagnostics),
        salvaged_label_count=salvaged_label_count,
        quality_unscorable_candidate_count=quality_unscorable_candidate_count,
        adv_candidate_removed_count=adv_candidate_removed_count,
    )


def _run_continuous_replay(
    replay: _ScenarioReplay,
    *,
    entry_liquidity: dict,
    starting_capital: float,
    allocation_pct: float,
    max_open_positions: int,
    max_gross_exposure_pct: float,
    round_trip_cost_bps: float,
    max_entry_day_participation_pct: float,
    max_expected_downside: float,
) -> tuple[dict[str, Any], dict[str, int]]:
    router = _AnnualV5StrategyRouter(replay.strategies)
    backtester = ExecutionRealisticHistoricalBacktester(
        starting_capital=starting_capital,
        round_trip_cost_bps=round_trip_cost_bps,
    )
    backtest, diagnostics = backtester.run(
        strategy=router,
        candidates=replay.candidates,
        opportunity_risk=BasicOpportunityRiskPolicy(
            max_expected_downside=max_expected_downside,
            min_expected_return=0.0,
            min_probability_positive=0.0,
        ),
        portfolio_policy=FixedAllocationPortfolioPolicy(
            allocation_pct=allocation_pct,
            max_open_positions=max_open_positions,
            max_gross_exposure_pct=max_gross_exposure_pct,
            one_position_per_company=True,
        ),
        portfolio_risk=PassThroughPortfolioRiskPolicy(),
        entry_liquidity=entry_liquidity,
        max_entry_day_participation_pct=max_entry_day_participation_pct,
    )
    return _scorecard(backtest), {
        "rejected_entry_liquidity": int(diagnostics.rejected_entry_liquidity),
        "rejected_cash": int(backtest.rejected_cash),
    }


def _scorecard(backtest: Any) -> dict[str, Any]:
    trades = tuple(backtest.trades)
    pnl_by_entry_year: dict[int, float] = defaultdict(float)
    trade_counts: Counter[int] = Counter()
    horizon_counts: Counter[int] = Counter()
    for trade in trades:
        pnl_by_entry_year[trade.entry_date.year] += trade.pnl
        trade_counts[trade.entry_date.year] += 1
        horizon_counts[trade.horizon_sessions] += 1

    best_year = (
        max(pnl_by_entry_year, key=lambda year: (pnl_by_entry_year[year], -year))
        if pnl_by_entry_year
        else None
    )
    net_profit = backtest.ending_capital - backtest.starting_capital
    average_alpha = (
        sum(item.alpha for item in trades) / len(trades)
        if trades
        else None
    )
    average_net_return = (
        sum(item.net_return for item in trades) / len(trades)
        if trades
        else None
    )
    win_rate = (
        sum(item.net_return > 0 for item in trades) / len(trades)
        if trades
        else None
    )
    return {
        "starting_capital": backtest.starting_capital,
        "ending_capital": backtest.ending_capital,
        "net_profit": net_profit,
        "total_return": backtest.total_return,
        "profit_factor": backtest.profit_factor,
        "realized_max_drawdown": backtest.realized_max_drawdown,
        "total_trades": len(trades),
        "win_rate": win_rate,
        "average_net_trade_return": average_net_return,
        "average_trade_alpha": average_alpha,
        "best_entry_year": best_year,
        "best_entry_year_pnl": (
            pnl_by_entry_year[best_year] if best_year is not None else None
        ),
        "net_profit_excluding_best_entry_year": (
            net_profit - pnl_by_entry_year[best_year]
            if best_year is not None
            else None
        ),
        "trade_horizon_counts": {
            str(horizon): int(count)
            for horizon, count in sorted(horizon_counts.items())
        },
        "entry_year_diagnostics": [
            {
                "year": year,
                "trade_count": int(trade_counts[year]),
                "realized_pnl": pnl_by_entry_year[year],
            }
            for year in sorted(trade_counts)
        ],
    }


def _label_is_eligible(
    row: Any,
    target: Any,
    *,
    horizon: int,
    test_start: date,
    invalid_target_keys: frozenset[tuple[str, int]],
) -> bool:
    return (
        (row.event_id, horizon) not in invalid_target_keys
        and target.exit_date < test_start
    )


def _candidate_all_horizons_quality_valid(
    event_id: str,
    horizons: Iterable[int],
    invalid_target_keys: frozenset[tuple[str, int]],
) -> bool:
    return not any(
        (event_id, int(horizon)) in invalid_target_keys
        for horizon in horizons
    )


def _adv_filter_rows(
    rows: tuple,
    *,
    required_capital: float,
    max_participation_pct: float,
) -> tuple[tuple, int]:
    kept = tuple(
        row
        for row in rows
        if trailing_adv_supports(
            row.features,
            required_capital=required_capital,
            max_participation_pct=max_participation_pct,
        )
    )
    return kept, len(rows) - len(kept)


def _invalid_counts_by_horizon(
    invalid_target_keys: frozenset[tuple[str, int]],
) -> dict[str, int]:
    counts = Counter(int(horizon) for _, horizon in invalid_target_keys)
    return {str(horizon): int(count) for horizon, count in sorted(counts.items())}


def _load_strict_reference(root: Path) -> dict[str, Any]:
    path = root / "v5_strict_profit_proof.json"
    if not path.exists():
        return {"available": False, "path": str(path), "scorecard": None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid strict V5 proof: {path}") from exc
    scorecard = payload.get("scorecard")
    if not isinstance(scorecard, dict):
        raise ValueError("strict V5 proof is missing scorecard")
    return {
        "available": True,
        "path": str(path),
        "verdict": payload.get("verdict"),
        "scorecard": scorecard,
    }


def _scorecard_delta(
    reference: dict[str, Any] | None,
    current: dict[str, Any],
) -> dict[str, Any] | None:
    if reference is None:
        return None
    current_alpha = current.get("average_trade_alpha")
    reference_alpha = reference.get("average_trade_alpha")
    return {
        "total_return_delta": (
            float(current["total_return"]) - float(reference["total_return"])
        ),
        "profit_factor_delta": (
            float(current["profit_factor"]) - float(reference["profit_factor"])
        ),
        "average_trade_alpha_delta": (
            float(current_alpha) - float(reference_alpha)
            if current_alpha is not None and reference_alpha is not None
            else None
        ),
        "trade_count_delta": (
            int(current["total_trades"]) - int(reference["total_trades"])
        ),
    }


def _interpret(scenarios: list[dict[str, Any]]) -> str:
    passed = [
        item["scenario"]
        for item in scenarios
        if item["viability"]["passes"]
    ]
    if passed:
        return (
            "At least one fixed-spec V5 replay clears minimum historical viability "
            "after correcting horizon-local label eligibility. This supports further "
            "forward validation but is not live-profit proof."
        )
    return (
        "Neither corrected fixed-spec replay clears minimum historical viability. "
        "The remaining V5 edge is not sufficient under these live-like constraints."
    )


def _compact_console(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload["schema_version"],
        "question": payload["question"],
        "parameter_search": payload["parameter_search"],
        "strategy_thresholds_changed": payload["strategy_thresholds_changed"],
        "data": payload["data"],
        "current_strict_reference": payload["current_strict_reference"],
        "scenarios": [
            {
                "scenario": item["scenario"],
                "training_adv_prefilter": item["training_adv_prefilter"],
                "candidate_count": item["candidate_count"],
                "salvaged_valid_label_count": item["salvaged_valid_label_count"],
                "quality_unscorable_candidate_count": item[
                    "quality_unscorable_candidate_count"
                ],
                "adv_candidate_removed_count": item["adv_candidate_removed_count"],
                "scorecard": item["scorecard"],
                "viability": item["viability"],
                "delta_vs_current_strict": item["delta_vs_current_strict"],
            }
            for item in payload["scenarios"]
        ],
        "best_scenario": payload["best_scenario"],
        "best_scenario_viability": payload["best_scenario_viability"],
        "interpretation": payload["interpretation"],
        "output_path": payload["output_path"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the fixed V5 strategy with per-horizon market-quality and "
            "maturity eligibility, comparing ADV pretraining against execution-only ADV."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, default=Path("data/runtime"))
    parser.add_argument("--market-db", type=Path)
    parser.add_argument("--benchmark-security-id")
    parser.add_argument("--starting-capital", type=float, default=10_000.0)
    parser.add_argument("--allocation-pct", type=float, default=0.02)
    parser.add_argument("--max-open-positions", type=int, default=15)
    parser.add_argument("--max-gross-exposure-pct", type=float, default=0.30)
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    parser.add_argument("--max-trailing-adv-participation-pct", type=float, default=0.01)
    parser.add_argument("--max-entry-day-participation-pct", type=float, default=0.01)
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
    result = run_v5_horizon_quality_profit_proof(
        args.experiment_dir,
        runtime_dir=args.runtime_dir,
        market_db=args.market_db,
        benchmark_security_id=args.benchmark_security_id,
        starting_capital=args.starting_capital,
        allocation_pct=args.allocation_pct,
        max_open_positions=args.max_open_positions,
        max_gross_exposure_pct=args.max_gross_exposure_pct,
        round_trip_cost_bps=args.round_trip_cost_bps,
        max_trailing_adv_participation_pct=args.max_trailing_adv_participation_pct,
        max_entry_day_participation_pct=args.max_entry_day_participation_pct,
        market_quality_manifest=args.market_quality_manifest,
        min_train_rows=args.min_train_rows,
        market_read_cache_series=args.market_read_cache_series,
    )
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    payload["output_path"] = str(result.output_path)
    print(json.dumps(_compact_console(payload), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
