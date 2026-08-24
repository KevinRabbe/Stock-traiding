from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from stock_trading.ml.online_calibration import RollingScoreHistory
from stock_trading.research.strategy_factory import apply_feature_profile
from stock_trading.strategies.v5_adaptive_horizon import (
    V5AdaptiveHorizonStrategy,
    V5CalibrationState,
)

from . import lightgbm_strategy_factory_executable as executable
from .lightgbm_validation_rank import _json_safe
from .v5_edge_attribution import (
    BASELINE_SCENARIO,
    TRIGGER_FEATURES,
    _HorizonExcludingStrategy,
    _execute_replay,
    _feature_is_active,
    _scorecard_delta,
)
from .v5_horizon_quality_profit_proof import (
    _prepare_horizon_aware_replay,
    _scorecard,
)
from .v5_profit_proof import (
    DEFAULT_MARKET_QUALITY_MANIFEST,
    DEFAULT_MAX_GROSS_EXPOSURE_PCT,
    _AnnualV5StrategyRouter,
    _resolve_market_inputs,
    current_v5_profit_proof_spec,
)


SCHEMA_VERSION = "v5-edge-counterfactual-proof-v1"


@dataclass(frozen=True, slots=True)
class V5EdgeCounterfactualProofResult:
    output_path: Path
    baseline_return: float
    baseline_average_trade_alpha: float | None


def _clone_score_history(source: RollingScoreHistory) -> RollingScoreHistory:
    clone = RollingScoreHistory(window_days=source.window_days)
    clone.seed(source.snapshot())
    return clone


def _fresh_strategy(source: V5AdaptiveHorizonStrategy) -> V5AdaptiveHorizonStrategy:
    calibration = source.calibration
    cloned_calibration = V5CalibrationState(
        profit_histories={
            horizon: _clone_score_history(history)
            for horizon, history in calibration.profit_histories.items()
        },
        alpha_histories={
            horizon: _clone_score_history(history)
            for horizon, history in calibration.alpha_histories.items()
        },
        final_history=_clone_score_history(calibration.final_history),
    )
    return V5AdaptiveHorizonStrategy(
        source.models,
        cloned_calibration,
        source.config,
    )


def _fresh_strategies(
    strategies: Mapping[int, V5AdaptiveHorizonStrategy],
) -> dict[int, V5AdaptiveHorizonStrategy]:
    return {
        year: _fresh_strategy(strategy)
        for year, strategy in strategies.items()
    }


def _scorecards_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = (
        "starting_capital",
        "ending_capital",
        "net_profit",
        "total_return",
        "profit_factor",
        "realized_max_drawdown",
        "total_trades",
        "win_rate",
        "average_net_trade_return",
        "average_trade_alpha",
        "net_profit_excluding_best_entry_year",
    )
    return all(left.get(key) == right.get(key) for key in keys)


def run_v5_edge_counterfactual_proof(
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
) -> V5EdgeCounterfactualProofResult:
    """Re-run V5 edge counterfactuals from identical calibration snapshots.

    The first edge-attribution experiment correctly described realized trades, but
    its sequential counterfactual replays reused stateful RollingScoreHistory
    instances. This proof keeps the trained annual models fixed and clones the
    original calibration snapshot before every replay. A no-op candidate filter is
    required to reproduce the baseline exactly.
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
        raise ValueError("V5 edge counterfactual proof requires the current 20 bps cost floor")

    prepared = executable._prepare_executable_data(
        root,
        market_db=resolved_market_db,
        benchmark_security_id=resolved_benchmark,
        market_quality_manifest=quality_manifest,
        market_read_cache_series=market_read_cache_series,
    )
    profiled_rows = apply_feature_profile(tuple(prepared.rows), spec.feature_profile)
    required_capital = starting_capital * allocation_pct
    replay = _prepare_horizon_aware_replay(
        name=BASELINE_SCENARIO,
        rows=profiled_rows,
        targets=prepared.targets,
        security_ids=prepared.security_ids,
        invalid_target_keys=prepared.invalid_target_keys,
        spec=spec,
        min_train_rows=min_train_rows,
        profitable_threshold=profitable_threshold,
        required_capital=required_capital,
        max_trailing_adv_participation_pct=max_trailing_adv_participation_pct,
        training_adv_prefilter=False,
    )

    def execute(strategy: Any, candidates: tuple) -> tuple[dict[str, Any], dict[str, int]]:
        backtest, diagnostics = _execute_replay(
            strategy=strategy,
            candidates=candidates,
            entry_liquidity=prepared.entry_liquidity,
            starting_capital=starting_capital,
            allocation_pct=allocation_pct,
            max_open_positions=max_open_positions,
            max_gross_exposure_pct=max_gross_exposure_pct,
            round_trip_cost_bps=round_trip_cost_bps,
            max_entry_day_participation_pct=max_entry_day_participation_pct,
            max_expected_downside=spec.max_expected_downside,
        )
        return _scorecard(backtest), diagnostics

    baseline_scorecard, baseline_diagnostics = execute(
        _AnnualV5StrategyRouter(_fresh_strategies(replay.strategies)),
        replay.candidates,
    )

    repeat_scorecard, repeat_diagnostics = execute(
        _AnnualV5StrategyRouter(_fresh_strategies(replay.strategies)),
        replay.candidates,
    )
    repeated_baseline_matches = _scorecards_match(baseline_scorecard, repeat_scorecard)
    if not repeated_baseline_matches:
        raise RuntimeError("fresh calibration clone does not reproduce baseline deterministically")

    horizon_results: dict[str, Any] = {}
    for horizon in spec.horizons:
        strategy = _HorizonExcludingStrategy(
            _AnnualV5StrategyRouter(_fresh_strategies(replay.strategies)),
            int(horizon),
        )
        scorecard, diagnostics = execute(strategy, replay.candidates)
        horizon_results[str(horizon)] = {
            "scorecard": scorecard,
            "delta_vs_baseline": _scorecard_delta(baseline_scorecard, scorecard),
            "execution_diagnostics": diagnostics,
        }

    trigger_results: dict[str, Any] = {}
    for family, feature_name in TRIGGER_FEATURES.items():
        candidates = tuple(
            candidate
            for candidate in replay.candidates
            if not _feature_is_active(candidate.snapshot.features, feature_name)
        )
        scorecard, diagnostics = execute(
            _AnnualV5StrategyRouter(_fresh_strategies(replay.strategies)),
            candidates,
        )
        removed = len(replay.candidates) - len(candidates)
        no_op_matches_baseline = (
            _scorecards_match(baseline_scorecard, scorecard)
            if removed == 0
            else None
        )
        if removed == 0 and not no_op_matches_baseline:
            raise RuntimeError(
                f"no-op {family} counterfactual failed to reproduce baseline"
            )
        trigger_results[family] = {
            "candidate_count": len(candidates),
            "candidate_count_removed": removed,
            "no_op_matches_baseline": no_op_matches_baseline,
            "scorecard": scorecard,
            "delta_vs_baseline": _scorecard_delta(baseline_scorecard, scorecard),
            "execution_diagnostics": diagnostics,
        }

    payload = _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_dir": str(root),
            "runtime_dir": str(runtime_root),
            "question": (
                "What are the V5 leave-one-component-out effects when every replay "
                "starts from the exact same pre-test rolling-calibration snapshot?"
            ),
            "parameter_search": False,
            "strategy_thresholds_changed": False,
            "paper_policy_changed": False,
            "baseline_scenario": BASELINE_SCENARIO,
            "state_isolation": {
                "rolling_calibration_is_stateful": True,
                "fresh_calibration_clone_per_replay": True,
                "trained_models_reused": True,
                "models_retrained_per_counterfactual": False,
                "repeated_baseline_matches": repeated_baseline_matches,
            },
            "baseline": {
                "scorecard": baseline_scorecard,
                "execution_diagnostics": baseline_diagnostics,
            },
            "repeated_baseline": {
                "scorecard": repeat_scorecard,
                "execution_diagnostics": repeat_diagnostics,
            },
            "leave_one_horizon_out": horizon_results,
            "leave_one_trigger_family_out": trigger_results,
            "interpretation_guardrail": (
                "Horizon exclusion suppresses opportunities after V5 has selected their "
                "holding horizon; it does not retrain models or choose a second-best "
                "horizon. Trigger exclusions remove candidates before strategy evaluation."
            ),
        }
    )
    output_path = root / "v5_edge_counterfactual_proof.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return V5EdgeCounterfactualProofResult(
        output_path=output_path,
        baseline_return=float(baseline_scorecard["total_return"]),
        baseline_average_trade_alpha=baseline_scorecard.get("average_trade_alpha"),
    )


def _compact_console(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload["schema_version"],
        "question": payload["question"],
        "parameter_search": payload["parameter_search"],
        "strategy_thresholds_changed": payload["strategy_thresholds_changed"],
        "state_isolation": payload["state_isolation"],
        "baseline": payload["baseline"],
        "leave_one_horizon_out": payload["leave_one_horizon_out"],
        "leave_one_trigger_family_out": payload["leave_one_trigger_family_out"],
        "interpretation_guardrail": payload["interpretation_guardrail"],
        "output_path": str(
            Path(payload["experiment_dir"]) / "v5_edge_counterfactual_proof.json"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run V5 edge leave-one-component-out diagnostics with isolated "
            "rolling-calibration state."
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
    result = run_v5_edge_counterfactual_proof(
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
    print(json.dumps(_compact_console(payload), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
