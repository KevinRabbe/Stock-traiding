from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import lightgbm_strategy_factory as base_factory
from . import lightgbm_strategy_factory_executable as executable
from . import lightgbm_strategy_factory_executable_maturity as maturity
from .lightgbm_validation_rank import _json_safe
from .v5_profit_attribution import _load_legacy_saved_scenario, _load_strict_scenario
from .v5_profit_proof import (
    DEFAULT_MARKET_QUALITY_MANIFEST,
    _resolve_market_inputs,
    current_v5_profit_proof_spec,
)


SCHEMA_VERSION = "v5-retrain-attribution-v1"


@dataclass(frozen=True, slots=True)
class V5RetrainAttributionResult:
    output_path: Path
    primary_break: str


def run_v5_retrain_attribution(
    experiment_dir: str | Path,
    *,
    runtime_dir: str | Path = "data/runtime",
    market_db: str | Path | None = None,
    benchmark_security_id: str | None = None,
    starting_capital: float = 10_000.0,
    allocation_pct: float = 0.02,
    max_open_positions: int = 15,
    round_trip_cost_bps: float = 20.0,
    max_trailing_adv_participation_pct: float = 0.01,
    max_entry_day_participation_pct: float = 0.01,
    market_quality_manifest: str | Path = DEFAULT_MARKET_QUALITY_MANIFEST,
    min_train_rows: int = 100,
    market_read_cache_series: int = 200,
) -> V5RetrainAttributionResult:
    """Split the legacy-to-strict V5 collapse into reproducibility corrections.

    The first controlled pair keeps the legacy 20-session maturity boundary,
    annual-reset portfolio, generic V5 strategy adapter, and non-executable
    historical backtester constant. It changes only saved historical model
    artifacts versus a fresh deterministic retrain of the exact V5 design on the
    current prepared historical dataset.

    The next step adds verified market-quality/PIT-liquidity filters and hidden
    entry-day full-fill checks. Full-horizon maturity and continuous portfolio
    semantics are then applied in the final two steps.
    """

    if starting_capital <= 0:
        raise ValueError("starting_capital must be > 0")
    if not 0.0 < allocation_pct <= 1.0:
        raise ValueError("allocation_pct must be in (0, 1]")
    if max_open_positions <= 0 or min_train_rows <= 0 or market_read_cache_series <= 0:
        raise ValueError("position/train/cache limits must be > 0")
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
        raise ValueError("V5 retrain attribution requires the live 20 bps cost floor")

    baseline_prepared = base_factory._prepare_factory_data(
        root,
        market_db=resolved_market_db,
        benchmark_security_id=resolved_benchmark,
        market_read_cache_series=market_read_cache_series,
    )
    base_factory._initialize_worker(
        {
            "starting_capital": starting_capital,
            "allocation_pct": allocation_pct,
            "max_open_positions": max_open_positions,
            "round_trip_cost_bps": round_trip_cost_bps,
            "min_train_rows": min_train_rows,
        },
        baseline_prepared,
    )
    try:
        fresh_unfiltered = base_factory._evaluate_variant(spec)
    finally:
        base_factory._CONTEXT = None

    executable_prepared = executable._prepare_executable_data(
        root,
        market_db=resolved_market_db,
        benchmark_security_id=resolved_benchmark,
        market_quality_manifest=quality_manifest,
        market_read_cache_series=market_read_cache_series,
    )
    executable._initialize_worker(
        {
            "starting_capital": starting_capital,
            "allocation_pct": allocation_pct,
            "max_open_positions": max_open_positions,
            "round_trip_cost_bps": round_trip_cost_bps,
            "min_train_rows": min_train_rows,
            "max_trailing_adv_participation_pct": max_trailing_adv_participation_pct,
            "max_entry_day_participation_pct": max_entry_day_participation_pct,
        },
        executable_prepared,
    )
    try:
        fresh_executable = executable._evaluate_variant(spec)
        fresh_maturity = maturity._evaluate_variant(spec)
    finally:
        executable._CONTEXT = None

    scenarios = [
        _load_legacy_saved_scenario(root),
        _scenario_from_base("fresh_retrain_unfiltered_20d", fresh_unfiltered),
        _scenario_from_executable(
            "fresh_retrain_executable_20d",
            fresh_executable,
            maturity_mode="stored_20d_exit_date",
        ),
        _scenario_from_executable(
            "fresh_retrain_executable_full_maturity",
            fresh_maturity,
            maturity_mode="latest_requested_horizon_exit",
        ),
        _load_strict_scenario(root),
    ]
    attribution = _attribute_retrain_break(scenarios)

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
                    "Does the V5 edge disappear when the saved models are freshly "
                    "retrained, or only after executable-data constraints are added?"
                ),
            },
            "controlled_retrain_comparison": {
                "from": "legacy_saved_models_reference",
                "to": "fresh_retrain_unfiltered_20d",
                "saved_models_vs_fresh_retrain": True,
                "same_20d_maturity_mode": True,
                "same_annual_reset_portfolio": True,
                "execution_realism_enabled_in_both": False,
                "same_current_strategy_adapter": True,
            },
            "controlled_execution_comparison": {
                "from": "fresh_retrain_unfiltered_20d",
                "to": "fresh_retrain_executable_20d",
                "same_fresh_strategy_spec": True,
                "same_20d_maturity_mode": True,
                "same_annual_reset_portfolio": True,
                "added": [
                    "verified_market_quality_filter",
                    "pit_trailing_liquidity_filter",
                    "hidden_entry_day_full_fill_check",
                ],
            },
            "data": {
                "baseline_prepared_row_count": len(baseline_prepared.rows),
                "executable_prepared_row_count": len(executable_prepared.rows),
                "invalid_target_count": len(executable_prepared.invalid_target_keys),
                "quality_exclusion_count": executable_prepared.quality_exclusion_count,
                "market_cache_stats": executable_prepared.market_cache_stats,
            },
            "scenarios": scenarios,
            "attribution": attribution,
        }
    )
    output_path = root / "v5_retrain_attribution.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return V5RetrainAttributionResult(
        output_path=output_path,
        primary_break=str(attribution["primary_break"]),
    )


def _scenario_from_base(name: str, result: Any) -> dict[str, Any]:
    scorecard = result.as_json()["scorecard"]
    return {
        "scenario": name,
        "source": "fresh_fixed_spec_retrain",
        "maturity_mode": "stored_20d_exit_date",
        "portfolio_mode": "annual_reset",
        "execution_realism": False,
        "return": scorecard["compounded_return"],
        "profit_factor": scorecard["profit_factor"],
        "drawdown": scorecard["worst_realized_drawdown"],
        "total_trades": scorecard["total_trades"],
        "average_trade_alpha": scorecard["average_trade_alpha"],
        "profitable_year_rate": scorecard["profitable_year_rate"],
        "return_excluding_best_year": scorecard["compounded_return_excluding_best_year"],
        "best_year": scorecard["best_year"],
        "rejected_entry_liquidity": None,
        "trade_horizon_counts": {
            str(key): int(value)
            for key, value in sorted(result.trade_horizon_counts.items())
        },
        "trade_candidate_ids": list(result.trade_candidate_ids),
    }


def _scenario_from_executable(
    name: str,
    evaluation: Any,
    *,
    maturity_mode: str,
) -> dict[str, Any]:
    result = evaluation.result
    scorecard = result.as_json()["scorecard"]
    return {
        "scenario": name,
        "source": "fresh_fixed_spec_retrain",
        "maturity_mode": maturity_mode,
        "portfolio_mode": "annual_reset",
        "execution_realism": True,
        "return": scorecard["compounded_return"],
        "profit_factor": scorecard["profit_factor"],
        "drawdown": scorecard["worst_realized_drawdown"],
        "total_trades": scorecard["total_trades"],
        "average_trade_alpha": scorecard["average_trade_alpha"],
        "profitable_year_rate": scorecard["profitable_year_rate"],
        "return_excluding_best_year": scorecard["compounded_return_excluding_best_year"],
        "best_year": scorecard["best_year"],
        "rejected_entry_liquidity": int(evaluation.rejected_entry_liquidity),
        "trade_horizon_counts": {
            str(key): int(value)
            for key, value in sorted(result.trade_horizon_counts.items())
        },
        "trade_candidate_ids": list(result.trade_candidate_ids),
    }


def _attribute_retrain_break(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    if len(scenarios) != 5:
        raise ValueError("V5 retrain attribution requires exactly five ordered scenarios")
    step_names = (
        "saved_models_to_fresh_retrain",
        "fresh_retrain_to_executable_data",
        "executable_20d_to_full_maturity",
        "annual_reset_to_continuous_portfolio",
    )
    steps: list[dict[str, Any]] = []
    for step_name, previous, current in zip(step_names, scenarios, scenarios[1:], strict=True):
        previous_alpha = previous.get("average_trade_alpha")
        current_alpha = current.get("average_trade_alpha")
        alpha_delta = (
            float(current_alpha) - float(previous_alpha)
            if previous_alpha is not None and current_alpha is not None
            else None
        )
        previous_ids = previous.get("trade_candidate_ids")
        current_ids = current.get("trade_candidate_ids")
        overlap = None
        if previous_ids is not None and current_ids is not None:
            left = set(previous_ids)
            right = set(current_ids)
            union = left | right
            overlap = len(left & right) / len(union) if union else 1.0
        steps.append(
            {
                "step": step_name,
                "from": previous["scenario"],
                "to": current["scenario"],
                "return_delta": float(current["return"]) - float(previous["return"]),
                "profit_factor_delta": (
                    float(current["profit_factor"]) - float(previous["profit_factor"])
                ),
                "average_trade_alpha_delta": alpha_delta,
                "trade_count_delta": int(current["total_trades"]) - int(previous["total_trades"]),
                "trade_jaccard_overlap": overlap,
            }
        )

    comparable = [item for item in steps if item["average_trade_alpha_delta"] is not None]
    primary = (
        min(comparable, key=lambda item: float(item["average_trade_alpha_delta"]))["step"]
        if comparable
        else "undetermined"
    )
    by_name = {item["step"]: item for item in steps}
    return {
        "primary_break": primary,
        "method": "largest_negative_step_in_average_trade_alpha",
        "saved_models_to_fresh_retrain": by_name["saved_models_to_fresh_retrain"],
        "fresh_retrain_to_executable_data": by_name["fresh_retrain_to_executable_data"],
        "steps": steps,
    }


def _compact_console(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload["schema_version"],
        "scenarios": [
            {
                key: item[key]
                for key in (
                    "scenario",
                    "source",
                    "maturity_mode",
                    "portfolio_mode",
                    "execution_realism",
                    "return",
                    "profit_factor",
                    "drawdown",
                    "total_trades",
                    "average_trade_alpha",
                    "profitable_year_rate",
                    "return_excluding_best_year",
                    "rejected_entry_liquidity",
                )
            }
            for item in payload["scenarios"]
        ],
        "attribution": payload["attribution"],
        "output_path": str(Path(payload["experiment_dir"]) / "v5_retrain_attribution.json"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split the V5 saved-model profitability collapse into fresh retraining, "
            "executable-data filtering, maturity, and continuous-portfolio steps."
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
    result = run_v5_retrain_attribution(
        args.experiment_dir,
        runtime_dir=args.runtime_dir,
        market_db=args.market_db,
        benchmark_security_id=args.benchmark_security_id,
        starting_capital=args.starting_capital,
        allocation_pct=args.allocation_pct,
        max_open_positions=args.max_open_positions,
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
