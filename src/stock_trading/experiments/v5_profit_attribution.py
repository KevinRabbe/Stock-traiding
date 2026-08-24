from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import lightgbm_strategy_factory_executable as executable
from . import lightgbm_strategy_factory_executable_maturity as maturity
from .lightgbm_validation_rank import _json_safe
from .v5_profit_proof import (
    DEFAULT_MARKET_QUALITY_MANIFEST,
    _resolve_market_inputs,
    current_v5_profit_proof_spec,
)


SCHEMA_VERSION = "v5-profit-attribution-v1"


@dataclass(frozen=True, slots=True)
class V5ProfitAttributionResult:
    output_path: Path
    primary_break: str


def run_v5_profit_attribution(
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
) -> V5ProfitAttributionResult:
    """Identify which research correction removes the apparent V5 edge.

    This is an attribution diagnostic, not a strategy search. It evaluates the exact
    current V5 spec twice through the same executable evaluator. The controlled pair
    differs only in walk-forward target maturity: the legacy stored 20-session fence
    versus the corrected latest-required-horizon fence. Existing legacy saved-model
    and strict continuous results are included as outer reference points.
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
        raise ValueError("V5 attribution requires the live 20 bps round-trip cost floor")

    prepared = executable._prepare_executable_data(
        root,
        market_db=resolved_market_db,
        benchmark_security_id=resolved_benchmark,
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
    executable._initialize_worker(common, prepared)
    try:
        legacy_20d = executable._evaluate_variant(spec)
        full_maturity = maturity._evaluate_variant(spec)
    finally:
        executable._CONTEXT = None

    scenarios = [
        _load_legacy_saved_scenario(root),
        _scenario_from_evaluation(
            "retrained_executable_20d_maturity",
            legacy_20d,
            maturity_mode="stored_20d_exit_date",
        ),
        _scenario_from_evaluation(
            "retrained_executable_full_horizon_maturity",
            full_maturity,
            maturity_mode="latest_requested_horizon_exit",
        ),
        _load_strict_scenario(root),
    ]
    attribution = _attribute_break(scenarios)

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
                "question": "Which research correction removes the apparent V5 historical edge?",
            },
            "controlled_comparison": {
                "from": "retrained_executable_20d_maturity",
                "to": "retrained_executable_full_horizon_maturity",
                "only_intended_difference": "walk_forward_target_maturity_fence",
                "same_strategy_spec": True,
                "same_training_code": True,
                "same_execution_filters": True,
                "same_annual_reset_portfolio": True,
            },
            "data": {
                "prepared_row_count": len(prepared.rows),
                "invalid_target_count": len(prepared.invalid_target_keys),
                "quality_exclusion_count": prepared.quality_exclusion_count,
                "market_cache_stats": prepared.market_cache_stats,
            },
            "scenarios": scenarios,
            "attribution": attribution,
        }
    )
    output_path = root / "v5_profit_attribution.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return V5ProfitAttributionResult(
        output_path=output_path,
        primary_break=str(attribution["primary_break"]),
    )


def _scenario_from_evaluation(
    scenario: str,
    evaluation: Any,
    *,
    maturity_mode: str,
) -> dict[str, Any]:
    result = evaluation.result
    scorecard = result.as_json()["scorecard"]
    return {
        "scenario": scenario,
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


def _load_legacy_saved_scenario(root: Path) -> dict[str, Any]:
    path = root / "strategy_engine_v5_replay.json"
    payload = _load_json(path, "legacy V5 replay")
    if payload.get("schema_version") != "strategy-engine-v5-exact-replay":
        raise ValueError("unexpected legacy V5 replay schema")
    observed = payload.get("observed")
    if not isinstance(observed, dict):
        raise ValueError("legacy V5 replay is missing observed scorecard")
    return {
        "scenario": "legacy_saved_models_reference",
        "source": "existing_saved_model_replay",
        "maturity_mode": "stored_20d_exit_date",
        "portfolio_mode": "annual_reset",
        "execution_realism": False,
        "return": observed["compounded_return"],
        "profit_factor": observed["aggregate_profit_factor"],
        "drawdown": observed["worst_realized_drawdown"],
        "total_trades": observed["total_trades"],
        "average_trade_alpha": observed["average_trade_alpha"],
        "profitable_year_rate": observed["profitable_year_rate"],
        "return_excluding_best_year": None,
        "best_year": None,
        "rejected_entry_liquidity": None,
        "trade_horizon_counts": None,
        "trade_candidate_ids": None,
    }


def _load_strict_scenario(root: Path) -> dict[str, Any]:
    path = root / "v5_strict_profit_proof.json"
    payload = _load_json(path, "strict V5 profit proof")
    if payload.get("schema_version") != "v5-strict-profit-proof-v1":
        raise ValueError("unexpected strict V5 profit proof schema")
    scorecard = payload.get("scorecard")
    if not isinstance(scorecard, dict):
        raise ValueError("strict V5 profit proof is missing scorecard")
    return {
        "scenario": "strict_continuous_30pct_reference",
        "source": "existing_strict_profit_proof",
        "maturity_mode": "latest_requested_horizon_exit",
        "portfolio_mode": "continuous_30pct_gross_cap",
        "execution_realism": True,
        "return": scorecard["total_return"],
        "profit_factor": scorecard["profit_factor"],
        "drawdown": scorecard["realized_max_drawdown"],
        "total_trades": scorecard["total_trades"],
        "average_trade_alpha": scorecard["average_trade_alpha"],
        "profitable_year_rate": None,
        "return_excluding_best_year": None,
        "best_year": scorecard["best_entry_year"],
        "rejected_entry_liquidity": payload["execution_diagnostics"]["rejected_entry_liquidity"],
        "trade_horizon_counts": scorecard["trade_horizon_counts"],
        "trade_candidate_ids": None,
    }


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid {label}: {path}")
    return payload


def _attribute_break(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    for previous, current in zip(scenarios, scenarios[1:]):
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
                "from": previous["scenario"],
                "to": current["scenario"],
                "return_delta": float(current["return"]) - float(previous["return"]),
                "profit_factor_delta": float(current["profit_factor"]) - float(previous["profit_factor"]),
                "average_trade_alpha_delta": alpha_delta,
                "trade_count_delta": int(current["total_trades"]) - int(previous["total_trades"]),
                "trade_jaccard_overlap": overlap,
                "controlled_maturity_step": (
                    previous["scenario"] == "retrained_executable_20d_maturity"
                    and current["scenario"] == "retrained_executable_full_horizon_maturity"
                ),
            }
        )

    comparable = [item for item in steps if item["average_trade_alpha_delta"] is not None]
    primary = (
        min(comparable, key=lambda item: item["average_trade_alpha_delta"])["to"]
        if comparable
        else "undetermined"
    )
    controlled = next(
        (item for item in steps if item["controlled_maturity_step"]),
        None,
    )
    return {
        "primary_break": primary,
        "method": "largest_negative_step_in_average_trade_alpha",
        "controlled_maturity_step": controlled,
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
        "output_path": str(Path(payload["experiment_dir"]) / "v5_profit_attribution.json"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute the current V5 profitability failure to retraining/execution, "
            "full-horizon maturity, or strict continuous portfolio semantics."
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
    result = run_v5_profit_attribution(
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
