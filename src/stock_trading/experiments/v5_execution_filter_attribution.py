from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from . import lightgbm_strategy_factory as base_factory
from . import lightgbm_strategy_factory_executable as executable
from .lightgbm_validation_rank import _json_safe
from .v5_profit_proof import (
    DEFAULT_MARKET_QUALITY_MANIFEST,
    _resolve_market_inputs,
    current_v5_profit_proof_spec,
)


SCHEMA_VERSION = "v5-execution-filter-attribution-v1"
ADV_FEATURE = "market.avg_dollar_volume_20d"


@dataclass(frozen=True, slots=True)
class V5ExecutionFilterAttributionResult:
    output_path: Path
    primary_break: str


@dataclass(frozen=True, slots=True)
class _AdvClassification:
    event_id: str
    execution_year: int
    reason: str
    value: float | None


def run_v5_execution_filter_attribution(
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
) -> V5ExecutionFilterAttributionResult:
    """Attribute the V5 collapse across individual execution-realism filters.

    This diagnostic keeps the exact current V5 design fixed. It evaluates four
    ordered scenarios under the legacy 20-session maturity boundary and annual-reset
    portfolio so each step has a narrow interpretation:

    1. unfiltered historical rows;
    2. verified market-quality exclusions only;
    3. market quality plus PIT trailing-ADV eligibility;
    4. the same prefiltered rows plus hidden execution-day full-fill rejection.

    The first three scenarios use the same generic historical backtester. The final
    scenario uses the execution-realistic backtester, so the only intended extra
    constraint is the already-decided order's entry-day liquidity capacity.
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
    prepared = executable._prepare_executable_data(
        root,
        market_db=resolved_market_db,
        benchmark_security_id=resolved_benchmark,
        market_quality_manifest=quality_manifest,
        market_read_cache_series=market_read_cache_series,
    )
    source_rows = tuple(prepared.rows)
    quality_rows = _quality_valid_rows(
        source_rows,
        horizons=tuple(int(item) for item in spec.horizons),
        invalid_target_keys=prepared.invalid_target_keys,
    )

    required_capital = starting_capital * allocation_pct
    adv_classifications = tuple(
        _classify_adv_row(
            row,
            required_capital=required_capital,
            max_participation_pct=max_trailing_adv_participation_pct,
        )
        for row in quality_rows
    )
    passing_adv_ids = frozenset(
        item.event_id for item in adv_classifications if item.reason == "passes"
    )
    adv_rows = tuple(row for row in quality_rows if row.event_id in passing_adv_ids)

    common_base = {
        "starting_capital": starting_capital,
        "allocation_pct": allocation_pct,
        "max_open_positions": max_open_positions,
        "round_trip_cost_bps": round_trip_cost_bps,
        "min_train_rows": min_train_rows,
    }
    unfiltered = _evaluate_base_subset(spec, prepared, source_rows, common_base)
    quality_only = _evaluate_base_subset(spec, prepared, quality_rows, common_base)
    quality_adv = _evaluate_base_subset(spec, prepared, adv_rows, common_base)

    executable._initialize_worker(
        {
            **common_base,
            "max_trailing_adv_participation_pct": max_trailing_adv_participation_pct,
            "max_entry_day_participation_pct": max_entry_day_participation_pct,
        },
        prepared,
    )
    try:
        full_execution = executable._evaluate_variant(spec)
    finally:
        executable._CONTEXT = None

    scenarios = [
        _scenario_from_base("unfiltered_20d", unfiltered),
        _scenario_from_base("quality_only_20d", quality_only),
        _scenario_from_base("quality_plus_trailing_adv_20d", quality_adv),
        _scenario_from_executable("quality_adv_plus_entry_fill_20d", full_execution),
    ]
    attribution = _attribute_filter_break(scenarios)

    source_ids = frozenset(row.event_id for row in source_rows)
    quality_ids = frozenset(row.event_id for row in quality_rows)
    adv_ids = frozenset(row.event_id for row in adv_rows)
    quality_removed_ids = source_ids - quality_ids
    adv_removed_ids = quality_ids - adv_ids
    baseline_trade_ids = frozenset(unfiltered.trade_candidate_ids)

    adv_reason_counts = Counter(item.reason for item in adv_classifications)
    adv_reason_by_year: dict[str, Counter[str]] = defaultdict(Counter)
    for item in adv_classifications:
        adv_reason_by_year[str(item.execution_year)][item.reason] += 1

    quality_removed_by_year = Counter(
        str(row.execution_date.year)
        for row in source_rows
        if row.event_id in quality_removed_ids
    )
    adv_values_by_reason: dict[str, list[float]] = defaultdict(list)
    for item in adv_classifications:
        if item.value is not None and math.isfinite(item.value):
            adv_values_by_reason[item.reason].append(item.value)

    direct_removal = {
        "baseline_trade_count": len(baseline_trade_ids),
        "quality_removed_baseline_trade_count": len(baseline_trade_ids & quality_removed_ids),
        "adv_removed_baseline_trade_count": len(baseline_trade_ids & adv_removed_ids),
        "baseline_trades_surviving_quality_count": len(baseline_trade_ids & quality_ids),
        "baseline_trades_surviving_quality_and_adv_count": len(baseline_trade_ids & adv_ids),
        "baseline_trade_survival_fraction_after_quality_and_adv": (
            len(baseline_trade_ids & adv_ids) / len(baseline_trade_ids)
            if baseline_trade_ids
            else None
        ),
        "interpretation": (
            "Direct removal counts distinguish trades made impossible by the filters "
            "from trade-set changes caused indirectly by retraining/calibration on the "
            "prefiltered historical universe."
        ),
    }

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
                    "Which execution/data-realism filter causes the reproducible V5 "
                    "historical edge to disappear?"
                ),
            },
            "controlled_steps": [
                {
                    "from": "unfiltered_20d",
                    "to": "quality_only_20d",
                    "only_intended_change": "verified_market_quality_exclusions",
                },
                {
                    "from": "quality_only_20d",
                    "to": "quality_plus_trailing_adv_20d",
                    "only_intended_change": "pit_trailing_adv_eligibility",
                },
                {
                    "from": "quality_plus_trailing_adv_20d",
                    "to": "quality_adv_plus_entry_fill_20d",
                    "only_intended_change": "hidden_execution_day_full_fill_requirement",
                },
            ],
            "filters": {
                "source_row_count": len(source_rows),
                "quality_removed_row_count": len(quality_removed_ids),
                "quality_remaining_row_count": len(quality_rows),
                "quality_removed_by_execution_year": dict(sorted(quality_removed_by_year.items())),
                "trailing_adv_feature": ADV_FEATURE,
                "reference_order_capital": required_capital,
                "max_trailing_adv_participation_pct": max_trailing_adv_participation_pct,
                "minimum_required_trailing_adv": (
                    required_capital / max_trailing_adv_participation_pct
                ),
                "adv_removed_row_count": len(adv_removed_ids),
                "adv_remaining_row_count": len(adv_rows),
                "adv_reason_counts": dict(sorted(adv_reason_counts.items())),
                "adv_reason_counts_by_execution_year": {
                    year: dict(sorted(counts.items()))
                    for year, counts in sorted(adv_reason_by_year.items())
                },
                "adv_value_summary_by_reason": {
                    reason: _value_summary(values)
                    for reason, values in sorted(adv_values_by_reason.items())
                },
                "entry_day_rejected_trade_count": int(full_execution.rejected_entry_liquidity),
            },
            "direct_baseline_trade_removal": direct_removal,
            "scenarios": scenarios,
            "attribution": attribution,
        }
    )
    output_path = root / "v5_execution_filter_attribution.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return V5ExecutionFilterAttributionResult(
        output_path=output_path,
        primary_break=str(attribution["primary_break"]),
    )


def _quality_valid_rows(
    rows: tuple,
    *,
    horizons: tuple[int, ...],
    invalid_target_keys: frozenset[tuple[str, int]],
) -> tuple:
    return tuple(
        row
        for row in rows
        if not any((row.event_id, horizon) in invalid_target_keys for horizon in horizons)
    )


def _classify_adv_row(
    row: Any,
    *,
    required_capital: float,
    max_participation_pct: float,
) -> _AdvClassification:
    if required_capital <= 0:
        raise ValueError("required_capital must be > 0")
    if not 0.0 < max_participation_pct <= 1.0:
        raise ValueError("max_participation_pct must be in (0, 1]")
    raw = row.features.get(ADV_FEATURE)
    if raw is None:
        return _AdvClassification(row.event_id, row.execution_date.year, "missing", None)
    if isinstance(raw, bool):
        return _AdvClassification(row.event_id, row.execution_date.year, "invalid", None)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _AdvClassification(row.event_id, row.execution_date.year, "invalid", None)
    if not math.isfinite(value):
        return _AdvClassification(row.event_id, row.execution_date.year, "invalid", value)
    if value <= 0:
        return _AdvClassification(row.event_id, row.execution_date.year, "nonpositive", value)
    capacity = value * max_participation_pct
    reason = "passes" if required_capital <= capacity + 1e-12 else "below_required"
    return _AdvClassification(row.event_id, row.execution_date.year, reason, value)


def _evaluate_base_subset(spec: Any, prepared: Any, rows: tuple, common: dict[str, Any]) -> Any:
    subset = base_factory._PreparedFactoryData(
        rows=rows,
        targets=prepared.targets,
        security_ids=prepared.security_ids,
        market_cache_stats=prepared.market_cache_stats,
    )
    base_factory._initialize_worker(common, subset)
    try:
        return base_factory._evaluate_variant(spec)
    finally:
        base_factory._CONTEXT = None


def _scenario_from_base(name: str, result: Any) -> dict[str, Any]:
    scorecard = result.as_json()["scorecard"]
    return {
        "scenario": name,
        "source": "fresh_fixed_spec_retrain",
        "maturity_mode": "stored_20d_exit_date",
        "portfolio_mode": "annual_reset",
        "entry_day_fill_realism": False,
        "return": scorecard["compounded_return"],
        "profit_factor": scorecard["profit_factor"],
        "drawdown": scorecard["worst_realized_drawdown"],
        "total_trades": scorecard["total_trades"],
        "average_trade_alpha": scorecard["average_trade_alpha"],
        "profitable_year_rate": scorecard["profitable_year_rate"],
        "return_excluding_best_year": scorecard["compounded_return_excluding_best_year"],
        "best_year": scorecard["best_year"],
        "rejected_entry_liquidity": 0,
        "trade_candidate_ids": list(result.trade_candidate_ids),
    }


def _scenario_from_executable(name: str, evaluation: Any) -> dict[str, Any]:
    result = evaluation.result
    scorecard = result.as_json()["scorecard"]
    return {
        "scenario": name,
        "source": "fresh_fixed_spec_retrain",
        "maturity_mode": "stored_20d_exit_date",
        "portfolio_mode": "annual_reset",
        "entry_day_fill_realism": True,
        "return": scorecard["compounded_return"],
        "profit_factor": scorecard["profit_factor"],
        "drawdown": scorecard["worst_realized_drawdown"],
        "total_trades": scorecard["total_trades"],
        "average_trade_alpha": scorecard["average_trade_alpha"],
        "profitable_year_rate": scorecard["profitable_year_rate"],
        "return_excluding_best_year": scorecard["compounded_return_excluding_best_year"],
        "best_year": scorecard["best_year"],
        "rejected_entry_liquidity": int(evaluation.rejected_entry_liquidity),
        "trade_candidate_ids": list(result.trade_candidate_ids),
    }


def _attribute_filter_break(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    if len(scenarios) != 4:
        raise ValueError("execution filter attribution requires exactly four scenarios")
    step_names = (
        "market_quality_exclusions",
        "pit_trailing_adv_filter",
        "hidden_entry_day_full_fill",
    )
    steps: list[dict[str, Any]] = []
    for step_name, previous, current in zip(
        step_names,
        scenarios[:-1],
        scenarios[1:],
        strict=True,
    ):
        left_ids = set(previous.get("trade_candidate_ids") or ())
        right_ids = set(current.get("trade_candidate_ids") or ())
        union = left_ids | right_ids
        steps.append(
            {
                "step": step_name,
                "from": previous["scenario"],
                "to": current["scenario"],
                "return_delta": float(current["return"]) - float(previous["return"]),
                "profit_factor_delta": (
                    float(current["profit_factor"]) - float(previous["profit_factor"])
                ),
                "average_trade_alpha_delta": (
                    float(current["average_trade_alpha"])
                    - float(previous["average_trade_alpha"])
                ),
                "trade_count_delta": int(current["total_trades"]) - int(previous["total_trades"]),
                "trade_jaccard_overlap": len(left_ids & right_ids) / len(union) if union else 1.0,
            }
        )
    primary = min(steps, key=lambda item: float(item["average_trade_alpha_delta"]))["step"]
    return {
        "primary_break": primary,
        "method": "largest_negative_step_in_average_trade_alpha",
        "steps": steps,
    }


def _value_summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": median(ordered),
        "max": ordered[-1],
    }


def _compact_console(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload["schema_version"],
        "filters": payload["filters"],
        "direct_baseline_trade_removal": payload["direct_baseline_trade_removal"],
        "scenarios": [
            {
                key: item[key]
                for key in (
                    "scenario",
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
        "output_path": payload["output_path"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute the reproducible V5 edge collapse across verified market-quality, "
            "PIT trailing-ADV, and hidden execution-day fill filters."
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
        default=Path(DEFAULT_MARKET_QUALITY_MANIFEST),
    )
    parser.add_argument("--min-train-rows", type=int, default=100)
    parser.add_argument("--market-read-cache-series", type=int, default=200)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_v5_execution_filter_attribution(
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
    payload["output_path"] = str(result.output_path)
    print(json.dumps(_compact_console(payload), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
