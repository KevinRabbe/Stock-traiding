from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from stock_trading.engine import (
    BasicOpportunityRiskPolicy,
    FixedAllocationPortfolioPolicy,
    PassThroughPortfolioRiskPolicy,
)
from stock_trading.research.execution_realism import ExecutionRealisticHistoricalBacktester
from stock_trading.research.strategy_factory import apply_feature_profile

from . import lightgbm_strategy_factory_executable as executable
from .lightgbm_validation_rank import _json_safe
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


SCHEMA_VERSION = "v5-edge-attribution-v1"
BASELINE_SCENARIO = "horizon_quality_adv_execution_only"
TRIGGER_FEATURES = {
    "insider": "trigger.is_insider",
    "contract": "trigger.is_contract",
    "lobbying": "trigger.is_lobbying",
}


@dataclass(frozen=True, slots=True)
class V5EdgeAttributionResult:
    output_path: Path
    average_trade_alpha: float | None
    remaining_alpha_shape: str


class _HorizonExcludingStrategy:
    def __init__(self, delegate: Any, excluded_horizon: int) -> None:
        self._delegate = delegate
        self._excluded_horizon = excluded_horizon
        self.strategy_id = delegate.strategy_id

    def evaluate(self, candidates: tuple, portfolio: Any) -> tuple:
        return tuple(
            opportunity
            for opportunity in self._delegate.evaluate(candidates, portfolio)
            if opportunity.horizon_sessions != self._excluded_horizon
        )


def run_v5_edge_attribution(
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
) -> V5EdgeAttributionResult:
    """Diagnose where the residual V5 alpha deficit lives without tuning V5.

    The model/training/execution specification is the exact corrected fixed-spec
    ``horizon_quality_adv_execution_only`` replay. Attribution is descriptive except
    for explicit leave-one-component-out execution counterfactuals, which reuse the
    already-trained annual models and do not retrain or alter thresholds.
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
        raise ValueError("V5 edge attribution requires the current 20 bps cost floor")

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

    baseline_backtest, baseline_execution = _execute_replay(
        strategy=_AnnualV5StrategyRouter(replay.strategies),
        candidates=replay.candidates,
        entry_liquidity=prepared.entry_liquidity,
        starting_capital=starting_capital,
        allocation_pct=allocation_pct,
        max_open_positions=max_open_positions,
        max_gross_exposure_pct=max_gross_exposure_pct,
        round_trip_cost_bps=round_trip_cost_bps,
        max_entry_day_participation_pct=max_entry_day_participation_pct,
        max_expected_downside=spec.max_expected_downside,
    )
    baseline_scorecard = _scorecard(baseline_backtest)
    trades = tuple(baseline_backtest.trades)
    candidate_by_id = {
        candidate.snapshot.candidate_id: candidate
        for candidate in replay.candidates
    }

    by_horizon = _grouped_trade_summaries(
        trades,
        lambda trade: str(trade.horizon_sessions),
    )
    by_year = _grouped_trade_summaries(
        trades,
        lambda trade: str(trade.entry_date.year),
    )
    horizon_year = _grouped_trade_summaries(
        trades,
        lambda trade: f"{trade.horizon_sessions}d:{trade.entry_date.year}",
    )
    score_deciles = _score_deciles(trades)
    trigger_attribution = _trigger_attribution(trades, candidate_by_id)

    counterfactuals = {
        "leave_one_horizon_out": {},
        "leave_one_trigger_family_out": {},
        "semantics": (
            "Execution-only leave-one-component-out replays reuse the exact trained "
            "annual V5 models. They are diagnostics, not parameter searches. Horizon "
            "counterfactuals suppress already-scored opportunities with that selected "
            "holding horizon. Trigger counterfactuals remove matching PIT candidates "
            "before strategy evaluation."
        ),
    }
    for horizon in spec.horizons:
        strategy = _HorizonExcludingStrategy(
            _AnnualV5StrategyRouter(replay.strategies),
            int(horizon),
        )
        backtest, diagnostics = _execute_replay(
            strategy=strategy,
            candidates=replay.candidates,
            entry_liquidity=prepared.entry_liquidity,
            starting_capital=starting_capital,
            allocation_pct=allocation_pct,
            max_open_positions=max_open_positions,
            max_gross_exposure_pct=max_gross_exposure_pct,
            round_trip_cost_bps=round_trip_cost_bps,
            max_entry_day_participation_pct=max_entry_day_participation_pct,
            max_expected_downside=spec.max_expected_downside,
        )
        scorecard = _scorecard(backtest)
        counterfactuals["leave_one_horizon_out"][str(horizon)] = {
            "scorecard": scorecard,
            "delta_vs_baseline": _scorecard_delta(baseline_scorecard, scorecard),
            "execution_diagnostics": diagnostics,
        }

    for family, feature_name in TRIGGER_FEATURES.items():
        candidates = tuple(
            candidate
            for candidate in replay.candidates
            if not _feature_is_active(candidate.snapshot.features, feature_name)
        )
        backtest, diagnostics = _execute_replay(
            strategy=_AnnualV5StrategyRouter(replay.strategies),
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
        scorecard = _scorecard(backtest)
        counterfactuals["leave_one_trigger_family_out"][family] = {
            "candidate_count": len(candidates),
            "candidate_count_removed": len(replay.candidates) - len(candidates),
            "scorecard": scorecard,
            "delta_vs_baseline": _scorecard_delta(baseline_scorecard, scorecard),
            "execution_diagnostics": diagnostics,
        }

    concentration = _alpha_concentration(by_horizon, by_year)
    top_losses = _top_alpha_loss_sources(
        {
            "horizon": by_horizon,
            "entry_year": by_year,
            "horizon_year": horizon_year,
            "score_decile": score_deciles,
            "trigger_family": trigger_attribution["overlapping_families"],
            "trigger_signature": trigger_attribution["exclusive_signatures"],
        }
    )
    remaining_alpha_shape = _remaining_alpha_shape(
        baseline_scorecard.get("average_trade_alpha"),
        concentration,
    )

    payload = _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_dir": str(root),
            "runtime_dir": str(runtime_root),
            "market_db": str(resolved_market_db),
            "benchmark_security_id": resolved_benchmark,
            "question": (
                "Is the remaining V5 average-trade-alpha deficit broad across the "
                "strategy, or concentrated in an identifiable horizon, period, score "
                "bucket, or trigger family?"
            ),
            "parameter_search": False,
            "strategy_thresholds_changed": False,
            "paper_policy_changed": False,
            "baseline_scenario": BASELINE_SCENARIO,
            "baseline_training_adv_prefilter": False,
            "strategy_spec": spec.as_json(),
            "data": {
                "source_row_count": len(profiled_rows),
                "candidate_count": len(replay.candidates),
                "model_years": sorted(replay.strategies),
                "invalid_target_count": len(prepared.invalid_target_keys),
                "quality_exclusion_count": prepared.quality_exclusion_count,
                "market_cache_stats": prepared.market_cache_stats,
            },
            "baseline": {
                "scorecard": baseline_scorecard,
                "execution_diagnostics": baseline_execution,
                "trade_summary": _trade_summary(trades),
            },
            "attribution": {
                "by_horizon": by_horizon,
                "by_entry_year": by_year,
                "by_horizon_year": horizon_year,
                "by_score_decile": score_deciles,
                "score_decile_note": (
                    "Executed trades are sorted globally by opportunity_score and split "
                    "into equal-count deciles from D01 (lowest) to D10 (highest). These "
                    "are descriptive score buckets, not retuned selection thresholds."
                ),
                "triggers": trigger_attribution,
                "alpha_concentration": {
                    **concentration,
                    "classification_rule": {
                        "concentrated_by_horizon_if_share_at_least": 0.60,
                        "concentrated_by_period_if_share_at_least": 0.40,
                    },
                },
                "top_alpha_loss_sources": top_losses,
                "remaining_alpha_shape": remaining_alpha_shape,
            },
            "counterfactuals": counterfactuals,
            "interpretation": _interpret(
                baseline_scorecard,
                concentration,
                remaining_alpha_shape,
            ),
        }
    )
    output_path = root / "v5_edge_attribution.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return V5EdgeAttributionResult(
        output_path=output_path,
        average_trade_alpha=baseline_scorecard.get("average_trade_alpha"),
        remaining_alpha_shape=remaining_alpha_shape,
    )


def _execute_replay(
    *,
    strategy: Any,
    candidates: tuple,
    entry_liquidity: Mapping[str, Any],
    starting_capital: float,
    allocation_pct: float,
    max_open_positions: int,
    max_gross_exposure_pct: float,
    round_trip_cost_bps: float,
    max_entry_day_participation_pct: float,
    max_expected_downside: float,
) -> tuple[Any, dict[str, int]]:
    backtester = ExecutionRealisticHistoricalBacktester(
        starting_capital=starting_capital,
        round_trip_cost_bps=round_trip_cost_bps,
    )
    backtest, diagnostics = backtester.run(
        strategy=strategy,
        candidates=candidates,
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
    return backtest, {
        "rejected_entry_liquidity": int(diagnostics.rejected_entry_liquidity),
        "rejected_cash": int(backtest.rejected_cash),
    }


def _trade_summary(trades: Iterable[Any]) -> dict[str, Any]:
    materialized = tuple(trades)
    if not materialized:
        return {
            "trade_count": 0,
            "total_pnl": 0.0,
            "profit_factor": 0.0,
            "win_rate": None,
            "average_gross_stock_return": None,
            "average_net_trade_return": None,
            "average_benchmark_return": None,
            "average_trade_alpha": None,
            "positive_alpha_rate": None,
            "alpha_sum": 0.0,
            "capital_weighted_alpha_sum": 0.0,
        }

    positive_pnl = sum(trade.pnl for trade in materialized if trade.pnl > 0)
    negative_pnl = -sum(trade.pnl for trade in materialized if trade.pnl < 0)
    profit_factor = (
        positive_pnl / negative_pnl
        if negative_pnl > 0
        else (inf if positive_pnl > 0 else 0.0)
    )
    count = len(materialized)
    return {
        "trade_count": count,
        "total_pnl": sum(trade.pnl for trade in materialized),
        "profit_factor": profit_factor,
        "win_rate": sum(trade.net_return > 0 for trade in materialized) / count,
        "average_gross_stock_return": (
            sum(trade.gross_return for trade in materialized) / count
        ),
        "average_net_trade_return": (
            sum(trade.net_return for trade in materialized) / count
        ),
        "average_benchmark_return": (
            sum(trade.gross_return - trade.alpha for trade in materialized) / count
        ),
        "average_trade_alpha": sum(trade.alpha for trade in materialized) / count,
        "positive_alpha_rate": sum(trade.alpha > 0 for trade in materialized) / count,
        "alpha_sum": sum(trade.alpha for trade in materialized),
        "capital_weighted_alpha_sum": sum(
            trade.allocated_capital * trade.alpha
            for trade in materialized
        ),
    }


def _grouped_trade_summaries(
    trades: Iterable[Any],
    key_fn: Callable[[Any], str],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Any]] = defaultdict(list)
    for trade in trades:
        groups[key_fn(trade)].append(trade)
    return {
        key: _trade_summary(group)
        for key, group in sorted(groups.items())
    }


def _score_deciles(trades: Iterable[Any]) -> dict[str, dict[str, Any]]:
    materialized = tuple(trades)
    if not materialized:
        return {}
    ordered = sorted(
        materialized,
        key=lambda trade: (trade.opportunity_score, trade.candidate_id),
    )
    groups: dict[str, list[Any]] = defaultdict(list)
    for index, trade in enumerate(ordered):
        bucket = min(9, index * 10 // len(ordered)) + 1
        groups[f"D{bucket:02d}"].append(trade)

    result: dict[str, dict[str, Any]] = {}
    for bucket, group in sorted(groups.items()):
        summary = _trade_summary(group)
        summary["minimum_opportunity_score"] = min(
            trade.opportunity_score for trade in group
        )
        summary["maximum_opportunity_score"] = max(
            trade.opportunity_score for trade in group
        )
        result[bucket] = summary
    return result


def _feature_is_active(features: Mapping[str, Any], name: str) -> bool:
    value = features.get(name)
    if value is None or isinstance(value, bool):
        return value is True
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return False


def _trigger_signature(features: Mapping[str, Any]) -> str:
    active = [
        family
        for family, feature_name in TRIGGER_FEATURES.items()
        if _feature_is_active(features, feature_name)
    ]
    return "+".join(active) if active else "none"


def _trigger_attribution(
    trades: Iterable[Any],
    candidate_by_id: Mapping[str, Any],
) -> dict[str, Any]:
    materialized = tuple(trades)
    overlapping: dict[str, dict[str, Any]] = {}
    for family, feature_name in TRIGGER_FEATURES.items():
        family_trades = tuple(
            trade
            for trade in materialized
            if _feature_is_active(
                candidate_by_id[trade.candidate_id].snapshot.features,
                feature_name,
            )
        )
        overlapping[family] = _trade_summary(family_trades)

    signatures = _grouped_trade_summaries(
        materialized,
        lambda trade: _trigger_signature(
            candidate_by_id[trade.candidate_id].snapshot.features
        ),
    )
    return {
        "overlapping_families": overlapping,
        "exclusive_signatures": signatures,
        "overlapping_family_note": (
            "A trade can appear in more than one family when multiple trigger flags "
            "are active. Exclusive signatures are mutually exclusive."
        ),
    }


def _alpha_concentration(
    by_horizon: Mapping[str, Mapping[str, Any]],
    by_year: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "horizon": _negative_alpha_concentration(by_horizon),
        "entry_year": _negative_alpha_concentration(by_year),
    }


def _negative_alpha_concentration(
    groups: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    negative = [
        (key, float(summary["alpha_sum"]))
        for key, summary in groups.items()
        if float(summary["alpha_sum"]) < 0
    ]
    if not negative:
        return {
            "negative_group_count": 0,
            "largest_negative_group": None,
            "largest_negative_alpha_sum": 0.0,
            "largest_share_of_negative_alpha_sum": None,
        }
    total_magnitude = sum(-value for _, value in negative)
    worst_key, worst_value = min(negative, key=lambda item: item[1])
    return {
        "negative_group_count": len(negative),
        "largest_negative_group": worst_key,
        "largest_negative_alpha_sum": worst_value,
        "largest_share_of_negative_alpha_sum": (
            -worst_value / total_magnitude if total_magnitude > 0 else None
        ),
    }


def _top_alpha_loss_sources(
    dimensions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    limit: int = 15,
) -> list[dict[str, Any]]:
    losses: list[dict[str, Any]] = []
    for dimension, groups in dimensions.items():
        for key, summary in groups.items():
            alpha_sum = float(summary["alpha_sum"])
            if alpha_sum >= 0:
                continue
            losses.append(
                {
                    "dimension": dimension,
                    "key": key,
                    "trade_count": int(summary["trade_count"]),
                    "average_trade_alpha": summary["average_trade_alpha"],
                    "alpha_sum": alpha_sum,
                    "capital_weighted_alpha_sum": summary[
                        "capital_weighted_alpha_sum"
                    ],
                }
            )
    losses.sort(key=lambda item: (item["alpha_sum"], item["dimension"], item["key"]))
    return losses[:limit]


def _scorecard_delta(
    baseline: Mapping[str, Any],
    counterfactual: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_alpha = baseline.get("average_trade_alpha")
    current_alpha = counterfactual.get("average_trade_alpha")
    return {
        "total_return_delta": (
            float(counterfactual["total_return"]) - float(baseline["total_return"])
        ),
        "profit_factor_delta": (
            float(counterfactual["profit_factor"]) - float(baseline["profit_factor"])
        ),
        "average_trade_alpha_delta": (
            float(current_alpha) - float(baseline_alpha)
            if current_alpha is not None and baseline_alpha is not None
            else None
        ),
        "trade_count_delta": (
            int(counterfactual["total_trades"]) - int(baseline["total_trades"])
        ),
    }


def _remaining_alpha_shape(
    average_alpha: float | None,
    concentration: Mapping[str, Mapping[str, Any]],
) -> str:
    if average_alpha is None:
        return "no_trades"
    if average_alpha >= 0:
        return "no_remaining_alpha_deficit"

    horizon = concentration["horizon"]
    year = concentration["entry_year"]
    horizon_share = horizon.get("largest_share_of_negative_alpha_sum")
    year_share = year.get("largest_share_of_negative_alpha_sum")
    if horizon_share is not None and horizon_share >= 0.60:
        return "concentrated_by_horizon"
    if year_share is not None and year_share >= 0.40:
        return "concentrated_by_period"
    return "broad_or_multi_component"


def _interpret(
    baseline: Mapping[str, Any],
    concentration: Mapping[str, Mapping[str, Any]],
    shape: str,
) -> str:
    alpha = baseline.get("average_trade_alpha")
    if alpha is None:
        return "The corrected execution-only replay produced no trades to attribute."
    if alpha >= 0:
        return (
            "The corrected execution-only replay now has non-negative average trade "
            "alpha; attribution should be treated as robustness analysis rather than "
            "diagnosis of an alpha deficit."
        )
    worst_horizon = concentration["horizon"].get("largest_negative_group")
    worst_year = concentration["entry_year"].get("largest_negative_group")
    return (
        f"The residual average trade alpha is {float(alpha):.6f}. Its diagnostic "
        f"shape is {shape}; the largest negative horizon group is {worst_horizon} "
        f"and the largest negative entry-year group is {worst_year}. Use the "
        "leave-one-component-out scorecards to distinguish descriptive concentration "
        "from portfolio interaction effects before changing V5."
    )


def _compact_console(payload: Mapping[str, Any]) -> dict[str, Any]:
    attribution = payload["attribution"]
    return {
        "schema_version": payload["schema_version"],
        "question": payload["question"],
        "parameter_search": payload["parameter_search"],
        "strategy_thresholds_changed": payload["strategy_thresholds_changed"],
        "baseline_scenario": payload["baseline_scenario"],
        "baseline": payload["baseline"],
        "by_horizon": attribution["by_horizon"],
        "by_entry_year": attribution["by_entry_year"],
        "by_score_decile": attribution["by_score_decile"],
        "triggers": attribution["triggers"],
        "alpha_concentration": attribution["alpha_concentration"],
        "top_alpha_loss_sources": attribution["top_alpha_loss_sources"],
        "remaining_alpha_shape": attribution["remaining_alpha_shape"],
        "counterfactuals": payload["counterfactuals"],
        "interpretation": payload["interpretation"],
        "output_path": str(
            Path(payload["experiment_dir"]) / "v5_edge_attribution.json"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute the residual alpha deficit in the corrected fixed-spec V5 "
            "horizon-quality execution-only replay."
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
    result = run_v5_edge_attribution(
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
