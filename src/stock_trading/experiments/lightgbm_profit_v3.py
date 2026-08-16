from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stock_trading.backtest import BacktestConfig, FixedAllocationBacktester, summarize_walk_forward
from stock_trading.backtest.portfolio import ScoredCandidate
from stock_trading.ml import LightGbmTrainer, LightGbmTrainingConfig, OpportunityPrediction, ProfitLightGbmTrainer
from stock_trading.ml.score_calibration import rolling_score_percentiles
from stock_trading.ml.system_context import SYSTEM_CONTEXT_FEATURES, augment_system_context_features
from stock_trading.ml.walk_forward import WalkForwardResult, annual_walk_forward_splits

from .lightgbm_diagnostics import _load_training_rows
from .lightgbm_profit import _average, _predict_profit_matrix
from .lightgbm_validation_rank import _json_safe


@dataclass(frozen=True, slots=True)
class ProfitTargetV3ExperimentResult:
    training_row_count: int
    model_years: tuple[int, ...]
    output_path: Path


def run_profit_target_v3_experiment(
    experiment_dir: str | Path,
    *,
    validation_top_fraction: float = 0.05,
    alpha_rank_weight: float = 0.25,
    calibration_window_days: int = 365,
    max_expected_downside: float = 0.06,
    starting_capital: float = 10_000.0,
    allocation_pct: float = 0.02,
    max_open_positions: int = 15,
    round_trip_cost_bps: float = 20.0,
    training_config: LightGbmTrainingConfig | None = None,
) -> ProfitTargetV3ExperimentResult:
    """Broad engineering iteration combining model, context and ranking fixes.

    V3 intentionally bundles obvious low-cost improvements instead of isolating
    each feature scientifically:
    - V2 same-company history plus market/regime/cross-sectional context,
    - a secondary alpha model as a scale-free ranking signal,
    - rolling PIT score-percentile calibration instead of one raw prior-year cutoff,
    - an absolute predicted-return floor equal to transaction costs,
    - the existing one-active-position-per-company portfolio constraint.

    Existing V1/V2 artifacts are read only as baselines and remain untouched.
    """

    if not 0 < validation_top_fraction < 1:
        raise ValueError("validation_top_fraction must be in (0, 1)")
    if not 0 <= alpha_rank_weight <= 1:
        raise ValueError("alpha_rank_weight must be in [0, 1]")
    if calibration_window_days <= 0:
        raise ValueError("calibration_window_days must be > 0")
    if max_expected_downside < 0:
        raise ValueError("max_expected_downside must be >= 0")
    if round_trip_cost_bps < 0:
        raise ValueError("round_trip_cost_bps must be >= 0")

    root = Path(experiment_dir)
    rows_path = root / "training_rows.jsonl"
    if not rows_path.exists():
        raise FileNotFoundError(f"missing training rows: {rows_path}")

    base_rows = _load_training_rows(rows_path)
    rows = augment_system_context_features(base_rows)
    splits = annual_walk_forward_splits(rows)
    if not splits:
        raise ValueError("no walk-forward splits available")

    profitable_return_threshold = round_trip_cost_bps / 10_000.0
    profit_trainer = ProfitLightGbmTrainer(training_config)
    alpha_trainer = LightGbmTrainer(training_config)
    backtester = FixedAllocationBacktester(
        BacktestConfig(
            starting_capital=starting_capital,
            allocation_pct=allocation_pct,
            max_open_positions=max_open_positions,
            min_expected_alpha=-1_000_000.0,
            min_probability_positive=0.0,
            max_expected_downside=max_expected_downside,
            round_trip_cost_bps=round_trip_cost_bps,
        )
    )

    models_root = root / "profit_models_v3"
    observed_results: list[WalkForwardResult] = []
    year_reports: list[dict] = []
    profit_importance: dict[str, float] = {}
    alpha_importance: dict[str, float] = {}

    for split in splits:
        profit_model = profit_trainer.train(
            split.train_rows,
            split.validation_rows,
            profitable_return_threshold=profitable_return_threshold,
        )
        alpha_model = alpha_trainer.train(split.train_rows, split.validation_rows)
        profit_model.save(models_root / str(split.test_year) / "profit")
        alpha_model.save(models_root / str(split.test_year) / "alpha")

        _accumulate_normalized_importance(
            profit_importance,
            profit_model.feature_importance(),
        )
        _accumulate_normalized_importance(
            alpha_importance,
            alpha_model.feature_importance(),
        )

        validation_profit = _predict_profit_matrix(profit_model, split.validation_rows)
        test_profit = _predict_profit_matrix(profit_model, split.test_rows)
        validation_alpha = _predict_alpha(alpha_model, split.validation_rows)
        test_alpha = _predict_alpha(alpha_model, split.test_rows)

        profit_percentile = rolling_score_percentiles(
            split.validation_rows,
            validation_profit[3],
            split.test_rows,
            test_profit[3],
            window_days=calibration_window_days,
        )
        alpha_percentile = rolling_score_percentiles(
            split.validation_rows,
            validation_alpha,
            split.test_rows,
            test_alpha,
            window_days=calibration_window_days,
        )
        combined_rank = (
            (1.0 - alpha_rank_weight) * profit_percentile
            + alpha_rank_weight * alpha_percentile
        )
        rank_threshold = 1.0 - validation_top_fraction
        expected_return, downside, probability, raw_profit_score = test_profit

        selected: list[ScoredCandidate] = []
        selected_indices: list[int] = []
        rejected_below_cost_floor = 0
        rejected_below_rank = 0
        for index, row in enumerate(split.test_rows):
            if combined_rank[index] < rank_threshold:
                rejected_below_rank += 1
                continue
            if expected_return[index] < profitable_return_threshold:
                rejected_below_cost_floor += 1
                continue
            selected_indices.append(index)
            selected.append(
                ScoredCandidate(
                    row=row,
                    prediction=OpportunityPrediction(
                        expected_alpha_20d=float(expected_return[index]),
                        expected_downside_20d=float(downside[index]),
                        probability_positive_alpha=float(probability[index]),
                        opportunity_score=float(combined_rank[index]),
                    ),
                )
            )

        portfolio = backtester.run(tuple(selected))
        observed_results.append(
            WalkForwardResult(
                test_year=split.test_year,
                train_count=len(split.train_rows),
                validation_count=len(split.validation_rows),
                test_count=len(split.test_rows),
                backtest=portfolio,
            )
        )

        selected_rows = [split.test_rows[index] for index in selected_indices]
        trades = portfolio.trades
        year_reports.append(
            {
                "year": split.test_year,
                "test_count": len(split.test_rows),
                "rank_eligible": int((combined_rank >= rank_threshold).sum()),
                "selected_after_absolute_return_floor": len(selected),
                "rejected_below_rank": rejected_below_rank,
                "rejected_below_cost_floor_after_rank": rejected_below_cost_floor,
                "trades": len(trades),
                "return": portfolio.total_return,
                "profit_factor": portfolio.profit_factor,
                "realized_drawdown": portfolio.realized_max_drawdown,
                "rejected_duplicate_company": portfolio.rejected_duplicate_company,
                "rejected_capacity": portfolio.rejected_capacity,
                "average_combined_rank_selected": _average(
                    [float(combined_rank[index]) for index in selected_indices]
                ),
                "average_raw_profit_score_selected": _average(
                    [float(raw_profit_score[index]) for index in selected_indices]
                ),
                "selected_average_stock_return_20d": _average(
                    [row.stock_return_20d for row in selected_rows]
                ),
                "selected_average_alpha_20d": _average(
                    [row.alpha_20d for row in selected_rows]
                ),
                "trade_average_stock_return_20d": _average(
                    [trade.gross_return for trade in trades]
                ),
                "trade_average_alpha_20d": _average(
                    [trade.alpha_20d for trade in trades]
                ),
            }
        )

    observed_summary = summarize_walk_forward(observed_results)
    all_trades = [trade for result in observed_results for trade in result.backtest.trades]
    observed = {
        "compounded_return": observed_summary.compounded_return,
        "profitable_year_rate": observed_summary.profitable_year_rate,
        "total_trades": observed_summary.total_trades,
        "average_trade_stock_return": _average([trade.gross_return for trade in all_trades]),
        "average_trade_alpha": observed_summary.average_trade_alpha,
        "aggregate_profit_factor": observed_summary.aggregate_profit_factor,
        "worst_realized_drawdown": observed_summary.worst_realized_drawdown,
    }

    payload = _json_safe(
        {
            "schema_version": "profit-target-lightgbm-v3-system-context",
            "experiment_dir": str(root),
            "training_row_count": len(rows),
            "model_years": [split.test_year for split in splits],
            "development_mode": {
                "kind": "broad_engineering_iteration",
                "permutation_null_run": False,
                "reason": "fast architecture screen before another fresh holdout",
            },
            "feature_augmentation": {
                "kind": "company_history_plus_market_regime_plus_cross_section",
                "uses_realized_labels": False,
                "system_features": list(SYSTEM_CONTEXT_FEATURES),
            },
            "ranking_policy": {
                "profit_rank_weight": 1.0 - alpha_rank_weight,
                "alpha_rank_weight": alpha_rank_weight,
                "rolling_calibration_window_days": calibration_window_days,
                "rank_threshold": 1.0 - validation_top_fraction,
                "calibration_uses_outcomes": False,
                "absolute_expected_return_floor": profitable_return_threshold,
                "absolute_floor_reason": "must clear modeled round-trip costs",
            },
            "portfolio_policy": {
                "max_expected_downside": max_expected_downside,
                "starting_capital": starting_capital,
                "allocation_pct": allocation_pct,
                "max_open_positions": max_open_positions,
                "max_active_positions_per_company": 1,
                "round_trip_cost_bps": round_trip_cost_bps,
            },
            "v1_baseline": _load_baseline(root / "profit_target_backtest.json"),
            "v2_baseline": _load_baseline(root / "profit_target_v2_backtest.json"),
            "observed": observed,
            "feature_importance": {
                "profit_return_head_top30": _top_importance(profit_importance, 30),
                "alpha_head_top30": _top_importance(alpha_importance, 30),
            },
            "years": year_reports,
        }
    )
    output_path = root / "profit_target_v3_backtest.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return ProfitTargetV3ExperimentResult(
        training_row_count=len(rows),
        model_years=tuple(split.test_year for split in splits),
        output_path=output_path,
    )


def _predict_alpha(model, rows) -> np.ndarray:
    matrix = model.feature_schema.matrix(rows)
    return np.asarray(model.alpha_model.predict(matrix), dtype=np.float64)


def _accumulate_normalized_importance(
    destination: dict[str, float],
    importance: dict[str, float],
) -> None:
    total = sum(importance.values())
    if total <= 0:
        return
    for name, value in importance.items():
        destination[name] = destination.get(name, 0.0) + value / total


def _top_importance(values: dict[str, float], limit: int) -> list[dict[str, float | str]]:
    return [
        {"feature": name, "normalized_gain_sum": value}
        for name, value in sorted(values.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _load_baseline(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    observed = payload.get("observed")
    return observed if isinstance(observed, dict) else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the broad LightGBM V3 engineering iteration with PIT company history, "
            "market/cross-sectional context, multi-objective rank fusion and rolling "
            "score calibration."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--validation-top-fraction", type=float, default=0.05)
    parser.add_argument("--alpha-rank-weight", type=float, default=0.25)
    parser.add_argument("--calibration-window-days", type=int, default=365)
    parser.add_argument("--max-expected-downside", type=float, default=0.06)
    parser.add_argument("--starting-capital", type=float, default=10_000.0)
    parser.add_argument("--allocation-pct", type=float, default=0.02)
    parser.add_argument("--max-open-positions", type=int, default=15)
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_profit_target_v3_experiment(
        args.experiment_dir,
        validation_top_fraction=args.validation_top_fraction,
        alpha_rank_weight=args.alpha_rank_weight,
        calibration_window_days=args.calibration_window_days,
        max_expected_downside=args.max_expected_downside,
        starting_capital=args.starting_capital,
        allocation_pct=args.allocation_pct,
        max_open_positions=args.max_open_positions,
        round_trip_cost_bps=args.round_trip_cost_bps,
    )
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
