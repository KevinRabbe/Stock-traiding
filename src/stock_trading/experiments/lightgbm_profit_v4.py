from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import prod
from pathlib import Path

import numpy as np

from stock_trading.backtest import BacktestConfig, FixedAllocationBacktester, summarize_walk_forward
from stock_trading.backtest.portfolio import ScoredCandidate
from stock_trading.ml import LightGbmTrainer, LightGbmTrainingConfig, OpportunityPrediction, ProfitLightGbmTrainer
from stock_trading.ml.score_calibration import (
    rolling_filtered_score_percentiles,
    rolling_score_percentiles,
    static_score_percentiles,
)
from stock_trading.ml.system_context import SYSTEM_CONTEXT_FEATURES, augment_system_context_features
from stock_trading.ml.walk_forward import WalkForwardResult, annual_walk_forward_splits

from .lightgbm_diagnostics import _load_training_rows
from .lightgbm_profit import _average, _predict_profit_matrix
from .lightgbm_profit_v3 import (
    _accumulate_normalized_importance,
    _load_baseline,
    _predict_alpha,
    _top_importance,
)
from .lightgbm_validation_rank import _json_safe


@dataclass(frozen=True, slots=True)
class ProfitTargetV4ExperimentResult:
    training_row_count: int
    model_years: tuple[int, ...]
    output_path: Path


def run_profit_target_v4_experiment(
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
) -> ProfitTargetV4ExperimentResult:
    """V4 fixes V3 selection compression while retaining the broad V3 model context.

    V3 fused two separately calibrated head percentiles and compared their weighted
    average directly with a 95th-percentile threshold. That is substantially more
    selective than taking the top 5% of the *combined* signal when the heads are not
    perfectly correlated. It also ranked candidates before the hard downside gate,
    allowing untradeable rows to consume the scarce high-rank tail.

    V4 therefore:
    - retains V3 company-history, regime and cross-sectional model features,
    - retains the profit + alpha multi-objective heads,
    - builds the weighted combined signal first,
    - applies the cost/downside hard eligibility gates before final calibration,
    - calibrates the combined signal itself over only historically tradable rows,
    - selects the requested top fraction from that final PIT combined percentile,
    - keeps one active position per company and the existing portfolio sizing.

    No realized labels are used by calibration or selection.
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

    rows = augment_system_context_features(_load_training_rows(rows_path))
    splits = annual_walk_forward_splits(rows)
    if not splits:
        raise ValueError("no walk-forward splits available")

    profitable_return_threshold = round_trip_cost_bps / 10_000.0
    rank_threshold = 1.0 - validation_top_fraction
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

    models_root = root / "profit_models_v4"
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

        _accumulate_normalized_importance(profit_importance, profit_model.feature_importance())
        _accumulate_normalized_importance(alpha_importance, alpha_model.feature_importance())

        validation_profit = _predict_profit_matrix(profit_model, split.validation_rows)
        test_profit = _predict_profit_matrix(profit_model, split.test_rows)
        validation_alpha = _predict_alpha(alpha_model, split.validation_rows)
        test_alpha = _predict_alpha(alpha_model, split.test_rows)

        # Validation is wholly historical relative to the test year, so its score
        # distribution can seed the final combined-signal calibration without any
        # test-year information or realized outcome labels.
        validation_profit_percentile = static_score_percentiles(validation_profit[3])
        validation_alpha_percentile = static_score_percentiles(validation_alpha)
        validation_combined_signal = (
            (1.0 - alpha_rank_weight) * validation_profit_percentile
            + alpha_rank_weight * validation_alpha_percentile
        )

        # Test head ranks remain PIT: each execution date sees validation plus only
        # strictly earlier test dates, never same-day or future scores.
        test_profit_percentile = rolling_score_percentiles(
            split.validation_rows,
            validation_profit[3],
            split.test_rows,
            test_profit[3],
            window_days=calibration_window_days,
        )
        test_alpha_percentile = rolling_score_percentiles(
            split.validation_rows,
            validation_alpha,
            split.test_rows,
            test_alpha,
            window_days=calibration_window_days,
        )
        test_combined_signal = (
            (1.0 - alpha_rank_weight) * test_profit_percentile
            + alpha_rank_weight * test_alpha_percentile
        )

        validation_expected_return, validation_downside, _, _ = validation_profit
        expected_return, downside, probability, raw_profit_score = test_profit
        validation_eligible = (
            (validation_expected_return >= profitable_return_threshold)
            & (validation_downside <= max_expected_downside)
        )
        test_eligible = (
            (expected_return >= profitable_return_threshold)
            & (downside <= max_expected_downside)
        )

        # This second calibration is the key V4 change: 0.95 now means the top 5%
        # of the *combined, tradable* signal distribution instead of requiring a
        # weighted average of two independent percentile ranks to itself exceed .95.
        final_percentile = rolling_filtered_score_percentiles(
            split.validation_rows,
            validation_combined_signal,
            validation_eligible,
            split.test_rows,
            test_combined_signal,
            test_eligible,
            window_days=calibration_window_days,
            ineligible_percentile=0.0,
        )

        selected_indices = [
            index
            for index in range(len(split.test_rows))
            if test_eligible[index] and final_percentile[index] >= rank_threshold
        ]
        selected = tuple(
            ScoredCandidate(
                row=split.test_rows[index],
                prediction=OpportunityPrediction(
                    expected_alpha_20d=float(expected_return[index]),
                    expected_downside_20d=float(downside[index]),
                    probability_positive_alpha=float(probability[index]),
                    opportunity_score=float(final_percentile[index]),
                ),
            )
            for index in selected_indices
        )

        portfolio = backtester.run(selected)
        if portfolio.rejected_by_signal != 0:
            raise RuntimeError(
                "V4 pre-ranking safety gates diverged from backtester signal gates"
            )
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
        below_cost = expected_return < profitable_return_threshold
        above_downside_after_cost = (~below_cost) & (downside > max_expected_downside)
        safe_count = int(test_eligible.sum())
        rank_eligible_count = len(selected_indices)
        year_reports.append(
            {
                "year": split.test_year,
                "test_count": len(split.test_rows),
                "eligible_after_cost_and_downside": safe_count,
                "rank_eligible": rank_eligible_count,
                "rank_eligible_fraction_of_safe": (
                    rank_eligible_count / safe_count if safe_count else None
                ),
                "rejected_below_expected_return_cost_floor": int(below_cost.sum()),
                "rejected_above_downside_after_cost_floor": int(
                    above_downside_after_cost.sum()
                ),
                "rejected_below_final_combined_rank": safe_count - rank_eligible_count,
                "rejected_by_signal": portfolio.rejected_by_signal,
                "rejected_duplicate_company": portfolio.rejected_duplicate_company,
                "rejected_capacity": portfolio.rejected_capacity,
                "trades": len(trades),
                "return": portfolio.total_return,
                "profit_factor": portfolio.profit_factor,
                "realized_drawdown": portfolio.realized_max_drawdown,
                "average_final_percentile_selected": _average(
                    [float(final_percentile[index]) for index in selected_indices]
                ),
                "average_combined_signal_selected": _average(
                    [float(test_combined_signal[index]) for index in selected_indices]
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

    best_year = max(year_reports, key=lambda item: float(item["return"]))
    compounded_without_best_year = (
        prod(
            1.0 + float(item["return"])
            for item in year_reports
            if item["year"] != best_year["year"]
        )
        - 1.0
    )
    concentration = {
        "best_year": best_year["year"],
        "best_year_return": best_year["return"],
        "best_year_trade_fraction": (
            best_year["trades"] / observed_summary.total_trades
            if observed_summary.total_trades
            else None
        ),
        "compounded_return_excluding_best_year": compounded_without_best_year,
    }

    payload = _json_safe(
        {
            "schema_version": "profit-target-lightgbm-v4-selection-architecture",
            "experiment_dir": str(root),
            "training_row_count": len(rows),
            "model_years": [split.test_year for split in splits],
            "development_mode": {
                "kind": "fast_system_engineering",
                "permutation_null_run": False,
                "reason": "repair V3 selection compression before larger data/schema expansion",
            },
            "feature_augmentation": {
                "kind": "same_v3_company_history_market_regime_cross_section",
                "uses_realized_labels": False,
                "system_features": list(SYSTEM_CONTEXT_FEATURES),
            },
            "ranking_policy": {
                "profit_rank_weight": 1.0 - alpha_rank_weight,
                "alpha_rank_weight": alpha_rank_weight,
                "rolling_calibration_window_days": calibration_window_days,
                "final_combined_rank_threshold": rank_threshold,
                "final_rank_meaning": "top fraction of combined signal among cost/downside eligible historical candidates",
                "calibration_uses_outcomes": False,
                "expected_return_cost_floor": profitable_return_threshold,
                "max_expected_downside_before_rank": max_expected_downside,
            },
            "portfolio_policy": {
                "starting_capital": starting_capital,
                "allocation_pct": allocation_pct,
                "max_open_positions": max_open_positions,
                "max_active_positions_per_company": 1,
                "round_trip_cost_bps": round_trip_cost_bps,
            },
            "v1_baseline": _load_baseline(root / "profit_target_backtest.json"),
            "v2_baseline": _load_baseline(root / "profit_target_v2_backtest.json"),
            "v3_baseline": _load_baseline(root / "profit_target_v3_backtest.json"),
            "observed": observed,
            "concentration": concentration,
            "feature_importance": {
                "profit_return_head_top30": _top_importance(profit_importance, 30),
                "alpha_head_top30": _top_importance(alpha_importance, 30),
            },
            "years": year_reports,
        }
    )
    output_path = root / "profit_target_v4_backtest.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return ProfitTargetV4ExperimentResult(
        training_row_count=len(rows),
        model_years=tuple(split.test_year for split in splits),
        output_path=output_path,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run LightGBM V4 with V3 context plus safety-first combined-signal "
            "calibration and a stable top-fraction selection layer."
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
    result = run_profit_target_v4_experiment(
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
