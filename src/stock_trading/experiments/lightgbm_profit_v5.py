from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from math import prod
from pathlib import Path

import numpy as np

from stock_trading.backtest import BacktestConfig, FixedAllocationBacktester, summarize_walk_forward
from stock_trading.backtest.portfolio import ScoredCandidate
from stock_trading.market import DuckDbMarketStore
from stock_trading.ml import LightGbmTrainer, LightGbmTrainingConfig, OpportunityPrediction, ProfitLightGbmTrainer
from stock_trading.ml.multi_horizon import build_multi_horizon_targets, row_for_horizon
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


_HORIZONS = (5, 20, 60)


@dataclass(frozen=True, slots=True)
class ProfitTargetV5ExperimentResult:
    source_training_row_count: int
    complete_multi_horizon_row_count: int
    model_years: tuple[int, ...]
    output_path: Path


def run_profit_target_v5_experiment(
    experiment_dir: str | Path,
    *,
    market_db: str | Path,
    benchmark_security_id: str,
    validation_top_fraction: float = 0.05,
    alpha_rank_weight: float = 0.25,
    calibration_window_days: int = 365,
    max_expected_downside: float = 0.06,
    starting_capital: float = 10_000.0,
    allocation_pct: float = 0.02,
    max_open_positions: int = 15,
    round_trip_cost_bps: float = 20.0,
    market_read_cache_series: int = 160,
    training_config: LightGbmTrainingConfig | None = None,
) -> ProfitTargetV5ExperimentResult:
    """Broad V5 step: V4 selection plus adaptive 5/20/60-session holding horizons.

    The original dataset builder already generated several forward horizons but the
    persisted TrainingRow retained only 20 sessions. V5 reconstructs 5/20/60 labels
    from the existing local market database, verifies the reconstructed 20-session
    target against the persisted row, and then trains one profit + alpha pair per
    horizon.

    Horizon choice is prediction-only. For each opportunity, V5 chooses the safest
    eligible horizon with the strongest calibrated multi-objective signal, then
    applies V4's final rolling calibration to the chosen signal. Realized labels are
    used only after selection by the backtester.
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
    if market_read_cache_series <= 0:
        raise ValueError("market_read_cache_series must be > 0")

    root = Path(experiment_dir)
    rows_path = root / "training_rows.jsonl"
    if not rows_path.exists():
        raise FileNotFoundError(f"missing training rows: {rows_path}")

    source_rows = _load_training_rows(rows_path)
    market_store = DuckDbMarketStore(market_db)
    market_store.enable_read_cache(max_series=market_read_cache_series)
    targets = build_multi_horizon_targets(
        source_rows,
        market_store,
        benchmark_security_id=benchmark_security_id,
        horizons=_HORIZONS,
        verify_existing_20d=True,
    )
    complete_rows = tuple(row for row in source_rows if row.event_id in targets)
    if not complete_rows:
        raise ValueError("no rows have complete 5/20/60-session labels")

    rows = augment_system_context_features(complete_rows)
    splits = annual_walk_forward_splits(rows)
    if not splits:
        raise ValueError("no walk-forward splits available after multi-horizon maturity filter")

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

    models_root = root / "profit_models_v5"
    observed_results: list[WalkForwardResult] = []
    year_reports: list[dict] = []
    selected_horizons_global: Counter[int] = Counter()
    trade_horizons_global: Counter[int] = Counter()
    profit_importance: dict[int, dict[str, float]] = {horizon: {} for horizon in _HORIZONS}
    alpha_importance: dict[int, dict[str, float]] = {horizon: {} for horizon in _HORIZONS}

    for split in splits:
        validation_signals: dict[int, np.ndarray] = {}
        test_signals: dict[int, np.ndarray] = {}
        validation_eligible: dict[int, np.ndarray] = {}
        test_eligible: dict[int, np.ndarray] = {}
        validation_expected_returns: dict[int, np.ndarray] = {}
        test_predictions: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

        for horizon in _HORIZONS:
            train_rows_h = tuple(
                row_for_horizon(row, targets[row.event_id][horizon])
                for row in split.train_rows
            )
            validation_rows_h = tuple(
                row_for_horizon(row, targets[row.event_id][horizon])
                for row in split.validation_rows
            )
            test_rows_h = tuple(
                row_for_horizon(row, targets[row.event_id][horizon])
                for row in split.test_rows
            )

            profit_model = profit_trainer.train(
                train_rows_h,
                validation_rows_h,
                profitable_return_threshold=profitable_return_threshold,
            )
            alpha_model = alpha_trainer.train(train_rows_h, validation_rows_h)
            profit_model.save(models_root / str(split.test_year) / f"{horizon}d" / "profit")
            alpha_model.save(models_root / str(split.test_year) / f"{horizon}d" / "alpha")
            _accumulate_normalized_importance(
                profit_importance[horizon],
                profit_model.feature_importance(),
            )
            _accumulate_normalized_importance(
                alpha_importance[horizon],
                alpha_model.feature_importance(),
            )

            validation_profit = _predict_profit_matrix(profit_model, validation_rows_h)
            current_test_profit = _predict_profit_matrix(profit_model, test_rows_h)
            validation_alpha = _predict_alpha(alpha_model, validation_rows_h)
            test_alpha = _predict_alpha(alpha_model, test_rows_h)

            validation_profit_percentile = static_score_percentiles(validation_profit[3])
            validation_alpha_percentile = static_score_percentiles(validation_alpha)
            validation_signals[horizon] = (
                (1.0 - alpha_rank_weight) * validation_profit_percentile
                + alpha_rank_weight * validation_alpha_percentile
            )

            test_profit_percentile = rolling_score_percentiles(
                split.validation_rows,
                validation_profit[3],
                split.test_rows,
                current_test_profit[3],
                window_days=calibration_window_days,
            )
            test_alpha_percentile = rolling_score_percentiles(
                split.validation_rows,
                validation_alpha,
                split.test_rows,
                test_alpha,
                window_days=calibration_window_days,
            )
            test_signals[horizon] = (
                (1.0 - alpha_rank_weight) * test_profit_percentile
                + alpha_rank_weight * test_alpha_percentile
            )

            validation_expected_return, validation_downside, _, _ = validation_profit
            expected_return, downside, _, _ = current_test_profit
            validation_expected_returns[horizon] = validation_expected_return
            validation_eligible[horizon] = (
                (validation_expected_return >= profitable_return_threshold)
                & (validation_downside <= max_expected_downside)
            )
            test_eligible[horizon] = (
                (expected_return >= profitable_return_threshold)
                & (downside <= max_expected_downside)
            )
            test_predictions[horizon] = current_test_profit

        validation_choice, validation_choice_signal, validation_any_eligible = _choose_horizons(
            validation_signals,
            validation_expected_returns,
            validation_eligible,
        )
        test_expected_returns = {
            horizon: test_predictions[horizon][0] for horizon in _HORIZONS
        }
        test_choice, test_choice_signal, test_any_eligible = _choose_horizons(
            test_signals,
            test_expected_returns,
            test_eligible,
        )

        final_percentile = rolling_filtered_score_percentiles(
            split.validation_rows,
            validation_choice_signal,
            validation_any_eligible,
            split.test_rows,
            test_choice_signal,
            test_any_eligible,
            window_days=calibration_window_days,
            ineligible_percentile=0.0,
        )
        selected_indices = [
            index
            for index in range(len(split.test_rows))
            if test_any_eligible[index] and final_percentile[index] >= rank_threshold
        ]

        selected: list[ScoredCandidate] = []
        selected_horizon_by_event: dict[str, int] = {}
        selected_projected_rows = []
        for index in selected_indices:
            row = split.test_rows[index]
            horizon = int(test_choice[index])
            if horizon not in _HORIZONS:
                raise RuntimeError("selected V5 row has no chosen horizon")
            expected_return, downside, probability, _ = test_predictions[horizon]
            projected_row = row_for_horizon(row, targets[row.event_id][horizon])
            selected_projected_rows.append(projected_row)
            selected_horizon_by_event[row.event_id] = horizon
            selected_horizons_global[horizon] += 1
            selected.append(
                ScoredCandidate(
                    row=projected_row,
                    prediction=OpportunityPrediction(
                        expected_alpha_20d=float(expected_return[index]),
                        expected_downside_20d=float(downside[index]),
                        probability_positive_alpha=float(probability[index]),
                        opportunity_score=float(final_percentile[index]),
                    ),
                )
            )

        portfolio = backtester.run(tuple(selected))
        if portfolio.rejected_by_signal != 0:
            raise RuntimeError("V5 pre-ranking safety gates diverged from backtester signal gates")
        observed_results.append(
            WalkForwardResult(
                test_year=split.test_year,
                train_count=len(split.train_rows),
                validation_count=len(split.validation_rows),
                test_count=len(split.test_rows),
                backtest=portfolio,
            )
        )

        trade_horizons = Counter(
            selected_horizon_by_event[trade.event_id] for trade in portfolio.trades
        )
        trade_horizons_global.update(trade_horizons)
        selected_horizons = Counter(
            int(test_choice[index]) for index in selected_indices
        )
        safe_horizons = Counter(
            int(test_choice[index])
            for index in range(len(split.test_rows))
            if test_any_eligible[index]
        )
        trades = portfolio.trades
        year_reports.append(
            {
                "year": split.test_year,
                "test_count": len(split.test_rows),
                "eligible_any_horizon": int(test_any_eligible.sum()),
                "chosen_horizon_counts_safe": _counter_json(safe_horizons),
                "rank_eligible": len(selected_indices),
                "selected_horizon_counts": _counter_json(selected_horizons),
                "trade_horizon_counts": _counter_json(trade_horizons),
                "rejected_below_final_combined_rank": int(test_any_eligible.sum()) - len(selected_indices),
                "rejected_by_signal": portfolio.rejected_by_signal,
                "rejected_duplicate_company": portfolio.rejected_duplicate_company,
                "rejected_capacity": portfolio.rejected_capacity,
                "trades": len(trades),
                "return": portfolio.total_return,
                "profit_factor": portfolio.profit_factor,
                "realized_drawdown": portfolio.realized_max_drawdown,
                "average_selected_horizon": _average(
                    [float(test_choice[index]) for index in selected_indices]
                ),
                "selected_average_stock_return": _average(
                    [row.stock_return_20d for row in selected_projected_rows]
                ),
                "selected_average_alpha": _average(
                    [row.alpha_20d for row in selected_projected_rows]
                ),
                "trade_average_stock_return": _average(
                    [trade.gross_return for trade in trades]
                ),
                "trade_average_alpha": _average(
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
        "selected_horizon_counts": _counter_json(selected_horizons_global),
        "trade_horizon_counts": _counter_json(trade_horizons_global),
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
            "schema_version": "profit-target-lightgbm-v5-adaptive-horizon",
            "experiment_dir": str(root),
            "market_db": str(market_db),
            "benchmark_security_id": benchmark_security_id,
            "source_training_row_count": len(source_rows),
            "complete_multi_horizon_row_count": len(rows),
            "dropped_incomplete_multi_horizon_rows": len(source_rows) - len(rows),
            "model_years": [split.test_year for split in splits],
            "development_mode": {
                "kind": "broad_system_engineering",
                "permutation_null_run": False,
                "reason": "promote fixed 20-session V4 into adaptive multi-horizon holding",
            },
            "target_policy": {
                "horizons_sessions": list(_HORIZONS),
                "horizon_choice": "highest calibrated combined signal among cost/downside eligible horizons",
                "tie_break": "higher expected return, then shorter horizon",
                "reconstructed_from_local_market_db": True,
                "persisted_20d_identity_verified": True,
            },
            "feature_augmentation": {
                "kind": "same_v4_company_history_market_regime_cross_section",
                "uses_realized_labels": False,
                "system_features": list(SYSTEM_CONTEXT_FEATURES),
            },
            "ranking_policy": {
                "profit_rank_weight": 1.0 - alpha_rank_weight,
                "alpha_rank_weight": alpha_rank_weight,
                "rolling_calibration_window_days": calibration_window_days,
                "final_combined_rank_threshold": rank_threshold,
                "calibration_uses_outcomes": False,
                "expected_return_cost_floor_per_horizon": profitable_return_threshold,
                "max_expected_downside_per_horizon": max_expected_downside,
            },
            "portfolio_policy": {
                "starting_capital": starting_capital,
                "allocation_pct": allocation_pct,
                "max_open_positions": max_open_positions,
                "max_active_positions_per_company": 1,
                "round_trip_cost_bps": round_trip_cost_bps,
                "exit_date": "chosen horizon target end date",
            },
            "v4_baseline": _load_baseline(root / "profit_target_v4_backtest.json"),
            "observed": observed,
            "concentration": concentration,
            "feature_importance_by_horizon": {
                str(horizon): {
                    "profit_return_head_top20": _top_importance(profit_importance[horizon], 20),
                    "alpha_head_top20": _top_importance(alpha_importance[horizon], 20),
                }
                for horizon in _HORIZONS
            },
            "market_cache_stats": market_store.read_cache_stats(),
            "years": year_reports,
        }
    )
    output_path = root / "profit_target_v5_backtest.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return ProfitTargetV5ExperimentResult(
        source_training_row_count=len(source_rows),
        complete_multi_horizon_row_count=len(rows),
        model_years=tuple(split.test_year for split in splits),
        output_path=output_path,
    )


def _choose_horizons(
    signals: dict[int, np.ndarray],
    expected_returns: dict[int, np.ndarray],
    eligible: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = {len(values) for values in signals.values()}
    if len(lengths) != 1:
        raise ValueError("horizon signal lengths differ")
    row_count = next(iter(lengths), 0)
    choice = np.full(row_count, -1, dtype=np.int64)
    choice_signal = np.zeros(row_count, dtype=np.float64)
    any_eligible = np.zeros(row_count, dtype=bool)

    for index in range(row_count):
        candidates = [
            horizon for horizon in sorted(signals)
            if bool(eligible[horizon][index])
        ]
        if not candidates:
            continue
        horizon = max(
            candidates,
            key=lambda value: (
                float(signals[value][index]),
                float(expected_returns[value][index]),
                -value,
            ),
        )
        choice[index] = horizon
        choice_signal[index] = float(signals[horizon][index])
        any_eligible[index] = True
    return choice, choice_signal, any_eligible


def _counter_json(values: Counter[int]) -> dict[str, int]:
    return {str(horizon): int(values.get(horizon, 0)) for horizon in _HORIZONS}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run LightGBM V5 with V4 system context/selection and adaptive "
            "5/20/60-session profit horizons reconstructed from the local market DB."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--market-db", type=Path, required=True)
    parser.add_argument("--benchmark-security-id", required=True)
    parser.add_argument("--validation-top-fraction", type=float, default=0.05)
    parser.add_argument("--alpha-rank-weight", type=float, default=0.25)
    parser.add_argument("--calibration-window-days", type=int, default=365)
    parser.add_argument("--max-expected-downside", type=float, default=0.06)
    parser.add_argument("--starting-capital", type=float, default=10_000.0)
    parser.add_argument("--allocation-pct", type=float, default=0.02)
    parser.add_argument("--max-open-positions", type=int, default=15)
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    parser.add_argument("--market-read-cache-series", type=int, default=160)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_profit_target_v5_experiment(
        args.experiment_dir,
        market_db=args.market_db,
        benchmark_security_id=args.benchmark_security_id,
        validation_top_fraction=args.validation_top_fraction,
        alpha_rank_weight=args.alpha_rank_weight,
        calibration_window_days=args.calibration_window_days,
        max_expected_downside=args.max_expected_downside,
        starting_capital=args.starting_capital,
        allocation_pct=args.allocation_pct,
        max_open_positions=args.max_open_positions,
        round_trip_cost_bps=args.round_trip_cost_bps,
        market_read_cache_series=args.market_read_cache_series,
    )
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
