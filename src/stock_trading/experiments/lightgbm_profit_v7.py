from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from stock_trading.backtest import BacktestConfig, FixedAllocationBacktester, summarize_walk_forward
from stock_trading.backtest.portfolio import ScoredCandidate
from stock_trading.backtest.risk_overlay_portfolio import (
    RiskOverlayBacktester,
    RiskOverlayConfig,
)
from stock_trading.market import DuckDbMarketStore
from stock_trading.ml import OpportunityPrediction
from stock_trading.ml.lightgbm_models import LightGbmModelBundle, ProfitLightGbmModelBundle
from stock_trading.ml.multi_horizon import build_multi_horizon_targets, row_for_horizon
from stock_trading.ml.score_calibration import (
    rolling_filtered_score_percentiles,
    rolling_score_percentiles,
    static_score_percentiles,
)
from stock_trading.ml.system_context import augment_system_context_features
from stock_trading.ml.walk_forward import WalkForwardResult, annual_walk_forward_splits

from .lightgbm_diagnostics import _load_training_rows
from .lightgbm_profit import _average, _predict_profit_matrix
from .lightgbm_profit_v3 import _predict_alpha
from .lightgbm_profit_v5 import _HORIZONS, _choose_horizons, _counter_json
from .lightgbm_profit_v6 import _compound_excluding, _weighted_diagnostic_average
from .lightgbm_validation_rank import _json_safe


@dataclass(frozen=True, slots=True)
class ProfitTargetV7ExperimentResult:
    source_training_row_count: int
    complete_multi_horizon_row_count: int
    model_years: tuple[int, ...]
    output_path: Path


def run_profit_target_v7_experiment(
    experiment_dir: str | Path,
    *,
    market_db: str | Path,
    benchmark_security_id: str,
    validation_top_fraction: float = 0.05,
    alpha_rank_weight: float = 0.25,
    calibration_window_days: int = 365,
    max_expected_downside: float = 0.06,
    starting_capital: float = 10_000.0,
    base_allocation_pct: float = 0.02,
    min_allocation_pct: float = 0.01,
    max_gross_exposure_pct: float = 0.20,
    max_open_positions: int = 15,
    correlation_lookback_sessions: int = 60,
    min_correlation_observations: int = 30,
    correlation_penalty_start: float = 0.50,
    high_correlation_threshold: float = 0.75,
    max_correlated_exposure_pct: float = 0.08,
    round_trip_cost_bps: float = 20.0,
    market_read_cache_series: int = 160,
) -> ProfitTargetV7ExperimentResult:
    """Replay V5 selection exactly, then apply a one-way risk-reduction overlay.

    V7 responds to the first V6 allocator: V6 slightly improved headline return but
    worsened PF, drawdown, year consistency and top-three-year concentration because
    it frequently sized above the proven V5 2% allocation. V7 makes 2% a hard
    ceiling. Predicted downside, benchmark regime, volatility expansion and trailing
    correlation may only reduce capital.
    """

    if not 0 < validation_top_fraction < 1:
        raise ValueError("validation_top_fraction must be in (0, 1)")
    if not 0 <= alpha_rank_weight <= 1:
        raise ValueError("alpha_rank_weight must be in [0, 1]")
    if calibration_window_days <= 0:
        raise ValueError("calibration_window_days must be > 0")
    if market_read_cache_series <= 0:
        raise ValueError("market_read_cache_series must be > 0")

    root = Path(experiment_dir)
    rows_path = root / "training_rows.jsonl"
    models_root = root / "profit_models_v5"
    v5_path = root / "profit_target_v5_backtest.json"
    v6_path = root / "profit_target_v6_backtest.json"
    for required in (rows_path, v5_path):
        if not required.exists():
            raise FileNotFoundError(f"missing V7 prerequisite: {required}")
    if not models_root.exists():
        raise FileNotFoundError(f"missing saved V5 models: {models_root}")

    v5_payload = json.loads(v5_path.read_text(encoding="utf-8"))
    v6_payload = (
        json.loads(v6_path.read_text(encoding="utf-8"))
        if v6_path.exists()
        else None
    )
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
    rows = augment_system_context_features(complete_rows)
    splits = annual_walk_forward_splits(rows)
    if not splits:
        raise ValueError("no walk-forward splits available for V7 replay")

    profitable_return_threshold = round_trip_cost_bps / 10_000.0
    rank_threshold = 1.0 - validation_top_fraction
    fixed_backtester = FixedAllocationBacktester(
        BacktestConfig(
            starting_capital=starting_capital,
            allocation_pct=base_allocation_pct,
            max_open_positions=max_open_positions,
            min_expected_alpha=-1_000_000.0,
            min_probability_positive=0.0,
            max_expected_downside=max_expected_downside,
            round_trip_cost_bps=round_trip_cost_bps,
        )
    )
    overlay_config = RiskOverlayConfig(
        starting_capital=starting_capital,
        base_allocation_pct=base_allocation_pct,
        min_allocation_pct=min_allocation_pct,
        max_gross_exposure_pct=max_gross_exposure_pct,
        max_open_positions=max_open_positions,
        max_expected_downside=max_expected_downside,
        correlation_lookback_sessions=correlation_lookback_sessions,
        min_correlation_observations=min_correlation_observations,
        correlation_penalty_start=correlation_penalty_start,
        high_correlation_threshold=high_correlation_threshold,
        max_correlated_exposure_pct=max_correlated_exposure_pct,
        round_trip_cost_bps=round_trip_cost_bps,
    )
    overlay_backtester = RiskOverlayBacktester(market_store, overlay_config)

    v5_year_by_year = {int(item["year"]): item for item in v5_payload["years"]}
    observed_results: list[WalkForwardResult] = []
    year_reports: list[dict] = []
    selected_horizons_global: Counter[int] = Counter()
    trade_horizons_global: Counter[int] = Counter()
    diagnostic_rows: list[tuple[int, dict]] = []

    for split in splits:
        validation_signals: dict[int, np.ndarray] = {}
        test_signals: dict[int, np.ndarray] = {}
        validation_eligible: dict[int, np.ndarray] = {}
        test_eligible: dict[int, np.ndarray] = {}
        validation_expected_returns: dict[int, np.ndarray] = {}
        test_predictions: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

        for horizon in _HORIZONS:
            validation_rows_h = tuple(
                row_for_horizon(row, targets[row.event_id][horizon])
                for row in split.validation_rows
            )
            test_rows_h = tuple(
                row_for_horizon(row, targets[row.event_id][horizon])
                for row in split.test_rows
            )
            model_dir = models_root / str(split.test_year) / f"{horizon}d"
            profit_model = ProfitLightGbmModelBundle.load(model_dir / "profit")
            alpha_model = LightGbmModelBundle.load(model_dir / "alpha")

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

        _, validation_choice_signal, validation_any_eligible = _choose_horizons(
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
        for index in selected_indices:
            row = split.test_rows[index]
            horizon = int(test_choice[index])
            expected_return, downside, probability, _ = test_predictions[horizon]
            projected_row = row_for_horizon(row, targets[row.event_id][horizon])
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

        fixed_replay = fixed_backtester.run(tuple(selected))
        expected_v5 = v5_year_by_year.get(split.test_year)
        if expected_v5 is None:
            raise RuntimeError(f"V5 report missing year {split.test_year}")
        if len(fixed_replay.trades) != int(expected_v5["trades"]):
            raise RuntimeError(
                f"V7 selection replay diverged from V5 trade count in {split.test_year}"
            )
        if not np.isclose(
            fixed_replay.total_return,
            float(expected_v5["return"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"V7 selection replay diverged from V5 return in {split.test_year}"
            )

        portfolio, diagnostics = overlay_backtester.run(tuple(selected))
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
        diagnostic_dict = asdict(diagnostics)
        diagnostic_rows.append((len(portfolio.trades), diagnostic_dict))
        year_reports.append(
            {
                "year": split.test_year,
                "rank_eligible": len(selected),
                "v5_fixed_trades": len(fixed_replay.trades),
                "v5_fixed_return": fixed_replay.total_return,
                "trades": len(portfolio.trades),
                "return": portfolio.total_return,
                "profit_factor": portfolio.profit_factor,
                "realized_drawdown": portfolio.realized_max_drawdown,
                "selected_horizon_counts": _counter_json(
                    Counter(int(test_choice[index]) for index in selected_indices)
                ),
                "trade_horizon_counts": _counter_json(trade_horizons),
                "risk_overlay": diagnostic_dict,
                "trade_average_stock_return": _average(
                    [trade.gross_return for trade in portfolio.trades]
                ),
                "trade_average_alpha": _average(
                    [trade.alpha_20d for trade in portfolio.trades]
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
        "average_allocation_pct": _weighted_diagnostic_average(
            diagnostic_rows,
            "average_allocation_pct",
        ),
        "average_risk_multiplier": _weighted_diagnostic_average(
            diagnostic_rows,
            "average_risk_multiplier",
        ),
        "average_entry_max_correlation": _weighted_diagnostic_average(
            diagnostic_rows,
            "average_entry_max_correlation",
        ),
        "max_gross_exposure_pct": max(
            (float(item["risk_overlay"]["max_gross_exposure_pct"]) for item in year_reports),
            default=0.0,
        ),
        "downsized_by_downside": sum(
            int(item["risk_overlay"]["downsized_by_downside"]) for item in year_reports
        ),
        "downsized_by_regime": sum(
            int(item["risk_overlay"]["downsized_by_regime"]) for item in year_reports
        ),
        "downsized_by_volatility": sum(
            int(item["risk_overlay"]["downsized_by_volatility"]) for item in year_reports
        ),
        "downsized_by_correlation": sum(
            int(item["risk_overlay"]["downsized_by_correlation"]) for item in year_reports
        ),
        "rejected_correlation_exposure": sum(
            int(item["risk_overlay"]["rejected_correlation_exposure"]) for item in year_reports
        ),
        "rejected_gross_exposure": sum(
            int(item["risk_overlay"]["rejected_gross_exposure"]) for item in year_reports
        ),
    }

    ranked_years = sorted(year_reports, key=lambda item: float(item["return"]), reverse=True)
    best_year = ranked_years[0]
    best_three_years = {int(item["year"]) for item in ranked_years[:3]}
    concentration = {
        "best_year": best_year["year"],
        "best_year_return": best_year["return"],
        "best_year_trade_fraction": (
            best_year["trades"] / observed_summary.total_trades
            if observed_summary.total_trades
            else None
        ),
        "compounded_return_excluding_best_year": _compound_excluding(
            year_reports,
            {int(best_year["year"])},
        ),
        "best_three_years": sorted(best_three_years),
        "compounded_return_excluding_best_three_years": _compound_excluding(
            year_reports,
            best_three_years,
        ),
    }

    payload = _json_safe(
        {
            "schema_version": "profit-target-lightgbm-v7-risk-overlay",
            "experiment_dir": str(root),
            "market_db": str(market_db),
            "benchmark_security_id": benchmark_security_id,
            "source_training_row_count": len(source_rows),
            "complete_multi_horizon_row_count": len(rows),
            "model_years": [split.test_year for split in splits],
            "development_mode": {
                "kind": "portfolio_risk_overlay",
                "predictor_retrained": False,
                "selection_identity_verified_against_v5": True,
                "reason": "retain V5 fixed-allocation ceiling and permit only PIT risk reductions",
            },
            "v5_baseline": v5_payload["observed"],
            "v6_baseline": (v6_payload["observed"] if v6_payload else None),
            "selection_policy": {
                "source": "saved V5 5d/20d/60d models",
                "validation_top_fraction": validation_top_fraction,
                "alpha_rank_weight": alpha_rank_weight,
                "calibration_window_days": calibration_window_days,
                "max_expected_downside": max_expected_downside,
                "round_trip_cost_bps": round_trip_cost_bps,
            },
            "portfolio_policy": asdict(overlay_config),
            "observed": observed,
            "concentration": concentration,
            "market_cache_stats": market_store.read_cache_stats(),
            "years": year_reports,
        }
    )
    output_path = root / "profit_target_v7_backtest.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return ProfitTargetV7ExperimentResult(
        source_training_row_count=len(source_rows),
        complete_multi_horizon_row_count=len(rows),
        model_years=tuple(split.test_year for split in splits),
        output_path=output_path,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay saved V5 adaptive-horizon selection exactly, then apply a V7 "
            "one-way downside/regime/volatility/correlation risk overlay capped at "
            "the proven fixed allocation."
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
    parser.add_argument("--base-allocation-pct", type=float, default=0.02)
    parser.add_argument("--min-allocation-pct", type=float, default=0.01)
    parser.add_argument("--max-gross-exposure-pct", type=float, default=0.20)
    parser.add_argument("--max-open-positions", type=int, default=15)
    parser.add_argument("--correlation-lookback-sessions", type=int, default=60)
    parser.add_argument("--min-correlation-observations", type=int, default=30)
    parser.add_argument("--correlation-penalty-start", type=float, default=0.50)
    parser.add_argument("--high-correlation-threshold", type=float, default=0.75)
    parser.add_argument("--max-correlated-exposure-pct", type=float, default=0.08)
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    parser.add_argument("--market-read-cache-series", type=int, default=160)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_profit_target_v7_experiment(
        args.experiment_dir,
        market_db=args.market_db,
        benchmark_security_id=args.benchmark_security_id,
        validation_top_fraction=args.validation_top_fraction,
        alpha_rank_weight=args.alpha_rank_weight,
        calibration_window_days=args.calibration_window_days,
        max_expected_downside=args.max_expected_downside,
        starting_capital=args.starting_capital,
        base_allocation_pct=args.base_allocation_pct,
        min_allocation_pct=args.min_allocation_pct,
        max_gross_exposure_pct=args.max_gross_exposure_pct,
        max_open_positions=args.max_open_positions,
        correlation_lookback_sessions=args.correlation_lookback_sessions,
        min_correlation_observations=args.min_correlation_observations,
        correlation_penalty_start=args.correlation_penalty_start,
        high_correlation_threshold=args.high_correlation_threshold,
        max_correlated_exposure_pct=args.max_correlated_exposure_pct,
        round_trip_cost_bps=args.round_trip_cost_bps,
        market_read_cache_series=args.market_read_cache_series,
    )
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
