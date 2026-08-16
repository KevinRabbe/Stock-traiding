from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stock_trading.backtest import BacktestConfig, FixedAllocationBacktester, summarize_walk_forward
from stock_trading.ml import LightGbmTrainingConfig, ProfitLightGbmTrainer
from stock_trading.ml.opportunity_history import (
    OPPORTUNITY_HISTORY_FEATURES,
    augment_opportunity_history_features,
)
from stock_trading.ml.walk_forward import WalkForwardResult, annual_walk_forward_splits

from .lightgbm_diagnostics import _load_training_rows
from .lightgbm_profit import (
    _YearInputs,
    _average,
    _permutation_null,
    _predict_profit_matrix,
    _scored_candidates,
)
from .lightgbm_validation_rank import _json_safe


@dataclass(frozen=True, slots=True)
class ProfitTargetV2ExperimentResult:
    training_row_count: int
    model_years: tuple[int, ...]
    permutations: int
    output_path: Path


def run_profit_target_v2_experiment(
    experiment_dir: str | Path,
    *,
    permutations: int = 250,
    seed: int = 42,
    validation_top_fraction: float = 0.05,
    max_expected_downside: float = 0.06,
    starting_capital: float = 10_000.0,
    allocation_pct: float = 0.02,
    max_open_positions: int = 15,
    round_trip_cost_bps: float = 20.0,
    training_config: LightGbmTrainingConfig | None = None,
) -> ProfitTargetV2ExperimentResult:
    """Train profit-targeted V2 models with PIT same-company history features.

    V2 changes model inputs only. Portfolio mechanics remain the single-active-
    position-per-company fixed-allocation backtester used by the original profit
    experiment. Opportunity history is derived from strictly earlier decision rows
    before walk-forward splitting, so validation/test rows can see prior history
    but never future opportunities or realized labels.
    """

    if permutations < 0:
        raise ValueError("permutations must be >= 0")
    if not 0 < validation_top_fraction < 1:
        raise ValueError("validation_top_fraction must be in (0, 1)")
    if max_expected_downside < 0:
        raise ValueError("max_expected_downside must be >= 0")
    if round_trip_cost_bps < 0:
        raise ValueError("round_trip_cost_bps must be >= 0")

    root = Path(experiment_dir)
    rows_path = root / "training_rows.jsonl"
    if not rows_path.exists():
        raise FileNotFoundError(f"missing training rows: {rows_path}")

    base_rows = _load_training_rows(rows_path)
    rows = augment_opportunity_history_features(base_rows)
    splits = annual_walk_forward_splits(rows)
    if not splits:
        raise ValueError("no walk-forward splits available")

    profitable_return_threshold = round_trip_cost_bps / 10_000.0
    trainer = ProfitLightGbmTrainer(training_config)
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

    models_root = root / "profit_models_v2"
    observed_results: list[WalkForwardResult] = []
    year_inputs: list[_YearInputs] = []
    year_reports: list[dict] = []

    for split in splits:
        model = trainer.train(
            split.train_rows,
            split.validation_rows,
            profitable_return_threshold=profitable_return_threshold,
        )
        model.save(models_root / str(split.test_year))

        validation_predictions = _predict_profit_matrix(model, split.validation_rows)
        test_predictions = _predict_profit_matrix(model, split.test_rows)
        score_threshold = float(
            np.quantile(validation_predictions[3], 1.0 - validation_top_fraction)
        )
        scored = _scored_candidates(split.test_rows, test_predictions)
        selected = tuple(
            candidate
            for candidate in scored
            if candidate.prediction.opportunity_score >= score_threshold
        )
        portfolio = backtester.run(selected)
        observed_results.append(
            WalkForwardResult(
                test_year=split.test_year,
                train_count=len(split.train_rows),
                validation_count=len(split.validation_rows),
                test_count=len(split.test_rows),
                backtest=portfolio,
            )
        )
        year_inputs.append(
            _YearInputs(
                year=split.test_year,
                train_count=len(split.train_rows),
                validation_count=len(split.validation_rows),
                test_rows=split.test_rows,
                score_threshold=score_threshold,
                predictions=test_predictions,
            )
        )

        selected_rows = [candidate.row for candidate in selected]
        trades = portfolio.trades
        year_reports.append(
            {
                "year": split.test_year,
                "validation_score_threshold": score_threshold,
                "test_selected": len(selected),
                "trades": len(trades),
                "return": portfolio.total_return,
                "profit_factor": portfolio.profit_factor,
                "realized_drawdown": portfolio.realized_max_drawdown,
                "rejected_duplicate_company": portfolio.rejected_duplicate_company,
                "rejected_capacity": portfolio.rejected_capacity,
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

    null = _permutation_null(
        year_inputs,
        backtester,
        observed,
        permutations=permutations,
        seed=seed,
    )

    baseline = None
    baseline_path = root / "profit_target_backtest.json"
    if baseline_path.exists():
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8")).get("observed")
        except (OSError, json.JSONDecodeError):
            baseline = None

    payload = _json_safe(
        {
            "schema_version": "profit-target-lightgbm-v2-opportunity-history",
            "experiment_dir": str(root),
            "training_row_count": len(rows),
            "model_years": [item.year for item in year_inputs],
            "feature_augmentation": {
                "kind": "strictly_prior_same_company_opportunity_history",
                "uses_realized_labels": False,
                "features": list(OPPORTUNITY_HISTORY_FEATURES),
            },
            "selection_policy": {
                "target": "absolute_stock_return_after_costs",
                "profitable_return_threshold": profitable_return_threshold,
                "validation_top_fraction": validation_top_fraction,
                "score_cutoff_source": "preceding validation year only",
                "max_expected_downside": max_expected_downside,
                "starting_capital": starting_capital,
                "allocation_pct": allocation_pct,
                "max_open_positions": max_open_positions,
                "max_active_positions_per_company": 1,
                "round_trip_cost_bps": round_trip_cost_bps,
            },
            "v1_baseline": baseline,
            "observed": observed,
            "null": null,
            "years": year_reports,
        }
    )
    output_path = root / "profit_target_v2_backtest.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return ProfitTargetV2ExperimentResult(
        training_row_count=len(rows),
        model_years=tuple(item.year for item in year_inputs),
        permutations=permutations,
        output_path=output_path,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train profit-targeted LightGBM V2 with PIT same-company opportunity-history "
            "features and single-position portfolio mechanics."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-top-fraction", type=float, default=0.05)
    parser.add_argument("--max-expected-downside", type=float, default=0.06)
    parser.add_argument("--starting-capital", type=float, default=10_000.0)
    parser.add_argument("--allocation-pct", type=float, default=0.02)
    parser.add_argument("--max-open-positions", type=int, default=15)
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_profit_target_v2_experiment(
        args.experiment_dir,
        permutations=args.permutations,
        seed=args.seed,
        validation_top_fraction=args.validation_top_fraction,
        max_expected_downside=args.max_expected_downside,
        starting_capital=args.starting_capital,
        allocation_pct=args.allocation_pct,
        max_open_positions=args.max_open_positions,
        round_trip_cost_bps=args.round_trip_cost_bps,
    )
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "training_row_count": result.training_row_count,
                "model_years": result.model_years,
                "permutations": result.permutations,
                "feature_augmentation": payload["feature_augmentation"],
                "selection_policy": payload["selection_policy"],
                "v1_baseline": payload["v1_baseline"],
                "observed": payload["observed"],
                "null": payload["null"],
                "years": payload["years"],
                "output_path": str(result.output_path),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
