import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from stock_trading.backtest import BacktestConfig, FixedAllocationBacktester, summarize_walk_forward
from stock_trading.backtest.portfolio import ScoredCandidate
from stock_trading.ml import OpportunityPrediction, ProfitLightGbmModelBundle, TrainingRow
from stock_trading.ml.walk_forward import WalkForwardResult, annual_walk_forward_splits

from .lightgbm_diagnostics import _load_training_rows
from .lightgbm_profit import (
    _average,
    _distribution_summary,
    _predict_profit_matrix,
    _scored_candidates,
)
from .lightgbm_validation_rank import _json_safe


@dataclass(frozen=True, slots=True)
class ProfitRollingRankExperimentResult:
    training_row_count: int
    model_years: tuple[int, ...]
    permutations: int
    output_path: Path


@dataclass(frozen=True, slots=True)
class _YearInputs:
    year: int
    train_count: int
    validation_count: int
    validation_rows: tuple[TrainingRow, ...]
    validation_scores: np.ndarray
    test_rows: tuple[TrainingRow, ...]
    predictions: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def run_profit_rolling_rank_experiment(
    experiment_dir: str | Path,
    *,
    permutations: int = 250,
    seed: int = 42,
    target_fraction: float = 0.05,
    calibration_days: int = 365,
    max_expected_downside: float = 0.06,
    starting_capital: float = 10_000.0,
    allocation_pct: float = 0.02,
    max_open_positions: int = 15,
    round_trip_cost_bps: float = 20.0,
) -> ProfitRollingRankExperimentResult:
    """Replay saved profit models with a live-safe rolling score percentile gate.

    The static validation threshold can become stale when the model's score scale
    shifts between years. This experiment keeps the already-trained annual profit
    model but calibrates its entry threshold from only scores that were observable
    before the current execution session. The rolling history is initialized with
    the preceding validation year and then updated after each test session.

    No test outcome enters the threshold. Only prediction scores and timestamps are
    used for rolling calibration, so the policy is reproducible in live trading.
    """

    if permutations < 0:
        raise ValueError("permutations must be >= 0")
    if not 0 < target_fraction < 1:
        raise ValueError("target_fraction must be in (0, 1)")
    if calibration_days <= 0:
        raise ValueError("calibration_days must be > 0")
    if max_expected_downside < 0:
        raise ValueError("max_expected_downside must be >= 0")
    if round_trip_cost_bps < 0:
        raise ValueError("round_trip_cost_bps must be >= 0")

    root = Path(experiment_dir)
    rows_path = root / "training_rows.jsonl"
    models_root = root / "profit_models"
    if not rows_path.exists():
        raise FileNotFoundError(f"missing training rows: {rows_path}")
    if not models_root.exists():
        raise FileNotFoundError(
            f"missing profit models: {models_root}; run lightgbm_profit first"
        )

    rows = _load_training_rows(rows_path)
    model_years = tuple(
        sorted(
            int(path.name)
            for path in models_root.iterdir()
            if path.is_dir() and path.name.isdigit() and (path / "metadata.json").exists()
        )
    )
    if not model_years:
        raise ValueError("no saved annual profit LightGBM models found")

    split_by_year = {
        split.test_year: split
        for split in annual_walk_forward_splits(rows, first_test_year=min(model_years))
    }
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

    observed_results: list[WalkForwardResult] = []
    year_inputs: list[_YearInputs] = []
    year_reports: list[dict] = []

    for year in model_years:
        split = split_by_year.get(year)
        if split is None:
            raise ValueError(f"could not reconstruct walk-forward split for model year {year}")

        model = ProfitLightGbmModelBundle.load(models_root / str(year))
        validation_predictions = _predict_profit_matrix(model, split.validation_rows)
        test_predictions = _predict_profit_matrix(model, split.test_rows)
        selected_indices, thresholds = _rolling_selected_indices(
            split.validation_rows,
            validation_predictions[3],
            split.test_rows,
            test_predictions[3],
            target_fraction=target_fraction,
            calibration_days=calibration_days,
        )
        scored = _scored_candidates(split.test_rows, test_predictions)
        selected = tuple(scored[index] for index in selected_indices)
        portfolio = backtester.run(selected)
        observed_results.append(
            WalkForwardResult(
                test_year=year,
                train_count=len(split.train_rows),
                validation_count=len(split.validation_rows),
                test_count=len(split.test_rows),
                backtest=portfolio,
            )
        )
        year_inputs.append(
            _YearInputs(
                year=year,
                train_count=len(split.train_rows),
                validation_count=len(split.validation_rows),
                validation_rows=split.validation_rows,
                validation_scores=validation_predictions[3],
                test_rows=split.test_rows,
                predictions=test_predictions,
            )
        )

        selected_rows = [candidate.row for candidate in selected]
        trades = portfolio.trades
        threshold_values = list(thresholds.values())
        year_reports.append(
            {
                "year": year,
                "test_row_count": len(split.test_rows),
                "test_selected": len(selected),
                "selected_fraction": len(selected) / len(split.test_rows),
                "trades": len(trades),
                "return": portfolio.total_return,
                "profit_factor": portfolio.profit_factor,
                "realized_drawdown": portfolio.realized_max_drawdown,
                "rejected_duplicate_company": portfolio.rejected_duplicate_company,
                "rejected_capacity": portfolio.rejected_capacity,
                "rolling_threshold_first": threshold_values[0] if threshold_values else None,
                "rolling_threshold_last": threshold_values[-1] if threshold_values else None,
                "rolling_threshold_median": (
                    float(np.median(np.asarray(threshold_values, dtype=np.float64)))
                    if threshold_values
                    else None
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
    all_observed_trades = [
        trade for result in observed_results for trade in result.backtest.trades
    ]
    observed = {
        "compounded_return": observed_summary.compounded_return,
        "profitable_year_rate": observed_summary.profitable_year_rate,
        "total_trades": observed_summary.total_trades,
        "average_trade_stock_return": _average(
            [trade.gross_return for trade in all_observed_trades]
        ),
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
        target_fraction=target_fraction,
        calibration_days=calibration_days,
    )
    payload = _json_safe(
        {
            "schema_version": "profit-rolling-rank-lightgbm-v1",
            "experiment_dir": str(root),
            "training_row_count": len(rows),
            "model_years": list(model_years),
            "selection_policy": {
                "target": "absolute_stock_return_after_costs",
                "score_gate": "rolling_prior-score-percentile",
                "target_fraction": target_fraction,
                "calibration_days": calibration_days,
                "same_session_scores_added_after_decision": True,
                "test_outcomes_used_for_calibration": False,
                "max_expected_downside": max_expected_downside,
                "starting_capital": starting_capital,
                "allocation_pct": allocation_pct,
                "max_open_positions": max_open_positions,
                "round_trip_cost_bps": round_trip_cost_bps,
            },
            "observed": observed,
            "null": null,
            "years": year_reports,
        }
    )
    output_path = root / "profit_rolling_rank_backtest.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return ProfitRollingRankExperimentResult(
        training_row_count=len(rows),
        model_years=model_years,
        permutations=permutations,
        output_path=output_path,
    )


def _rolling_selected_indices(
    validation_rows: tuple[TrainingRow, ...],
    validation_scores: np.ndarray,
    test_rows: tuple[TrainingRow, ...],
    test_scores: np.ndarray,
    *,
    target_fraction: float,
    calibration_days: int,
) -> tuple[tuple[int, ...], dict[date, float]]:
    """Select test rows from a trailing score distribution without outcome lookahead."""

    if len(validation_rows) != len(validation_scores):
        raise ValueError("validation row/score length mismatch")
    if len(test_rows) != len(test_scores):
        raise ValueError("test row/score length mismatch")
    if not validation_rows:
        raise ValueError("rolling calibration requires validation rows")
    if not 0 < target_fraction < 1:
        raise ValueError("target_fraction must be in (0, 1)")
    if calibration_days <= 0:
        raise ValueError("calibration_days must be > 0")

    history = deque(
        sorted(
            (
                (row.execution_date, float(validation_scores[index]))
                for index, row in enumerate(validation_rows)
            ),
            key=lambda item: item[0],
        )
    )
    by_session: dict[date, list[int]] = {}
    for index, row in enumerate(test_rows):
        by_session.setdefault(row.execution_date, []).append(index)

    selected: list[int] = []
    thresholds: dict[date, float] = {}
    for session_date in sorted(by_session):
        cutoff = session_date - timedelta(days=calibration_days)
        while history and history[0][0] < cutoff:
            history.popleft()
        if not history:
            raise ValueError(
                f"rolling score calibration has no prior scores for {session_date.isoformat()}"
            )

        history_scores = np.fromiter(
            (score for _, score in history),
            dtype=np.float64,
            count=len(history),
        )
        threshold = float(np.quantile(history_scores, 1.0 - target_fraction))
        thresholds[session_date] = threshold

        # Every opportunity resolving to this execution session uses the same
        # threshold derived strictly from earlier sessions. Only after the live
        # decision do today's scores enter the rolling calibration distribution.
        session_indices = by_session[session_date]
        selected.extend(
            index for index in session_indices if float(test_scores[index]) >= threshold
        )
        history.extend(
            (session_date, float(test_scores[index])) for index in session_indices
        )

    return tuple(selected), thresholds


def _permutation_null(
    year_inputs: list[_YearInputs],
    backtester: FixedAllocationBacktester,
    observed: dict,
    *,
    permutations: int,
    seed: int,
    target_fraction: float,
    calibration_days: int,
) -> dict:
    if permutations == 0:
        return {"permutations": 0}

    rng = np.random.default_rng(seed)
    compounded_returns: list[float] = []
    average_stock_returns: list[float] = []
    average_alphas: list[float] = []
    profitable_year_rates: list[float] = []
    trade_counts: list[float] = []
    drawdowns: list[float] = []

    for _ in range(permutations):
        results: list[WalkForwardResult] = []
        for item in year_inputs:
            expected_return, downside, probability, score = item.predictions
            order = rng.permutation(len(item.test_rows))
            permuted_expected_return = expected_return[order]
            permuted_downside = downside[order]
            permuted_probability = probability[order]
            permuted_score = score[order]
            selected_indices, _ = _rolling_selected_indices(
                item.validation_rows,
                item.validation_scores,
                item.test_rows,
                permuted_score,
                target_fraction=target_fraction,
                calibration_days=calibration_days,
            )
            selected = tuple(
                ScoredCandidate(
                    row=item.test_rows[index],
                    prediction=OpportunityPrediction(
                        expected_alpha_20d=float(permuted_expected_return[index]),
                        expected_downside_20d=float(permuted_downside[index]),
                        probability_positive_alpha=float(permuted_probability[index]),
                        opportunity_score=float(permuted_score[index]),
                    ),
                )
                for index in selected_indices
            )
            portfolio = backtester.run(selected)
            results.append(
                WalkForwardResult(
                    test_year=item.year,
                    train_count=item.train_count,
                    validation_count=item.validation_count,
                    test_count=len(item.test_rows),
                    backtest=portfolio,
                )
            )

        summary = summarize_walk_forward(results)
        trades = [trade for result in results for trade in result.backtest.trades]
        compounded_returns.append(summary.compounded_return)
        profitable_year_rates.append(summary.profitable_year_rate)
        trade_counts.append(float(summary.total_trades))
        drawdowns.append(summary.worst_realized_drawdown)
        stock_return = _average([trade.gross_return for trade in trades])
        if stock_return is not None:
            average_stock_returns.append(stock_return)
        if summary.average_trade_alpha is not None:
            average_alphas.append(summary.average_trade_alpha)

    return {
        "permutations": permutations,
        "compounded_return": _distribution_summary(
            compounded_returns,
            observed["compounded_return"],
            higher_is_better=True,
        ),
        "average_trade_stock_return": _distribution_summary(
            average_stock_returns,
            observed["average_trade_stock_return"],
            higher_is_better=True,
        ),
        "average_trade_alpha": _distribution_summary(
            average_alphas,
            observed["average_trade_alpha"],
            higher_is_better=True,
        ),
        "profitable_year_rate": _distribution_summary(
            profitable_year_rates,
            observed["profitable_year_rate"],
            higher_is_better=True,
        ),
        "total_trades": _distribution_summary(
            trade_counts,
            float(observed["total_trades"]),
            higher_is_better=None,
        ),
        "worst_realized_drawdown": _distribution_summary(
            drawdowns,
            observed["worst_realized_drawdown"],
            higher_is_better=False,
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay saved profit-targeted LightGBM models with a live-safe rolling "
            "score-percentile entry gate and a within-year permutation null."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-fraction", type=float, default=0.05)
    parser.add_argument("--calibration-days", type=int, default=365)
    parser.add_argument("--max-expected-downside", type=float, default=0.06)
    parser.add_argument("--starting-capital", type=float, default=10_000.0)
    parser.add_argument("--allocation-pct", type=float, default=0.02)
    parser.add_argument("--max-open-positions", type=int, default=15)
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_profit_rolling_rank_experiment(
        args.experiment_dir,
        permutations=args.permutations,
        seed=args.seed,
        target_fraction=args.target_fraction,
        calibration_days=args.calibration_days,
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
                "selection_policy": payload["selection_policy"],
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
