import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from stock_trading.backtest import (
    BacktestConfig,
    FixedAllocationBacktester,
    summarize_walk_forward,
)
from stock_trading.backtest.portfolio import ScoredCandidate
from stock_trading.ml import LightGbmModelBundle, OpportunityPrediction
from stock_trading.ml.walk_forward import WalkForwardResult, annual_walk_forward_splits

from .lightgbm_diagnostics import _load_training_rows, _predict_matrix
from .lightgbm_validation_rank import _json_safe, _scored_candidates


@dataclass(frozen=True, slots=True)
class PermutationNullResult:
    training_row_count: int
    model_years: tuple[int, ...]
    permutations: int
    output_path: Path


@dataclass(frozen=True, slots=True)
class _YearInputs:
    year: int
    train_count: int
    validation_count: int
    test_rows: tuple
    score_threshold: float
    predictions: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def run_permutation_null_test(
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
) -> PermutationNullResult:
    """Compare the frozen validation-ranked strategy with a prediction-permutation null.

    For every annual test set, the fitted model prediction bundle
    (alpha/downside/probability/score) is randomly reassigned to test rows while
    preserving the year's prediction distribution and the validation-derived
    score threshold. This destroys any relationship between model predictions
    and realized outcomes without changing the number of scores above the
    threshold. The same downside gate, capacity rules and transaction costs are
    then applied to each permuted strategy.

    Validation and test outcomes are never used to choose a threshold here; the
    experiment evaluates the already-declared top-fraction policy against a
    deterministic randomization null.
    """

    if permutations <= 0:
        raise ValueError("permutations must be > 0")
    if not 0 < validation_top_fraction < 1:
        raise ValueError("validation_top_fraction must be in (0, 1)")
    if max_expected_downside < 0:
        raise ValueError("max_expected_downside must be >= 0")

    root = Path(experiment_dir)
    rows_path = root / "training_rows.jsonl"
    models_root = root / "models"
    if not rows_path.exists():
        raise FileNotFoundError(f"missing training rows: {rows_path}")
    if not models_root.exists():
        raise FileNotFoundError(f"missing model directory: {models_root}")

    rows = _load_training_rows(rows_path)
    model_years = tuple(
        sorted(
            int(path.name)
            for path in models_root.iterdir()
            if path.is_dir() and path.name.isdigit() and (path / "metadata.json").exists()
        )
    )
    if not model_years:
        raise ValueError("no saved annual LightGBM models found")

    split_by_year = {
        split.test_year: split
        for split in annual_walk_forward_splits(rows, first_test_year=min(model_years))
    }
    backtest_config = BacktestConfig(
        starting_capital=starting_capital,
        allocation_pct=allocation_pct,
        max_open_positions=max_open_positions,
        min_expected_alpha=-1_000_000.0,
        min_probability_positive=0.0,
        max_expected_downside=max_expected_downside,
        round_trip_cost_bps=round_trip_cost_bps,
    )
    backtester = FixedAllocationBacktester(backtest_config)

    year_inputs: list[_YearInputs] = []
    observed_results: list[WalkForwardResult] = []
    observed_year_reports: list[dict] = []
    for year in model_years:
        split = split_by_year.get(year)
        if split is None:
            raise ValueError(f"could not reconstruct walk-forward split for model year {year}")
        model = LightGbmModelBundle.load(models_root / str(year))
        validation_predictions = _predict_matrix(model, split.validation_rows)
        predictions = _predict_matrix(model, split.test_rows)
        score_threshold = float(
            np.quantile(validation_predictions[3], 1.0 - validation_top_fraction)
        )

        observed_scored = _scored_candidates(split.test_rows, predictions)
        observed_selected = tuple(
            candidate
            for candidate in observed_scored
            if candidate.prediction.opportunity_score >= score_threshold
        )
        observed_portfolio = backtester.run(observed_selected)
        observed_results.append(
            WalkForwardResult(
                test_year=year,
                train_count=len(split.train_rows),
                validation_count=len(split.validation_rows),
                test_count=len(split.test_rows),
                backtest=observed_portfolio,
            )
        )
        observed_year_reports.append(
            {
                "year": year,
                "score_threshold_from_validation": score_threshold,
                "selected_count": len(observed_selected),
                "trade_count": len(observed_portfolio.trades),
                "return": observed_portfolio.total_return,
                "average_trade_alpha": _average_trade_alpha(observed_portfolio.trades),
            }
        )
        year_inputs.append(
            _YearInputs(
                year=year,
                train_count=len(split.train_rows),
                validation_count=len(split.validation_rows),
                test_rows=split.test_rows,
                score_threshold=score_threshold,
                predictions=predictions,
            )
        )

    observed_summary = summarize_walk_forward(observed_results)
    rng = np.random.default_rng(seed)
    null_compounded_returns: list[float] = []
    null_average_trade_alphas: list[float] = []
    null_profitable_year_rates: list[float] = []
    null_trade_counts: list[float] = []
    null_worst_drawdowns: list[float] = []
    per_year_null_returns = {item.year: [] for item in year_inputs}
    per_year_null_alphas = {item.year: [] for item in year_inputs}

    for _ in range(permutations):
        permutation_results: list[WalkForwardResult] = []
        for item in year_inputs:
            alpha, downside, probability, score = item.predictions
            order = rng.permutation(len(item.test_rows))
            permuted_score = score[order]
            selected_row_indices = np.flatnonzero(
                permuted_score >= item.score_threshold
            )
            selected = tuple(
                ScoredCandidate(
                    row=item.test_rows[int(row_index)],
                    prediction=OpportunityPrediction(
                        expected_alpha_20d=float(alpha[int(order[row_index])]),
                        expected_downside_20d=float(downside[int(order[row_index])]),
                        probability_positive_alpha=float(probability[int(order[row_index])]),
                        opportunity_score=float(score[int(order[row_index])]),
                    ),
                )
                for row_index in selected_row_indices
            )
            portfolio = backtester.run(selected)
            permutation_results.append(
                WalkForwardResult(
                    test_year=item.year,
                    train_count=item.train_count,
                    validation_count=item.validation_count,
                    test_count=len(item.test_rows),
                    backtest=portfolio,
                )
            )
            per_year_null_returns[item.year].append(portfolio.total_return)
            alpha_value = _average_trade_alpha(portfolio.trades)
            if alpha_value is not None:
                per_year_null_alphas[item.year].append(alpha_value)

        summary = summarize_walk_forward(permutation_results)
        null_compounded_returns.append(summary.compounded_return)
        if summary.average_trade_alpha is not None:
            null_average_trade_alphas.append(summary.average_trade_alpha)
        null_profitable_year_rates.append(summary.profitable_year_rate)
        null_trade_counts.append(float(summary.total_trades))
        null_worst_drawdowns.append(summary.worst_realized_drawdown)

    observed_metrics = {
        "compounded_return": observed_summary.compounded_return,
        "average_trade_alpha": observed_summary.average_trade_alpha,
        "profitable_year_rate": observed_summary.profitable_year_rate,
        "total_trades": observed_summary.total_trades,
        "worst_realized_drawdown": observed_summary.worst_realized_drawdown,
    }
    null_summary = {
        "compounded_return": _distribution_summary(
            null_compounded_returns,
            observed_summary.compounded_return,
            higher_is_better=True,
        ),
        "average_trade_alpha": _distribution_summary(
            null_average_trade_alphas,
            observed_summary.average_trade_alpha,
            higher_is_better=True,
        ),
        "profitable_year_rate": _distribution_summary(
            null_profitable_year_rates,
            observed_summary.profitable_year_rate,
            higher_is_better=True,
        ),
        "total_trades": _distribution_summary(
            null_trade_counts,
            float(observed_summary.total_trades),
            higher_is_better=None,
        ),
        "worst_realized_drawdown": _distribution_summary(
            null_worst_drawdowns,
            observed_summary.worst_realized_drawdown,
            higher_is_better=False,
        ),
    }

    year_reports: list[dict] = []
    observed_by_year = {item["year"]: item for item in observed_year_reports}
    for item in year_inputs:
        observed = observed_by_year[item.year]
        year_reports.append(
            {
                **observed,
                "null_return": _distribution_summary(
                    per_year_null_returns[item.year],
                    observed["return"],
                    higher_is_better=True,
                ),
                "null_average_trade_alpha": _distribution_summary(
                    per_year_null_alphas[item.year],
                    observed["average_trade_alpha"],
                    higher_is_better=True,
                ),
            }
        )

    payload = _json_safe(
        {
            "schema_version": "lightgbm-permutation-null-v1",
            "experiment_dir": str(root),
            "training_row_count": len(rows),
            "model_years": list(model_years),
            "null_design": {
                "permutations": permutations,
                "seed": seed,
                "prediction_bundle_permuted_within_test_year": True,
                "validation_top_fraction": validation_top_fraction,
                "validation_threshold_frozen": True,
                "max_expected_downside": max_expected_downside,
                "starting_capital": starting_capital,
                "allocation_pct": allocation_pct,
                "max_open_positions": max_open_positions,
                "round_trip_cost_bps": round_trip_cost_bps,
            },
            "observed": observed_metrics,
            "null": null_summary,
            "years": year_reports,
        }
    )
    output_path = root / "permutation_null.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return PermutationNullResult(
        training_row_count=len(rows),
        model_years=model_years,
        permutations=permutations,
        output_path=output_path,
    )


def _average_trade_alpha(trades) -> float | None:
    if not trades:
        return None
    return sum(trade.alpha_20d for trade in trades) / len(trades)


def _distribution_summary(
    values,
    observed: float | None,
    *,
    higher_is_better: bool | None,
) -> dict[str, float | None]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "p05": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "observed_percentile": None,
            "one_sided_p_value": None,
        }
    result: dict[str, float | None] = {
        "count": int(array.size),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "observed_percentile": None,
        "one_sided_p_value": None,
    }
    if observed is None:
        return result
    observed_value = float(observed)
    result["observed_percentile"] = float((array <= observed_value).mean())
    if higher_is_better is True:
        extreme = int((array >= observed_value).sum())
        result["one_sided_p_value"] = (extreme + 1.0) / (array.size + 1.0)
    elif higher_is_better is False:
        extreme = int((array <= observed_value).sum())
        result["one_sided_p_value"] = (extreme + 1.0) / (array.size + 1.0)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the saved validation-ranked LightGBM strategy against a "
            "within-year prediction-permutation null without rebuilding or retraining."
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
    result = run_permutation_null_test(
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
                "observed": payload["observed"],
                "null": payload["null"],
                "output_path": str(result.output_path),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
