import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stock_trading.backtest import (
    BacktestConfig,
    FixedAllocationTrancheBacktester,
    summarize_walk_forward,
)
from stock_trading.backtest.portfolio import ScoredCandidate
from stock_trading.ml import OpportunityPrediction, ProfitLightGbmModelBundle, TrainingRow
from stock_trading.ml.walk_forward import WalkForwardResult, annual_walk_forward_splits

from .lightgbm_diagnostics import _load_training_rows
from .lightgbm_profit import _average, _distribution_summary, _predict_profit_matrix
from .lightgbm_validation_rank import _json_safe


@dataclass(frozen=True, slots=True)
class ProfitTrancheExperimentResult:
    training_row_count: int
    model_years: tuple[int, ...]
    permutations: int
    output_path: Path


@dataclass(frozen=True, slots=True)
class _YearInputs:
    year: int
    train_count: int
    validation_count: int
    test_rows: tuple[TrainingRow, ...]
    score_threshold: float
    predictions: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def run_profit_tranche_experiment(
    experiment_dir: str | Path,
    *,
    permutations: int = 250,
    seed: int = 42,
    validation_top_fraction: float = 0.05,
    max_company_tranches: int = 2,
    max_expected_downside: float = 0.06,
    starting_capital: float = 10_000.0,
    allocation_pct: float = 0.02,
    max_open_positions: int = 15,
    round_trip_cost_bps: float = 20.0,
) -> ProfitTrancheExperimentResult:
    """Replay saved profit models while allowing bounded repeated-company tranches.

    The ranking policy is exactly the original profit-target policy: each annual
    score cutoff comes from the preceding validation year only. The only changed
    portfolio mechanic is that a later high-scoring opportunity may open another
    normal allocation slice in a company that is already held, capped by
    ``max_company_tranches``. Total open slots and allocation per slot stay fixed.
    """

    if permutations < 0:
        raise ValueError("permutations must be >= 0")
    if not 0 < validation_top_fraction < 1:
        raise ValueError("validation_top_fraction must be in (0, 1)")
    if max_company_tranches <= 0:
        raise ValueError("max_company_tranches must be > 0")

    root = Path(experiment_dir)
    rows_path = root / "training_rows.jsonl"
    models_root = root / "profit_models"
    if not rows_path.exists():
        raise FileNotFoundError(f"missing training rows: {rows_path}")
    if not models_root.exists():
        raise FileNotFoundError(
            f"missing profit model directory: {models_root}; run lightgbm_profit first"
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
        raise ValueError("no saved annual profit-targeted LightGBM models found")

    splits = {
        split.test_year: split
        for split in annual_walk_forward_splits(rows, first_test_year=min(model_years))
    }
    config = BacktestConfig(
        starting_capital=starting_capital,
        allocation_pct=allocation_pct,
        max_open_positions=max_open_positions,
        min_expected_alpha=-1_000_000.0,
        min_probability_positive=0.0,
        max_expected_downside=max_expected_downside,
        round_trip_cost_bps=round_trip_cost_bps,
    )
    backtester = FixedAllocationTrancheBacktester(
        config,
        max_company_tranches=max_company_tranches,
    )

    observed_results: list[WalkForwardResult] = []
    year_inputs: list[_YearInputs] = []
    year_reports: list[dict] = []

    for year in model_years:
        split = splits.get(year)
        if split is None:
            raise ValueError(f"could not reconstruct walk-forward split for {year}")
        model = ProfitLightGbmModelBundle.load(models_root / str(year))
        validation_predictions = _predict_profit_matrix(model, split.validation_rows)
        test_predictions = _predict_profit_matrix(model, split.test_rows)
        score_threshold = float(
            np.quantile(validation_predictions[3], 1.0 - validation_top_fraction)
        )
        selected = _selected_candidates(split.test_rows, test_predictions, score_threshold)
        portfolio = backtester.run(selected)
        result = WalkForwardResult(
            test_year=year,
            train_count=len(split.train_rows),
            validation_count=len(split.validation_rows),
            test_count=len(split.test_rows),
            backtest=portfolio,
        )
        observed_results.append(result)
        year_inputs.append(
            _YearInputs(
                year=year,
                train_count=len(split.train_rows),
                validation_count=len(split.validation_rows),
                test_rows=split.test_rows,
                score_threshold=score_threshold,
                predictions=test_predictions,
            )
        )
        year_reports.append(
            {
                "year": year,
                "validation_score_threshold": score_threshold,
                "test_selected": len(selected),
                "trades": len(portfolio.trades),
                "return": portfolio.total_return,
                "profit_factor": portfolio.profit_factor,
                "realized_drawdown": portfolio.realized_max_drawdown,
                "rejected_company_tranche_limit": portfolio.rejected_duplicate_company,
                "rejected_capacity": portfolio.rejected_capacity,
                "trade_average_stock_return_20d": _average(
                    [trade.gross_return for trade in portfolio.trades]
                ),
                "trade_average_alpha_20d": _average(
                    [trade.alpha_20d for trade in portfolio.trades]
                ),
            }
        )

    observed_summary = summarize_walk_forward(observed_results)
    observed_trades = [
        trade for result in observed_results for trade in result.backtest.trades
    ]
    observed = {
        "compounded_return": observed_summary.compounded_return,
        "profitable_year_rate": observed_summary.profitable_year_rate,
        "total_trades": observed_summary.total_trades,
        "average_trade_stock_return": _average(
            [trade.gross_return for trade in observed_trades]
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
    )

    baseline_path = root / "profit_target_backtest.json"
    baseline = None
    if baseline_path.exists():
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8")).get("observed")
        except (OSError, json.JSONDecodeError):
            baseline = None

    payload = _json_safe(
        {
            "schema_version": "profit-tranche-lightgbm-v1",
            "experiment_dir": str(root),
            "training_row_count": len(rows),
            "model_years": list(model_years),
            "selection_policy": {
                "target": "absolute_stock_return_after_costs",
                "score_cutoff_source": "preceding validation year only",
                "validation_top_fraction": validation_top_fraction,
                "max_company_tranches": max_company_tranches,
                "allocation_pct_per_tranche": allocation_pct,
                "max_open_tranches": max_open_positions,
                "max_expected_downside": max_expected_downside,
                "round_trip_cost_bps": round_trip_cost_bps,
                "starting_capital": starting_capital,
            },
            "single_tranche_baseline": baseline,
            "observed": observed,
            "null": null,
            "years": year_reports,
        }
    )
    output_path = root / "profit_tranche_backtest.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return ProfitTrancheExperimentResult(
        training_row_count=len(rows),
        model_years=model_years,
        permutations=permutations,
        output_path=output_path,
    )


def _selected_candidates(
    rows: tuple[TrainingRow, ...],
    predictions: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    threshold: float,
) -> tuple[ScoredCandidate, ...]:
    expected_return, downside, probability, score = predictions
    selected_indices = np.flatnonzero(score >= threshold)
    return tuple(
        ScoredCandidate(
            row=rows[int(index)],
            prediction=OpportunityPrediction(
                expected_alpha_20d=float(expected_return[int(index)]),
                expected_downside_20d=float(downside[int(index)]),
                probability_positive_alpha=float(probability[int(index)]),
                opportunity_score=float(score[int(index)]),
            ),
        )
        for index in selected_indices
    )


def _permutation_null(
    year_inputs: list[_YearInputs],
    backtester: FixedAllocationTrancheBacktester,
    observed: dict,
    *,
    permutations: int,
    seed: int,
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
            permuted_score = score[order]
            selected_row_indices = np.flatnonzero(permuted_score >= item.score_threshold)
            selected = tuple(
                ScoredCandidate(
                    row=item.test_rows[int(row_index)],
                    prediction=OpportunityPrediction(
                        expected_alpha_20d=float(expected_return[int(order[row_index])]),
                        expected_downside_20d=float(downside[int(order[row_index])]),
                        probability_positive_alpha=float(probability[int(order[row_index])]),
                        opportunity_score=float(score[int(order[row_index])]),
                    ),
                )
                for row_index in selected_row_indices
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
            compounded_returns, observed["compounded_return"], higher_is_better=True
        ),
        "average_trade_stock_return": _distribution_summary(
            average_stock_returns,
            observed["average_trade_stock_return"],
            higher_is_better=True,
        ),
        "average_trade_alpha": _distribution_summary(
            average_alphas, observed["average_trade_alpha"], higher_is_better=True
        ),
        "profitable_year_rate": _distribution_summary(
            profitable_year_rates,
            observed["profitable_year_rate"],
            higher_is_better=True,
        ),
        "total_trades": _distribution_summary(
            trade_counts, float(observed["total_trades"]), higher_is_better=None
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
            "Replay saved profit-targeted LightGBM models with bounded repeated-company "
            "allocation tranches and compare against a permutation null."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-top-fraction", type=float, default=0.05)
    parser.add_argument("--max-company-tranches", type=int, default=2)
    parser.add_argument("--max-expected-downside", type=float, default=0.06)
    parser.add_argument("--starting-capital", type=float, default=10_000.0)
    parser.add_argument("--allocation-pct", type=float, default=0.02)
    parser.add_argument("--max-open-positions", type=int, default=15)
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_profit_tranche_experiment(
        args.experiment_dir,
        permutations=args.permutations,
        seed=args.seed,
        validation_top_fraction=args.validation_top_fraction,
        max_company_tranches=args.max_company_tranches,
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
                "single_tranche_baseline": payload["single_tranche_baseline"],
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
