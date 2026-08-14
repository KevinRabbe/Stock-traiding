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


@dataclass(frozen=True, slots=True)
class ValidationRankBacktestResult:
    training_row_count: int
    model_years: tuple[int, ...]
    total_trades: int
    output_path: Path


def run_validation_rank_backtest(
    experiment_dir: str | Path,
    *,
    validation_top_fraction: float = 0.05,
    max_expected_downside: float = 0.06,
    starting_capital: float = 10_000.0,
    allocation_pct: float = 0.02,
    max_open_positions: int = 15,
    round_trip_cost_bps: float = 20.0,
) -> ValidationRankBacktestResult:
    """Backtest a score cutoff learned only from the preceding validation year.

    The absolute alpha/probability gates from the original baseline are disabled
    here because the first real run showed that their fixed scales were not
    reachable in most years. Instead, each annual model derives one score cutoff
    from the top ``validation_top_fraction`` of its validation predictions, then
    applies that frozen cutoff to the following test year. The existing downside
    risk gate and portfolio/cost rules remain unchanged.

    No test outcome is used to choose a threshold.
    """

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
        # Ranking threshold replaces the two scale-sensitive entry gates.
        min_expected_alpha=-1_000_000.0,
        min_probability_positive=0.0,
        max_expected_downside=max_expected_downside,
        round_trip_cost_bps=round_trip_cost_bps,
    )
    backtester = FixedAllocationBacktester(backtest_config)

    walk_results: list[WalkForwardResult] = []
    year_reports: list[dict] = []
    for year in model_years:
        split = split_by_year.get(year)
        if split is None:
            raise ValueError(f"could not reconstruct walk-forward split for model year {year}")

        model = LightGbmModelBundle.load(models_root / str(year))
        validation_predictions = _predict_matrix(model, split.validation_rows)
        test_predictions = _predict_matrix(model, split.test_rows)
        validation_score = validation_predictions[3]
        score_threshold = float(
            np.quantile(validation_score, 1.0 - validation_top_fraction)
        )

        test_scored = _scored_candidates(split.test_rows, test_predictions)
        selected = tuple(
            candidate
            for candidate in test_scored
            if candidate.prediction.opportunity_score >= score_threshold
        )
        portfolio = backtester.run(selected)
        walk_result = WalkForwardResult(
            test_year=year,
            train_count=len(split.train_rows),
            validation_count=len(split.validation_rows),
            test_count=len(split.test_rows),
            backtest=portfolio,
        )
        walk_results.append(walk_result)

        validation_selected_count = int((validation_score >= score_threshold).sum())
        selected_realized_alpha = [candidate.row.alpha_20d for candidate in selected]
        year_reports.append(
            {
                "test_year": year,
                "score_threshold_from_validation": score_threshold,
                "validation_row_count": len(split.validation_rows),
                "validation_selected_count": validation_selected_count,
                "test_row_count": len(split.test_rows),
                "test_selected_count_before_risk_and_capacity": len(selected),
                "selected_average_realized_alpha_20d": (
                    sum(selected_realized_alpha) / len(selected_realized_alpha)
                    if selected_realized_alpha
                    else None
                ),
                "portfolio": asdict(portfolio),
            }
        )

    summary = summarize_walk_forward(walk_results)
    payload = {
        "schema_version": "validation-ranked-lightgbm-v1",
        "experiment_dir": str(root),
        "training_row_count": len(rows),
        "model_years": list(model_years),
        "selection_policy": {
            "validation_top_fraction": validation_top_fraction,
            "score_cutoff_source": "preceding validation year only",
            "min_expected_alpha": None,
            "min_probability_positive": None,
            "max_expected_downside": max_expected_downside,
            "starting_capital": starting_capital,
            "allocation_pct": allocation_pct,
            "max_open_positions": max_open_positions,
            "round_trip_cost_bps": round_trip_cost_bps,
        },
        "walk_forward_summary": asdict(summary),
        "years": year_reports,
    }
    output_path = root / "validation_rank_backtest.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return ValidationRankBacktestResult(
        training_row_count=len(rows),
        model_years=model_years,
        total_trades=summary.total_trades,
        output_path=output_path,
    )


def _scored_candidates(rows, predictions) -> tuple[ScoredCandidate, ...]:
    alpha, downside, probability, score = predictions
    return tuple(
        ScoredCandidate(
            row=row,
            prediction=OpportunityPrediction(
                expected_alpha_20d=float(alpha[index]),
                expected_downside_20d=float(downside[index]),
                probability_positive_alpha=float(probability[index]),
                opportunity_score=float(score[index]),
            ),
        )
        for index, row in enumerate(rows)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest saved LightGBM models using a score threshold derived only "
            "from each preceding validation year."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--validation-top-fraction", type=float, default=0.05)
    parser.add_argument("--max-expected-downside", type=float, default=0.06)
    parser.add_argument("--starting-capital", type=float, default=10_000.0)
    parser.add_argument("--allocation-pct", type=float, default=0.02)
    parser.add_argument("--max-open-positions", type=int, default=15)
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_validation_rank_backtest(
        args.experiment_dir,
        validation_top_fraction=args.validation_top_fraction,
        max_expected_downside=args.max_expected_downside,
        starting_capital=args.starting_capital,
        allocation_pct=args.allocation_pct,
        max_open_positions=args.max_open_positions,
        round_trip_cost_bps=args.round_trip_cost_bps,
    )
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    summary = payload["walk_forward_summary"]
    print(
        json.dumps(
            {
                "training_row_count": result.training_row_count,
                "model_years": result.model_years,
                "selection_policy": payload["selection_policy"],
                "compounded_return": summary["compounded_return"],
                "profitable_year_rate": summary["profitable_year_rate"],
                "total_trades": summary["total_trades"],
                "average_trade_alpha": summary["average_trade_alpha"],
                "aggregate_profit_factor": summary["aggregate_profit_factor"],
                "worst_realized_drawdown": summary["worst_realized_drawdown"],
                "years": [
                    {
                        "year": item["test_year"],
                        "validation_score_threshold": item[
                            "score_threshold_from_validation"
                        ],
                        "test_selected": item[
                            "test_selected_count_before_risk_and_capacity"
                        ],
                        "trades": len(item["portfolio"]["trades"]),
                        "return": item["portfolio"]["total_return"],
                        "profit_factor": item["portfolio"]["profit_factor"],
                        "selected_average_realized_alpha_20d": item[
                            "selected_average_realized_alpha_20d"
                        ],
                    }
                    for item in payload["years"]
                ],
                "output_path": str(result.output_path),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
