import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from math import ceil
from pathlib import Path

import numpy as np

from stock_trading.backtest import BacktestConfig
from stock_trading.ml import LightGbmModelBundle, TrainingRow
from stock_trading.ml.walk_forward import annual_walk_forward_splits


@dataclass(frozen=True, slots=True)
class PredictionGateSummary:
    row_count: int
    alpha_gate_count: int
    probability_gate_count: int
    downside_gate_count: int
    alpha_probability_gate_count: int
    all_gate_count: int
    positive_score_count: int
    expected_alpha_quantiles: dict[str, float]
    probability_quantiles: dict[str, float]
    expected_downside_quantiles: dict[str, float]
    opportunity_score_quantiles: dict[str, float]
    top_5pct_realized_alpha: float | None
    top_5pct_realized_win_rate: float | None


@dataclass(frozen=True, slots=True)
class LightGbmDiagnosticsResult:
    training_row_count: int
    model_years: tuple[int, ...]
    output_path: Path


def run_lightgbm_diagnostics(
    experiment_dir: str | Path,
    *,
    gates: BacktestConfig | None = None,
) -> LightGbmDiagnosticsResult:
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
    config = gates or BacktestConfig()

    year_reports: list[dict] = []
    for year in model_years:
        split = split_by_year.get(year)
        if split is None:
            raise ValueError(f"could not reconstruct walk-forward split for model year {year}")
        model = LightGbmModelBundle.load(models_root / str(year))
        validation_predictions = _predict_matrix(model, split.validation_rows)
        test_predictions = _predict_matrix(model, split.test_rows)
        year_reports.append(
            {
                "year": year,
                "validation": asdict(
                    _summarize_predictions(
                        validation_predictions,
                        split.validation_rows,
                        config,
                    )
                ),
                "test": asdict(
                    _summarize_predictions(
                        test_predictions,
                        split.test_rows,
                        config,
                    )
                ),
            }
        )

    trigger_counts = {
        "insider": sum(row.features.get("trigger.is_insider") == 1.0 for row in rows),
        "contract": sum(row.features.get("trigger.is_contract") == 1.0 for row in rows),
        "lobbying": sum(row.features.get("trigger.is_lobbying") == 1.0 for row in rows),
    }
    payload = {
        "schema_version": "lightgbm-gate-diagnostics-v1",
        "experiment_dir": str(root),
        "training_row_count": len(rows),
        "model_years": list(model_years),
        "trigger_family_counts": trigger_counts,
        "default_gates": {
            "min_expected_alpha": config.min_expected_alpha,
            "min_probability_positive": config.min_probability_positive,
            "max_expected_downside": config.max_expected_downside,
        },
        "years": year_reports,
    }
    output_path = root / "gate_diagnostics.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return LightGbmDiagnosticsResult(
        training_row_count=len(rows),
        model_years=model_years,
        output_path=output_path,
    )


def _predict_matrix(
    model: LightGbmModelBundle,
    rows: tuple[TrainingRow, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = model.feature_schema.matrix(rows)
    alpha = np.asarray(model.alpha_model.predict(matrix), dtype=np.float64)
    downside = np.maximum(
        0.0,
        np.asarray(model.downside_model.predict(matrix), dtype=np.float64),
    )
    probability = np.clip(
        np.asarray(model.probability_model.predict(matrix), dtype=np.float64),
        0.0,
        1.0,
    )
    score = alpha * probability - model.downside_penalty * downside
    return alpha, downside, probability, score


def _summarize_predictions(
    predictions: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    rows: tuple[TrainingRow, ...],
    config: BacktestConfig,
) -> PredictionGateSummary:
    alpha, downside, probability, score = predictions
    if len(rows) == 0:
        raise ValueError("cannot summarize an empty row set")

    alpha_gate = alpha >= config.min_expected_alpha
    probability_gate = probability >= config.min_probability_positive
    downside_gate = downside <= config.max_expected_downside
    alpha_probability = alpha_gate & probability_gate
    all_gate = alpha_probability & downside_gate

    top_count = max(1, ceil(len(rows) * 0.05))
    order = np.argsort(-score, kind="stable")[:top_count]
    realized_alpha = np.asarray([rows[index].alpha_20d for index in order], dtype=np.float64)

    return PredictionGateSummary(
        row_count=len(rows),
        alpha_gate_count=int(alpha_gate.sum()),
        probability_gate_count=int(probability_gate.sum()),
        downside_gate_count=int(downside_gate.sum()),
        alpha_probability_gate_count=int(alpha_probability.sum()),
        all_gate_count=int(all_gate.sum()),
        positive_score_count=int((score > 0).sum()),
        expected_alpha_quantiles=_quantiles(alpha),
        probability_quantiles=_quantiles(probability),
        expected_downside_quantiles=_quantiles(downside),
        opportunity_score_quantiles=_quantiles(score),
        top_5pct_realized_alpha=float(realized_alpha.mean()),
        top_5pct_realized_win_rate=float((realized_alpha > 0).mean()),
    )


def _quantiles(values: np.ndarray) -> dict[str, float]:
    points = (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99), ("max", 1.0))
    return {name: float(np.quantile(values, quantile)) for name, quantile in points}


def _load_training_rows(path: Path) -> tuple[TrainingRow, ...]:
    rows: list[TrainingRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                rows.append(
                    TrainingRow(
                        event_id=item["event_id"],
                        company_id=item["company_id"],
                        decision_time=datetime.fromisoformat(item["decision_time"]),
                        execution_date=date.fromisoformat(item["execution_date"]),
                        exit_date_20d=date.fromisoformat(item["exit_date_20d"]),
                        features={
                            name: (float(value) if value is not None else None)
                            for name, value in item["features"].items()
                        },
                        stock_return_20d=float(item["stock_return_20d"]),
                        benchmark_return_20d=float(item["benchmark_return_20d"]),
                        alpha_20d=float(item["alpha_20d"]),
                        downside_20d=float(item["downside_20d"]),
                        mfe_20d=float(item["mfe_20d"]),
                        positive_alpha_20d=int(item["positive_alpha_20d"]),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid training row at line {line_number}") from exc
    if not rows:
        raise ValueError("training_rows.jsonl contains no rows")
    return tuple(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose LightGBM prediction scales and deterministic backtest gates "
            "from an existing experiment without rebuilding the dataset."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--min-expected-alpha", type=float, default=0.03)
    parser.add_argument("--min-probability-positive", type=float, default=0.60)
    parser.add_argument("--max-expected-downside", type=float, default=0.06)
    return parser


def main() -> None:
    args = _parser().parse_args()
    gates = BacktestConfig(
        min_expected_alpha=args.min_expected_alpha,
        min_probability_positive=args.min_probability_positive,
        max_expected_downside=args.max_expected_downside,
    )
    result = run_lightgbm_diagnostics(args.experiment_dir, gates=gates)
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "training_row_count": result.training_row_count,
                "model_years": result.model_years,
                "trigger_family_counts": payload["trigger_family_counts"],
                "default_gates": payload["default_gates"],
                "years": [
                    {
                        "year": item["year"],
                        "validation_all_gate_count": item["validation"]["all_gate_count"],
                        "test_all_gate_count": item["test"]["all_gate_count"],
                        "test_alpha_gate_count": item["test"]["alpha_gate_count"],
                        "test_probability_gate_count": item["test"]["probability_gate_count"],
                        "test_downside_gate_count": item["test"]["downside_gate_count"],
                        "test_expected_alpha_p99": item["test"]["expected_alpha_quantiles"]["p99"],
                        "test_expected_alpha_max": item["test"]["expected_alpha_quantiles"]["max"],
                        "test_probability_p99": item["test"]["probability_quantiles"]["p99"],
                        "test_probability_max": item["test"]["probability_quantiles"]["max"],
                        "test_expected_downside_p99": item["test"]["expected_downside_quantiles"]["p99"],
                        "test_top_5pct_realized_alpha": item["test"]["top_5pct_realized_alpha"],
                    }
                    for item in payload["years"]
                ],
                "output_path": str(result.output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
