import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

from stock_trading.backtest import (
    BacktestConfig,
    FixedAllocationBacktester,
    evaluate_score_buckets,
    profit_without_best_trades,
    summarize_walk_forward,
)
from stock_trading.core import EventType
from stock_trading.market import CandidateSnapshotBuilder, DuckDbMarketStore
from stock_trading.ml import (
    LightGbmTrainer,
    LightGbmTrainingConfig,
    TrainingDatasetBuilder,
    TrainingRow,
)
from stock_trading.ml.walk_forward import (
    WalkForwardResult,
    annual_walk_forward_splits,
)
from stock_trading.storage import DuckDbEventStore


_MODEL_EVENT_TYPES = (
    EventType.INSIDER_TRANSACTION,
    EventType.GOVERNMENT_CONTRACT,
    EventType.LOBBYING_ACTIVITY,
)


@dataclass(frozen=True, slots=True)
class HistoricalExperimentConfig:
    events_db: Path
    market_db: Path
    benchmark_company_id: str
    output_dir: Path
    first_test_year: int | None = None
    positive_alpha_threshold: float = 0.02
    target_horizon: int = 20
    feature_lookback_bars: int = 260
    min_train_rows: int = 100
    min_validation_rows: int = 20
    min_test_rows: int = 1
    top_feature_count: int = 30


@dataclass(frozen=True, slots=True)
class HistoricalExperimentResult:
    source_event_count: int
    mapped_company_count: int
    event_count: int
    trigger_count: int
    aggregated_trigger_event_count: int
    training_row_count: int
    tested_years: tuple[int, ...]
    output_dir: Path


def run_historical_experiment(
    config: HistoricalExperimentConfig,
    *,
    training_config: LightGbmTrainingConfig | None = None,
    backtest_config: BacktestConfig | None = None,
) -> HistoricalExperimentResult:
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)

    event_store = DuckDbEventStore(config.events_db)
    market_store = DuckDbMarketStore(config.market_db)
    source_event_count = event_store.count()
    mapped_company_ids = _mapped_company_ids(market_store)
    if not mapped_company_ids:
        raise ValueError("market store has no verified company-to-security mappings")

    # The normalized event database can contain millions of companies/events while
    # a development market universe may contain only tens or hundreds of verified
    # securities. Load only event families used by the model for companies that
    # have an explicit market mapping. This is both faster and safer than probing
    # unresolved companies one event at a time.
    all_events = event_store.all_events(
        company_ids=mapped_company_ids,
        event_types=_MODEL_EVENT_TYPES,
    )
    trigger_events = tuple(
        event
        for event in all_events
        if event.company_id and event.event_type in _MODEL_EVENT_TYPES
    )

    snapshot_builder = CandidateSnapshotBuilder(
        market_store,
        benchmark_security_id=config.benchmark_company_id,
        feature_lookback_bars=config.feature_lookback_bars,
        label_horizons=(1, 5, config.target_horizon, 60),
    )
    dataset_builder = TrainingDatasetBuilder(
        snapshot_builder,
        positive_alpha_threshold=config.positive_alpha_threshold,
        target_horizon=config.target_horizon,
    )
    rows = dataset_builder.build(trigger_events, all_events=all_events)
    if not rows:
        raise ValueError("no mature model-ready rows could be built from the supplied stores")

    aggregated_trigger_event_count = sum(len(row.trigger_event_ids) for row in rows)
    max_triggers_per_opportunity = max(len(row.trigger_event_ids) for row in rows)
    mean_triggers_per_opportunity = aggregated_trigger_event_count / len(rows)

    _write_training_rows(output / "training_rows.jsonl", rows)

    effective_training_config = training_config or LightGbmTrainingConfig()
    trainer = LightGbmTrainer(effective_training_config)
    backtester = FixedAllocationBacktester(backtest_config)
    splits = annual_walk_forward_splits(
        rows,
        first_test_year=config.first_test_year,
        min_train_rows=config.min_train_rows,
        min_validation_rows=config.min_validation_rows,
        min_test_rows=config.min_test_rows,
    )
    if not splits:
        raise ValueError("no walk-forward years satisfy the configured minimum row counts")

    walk_results: list[WalkForwardResult] = []
    year_reports: list[dict] = []
    for split in splits:
        model = trainer.train(
            split.train_rows,
            split.validation_rows,
            positive_alpha_threshold=config.positive_alpha_threshold,
        )
        model_dir = output / "models" / str(split.test_year)
        model.save(model_dir)

        scored = backtester.score_rows(split.test_rows, model)
        portfolio = backtester.run(scored)
        walk_result = WalkForwardResult(
            test_year=split.test_year,
            train_count=len(split.train_rows),
            validation_count=len(split.validation_rows),
            test_count=len(split.test_rows),
            backtest=portfolio,
        )
        walk_results.append(walk_result)

        importance = sorted(
            model.feature_importance().items(),
            key=lambda item: (-item[1], item[0]),
        )[: config.top_feature_count]
        year_reports.append(
            {
                "test_year": split.test_year,
                "train_count": len(split.train_rows),
                "validation_count": len(split.validation_rows),
                "test_count": len(split.test_rows),
                "portfolio": _jsonable(portfolio),
                "score_buckets": _jsonable(evaluate_score_buckets(scored)),
                "top_alpha_feature_importance": [
                    {"feature": name, "gain": gain} for name, gain in importance
                ],
                "profit_without_best_1": profit_without_best_trades(portfolio, 1),
                "profit_without_best_5": profit_without_best_trades(portfolio, 5),
                "profit_without_best_10": profit_without_best_trades(portfolio, 10),
            }
        )

    summary = summarize_walk_forward(walk_results)
    report = {
        "schema_version": "historical-lightgbm-v3-opportunity",
        "inputs": {
            "events_db": str(config.events_db),
            "events_db_sha256": _sha256_file(config.events_db),
            "market_db": str(config.market_db),
            "market_db_sha256": _sha256_file(config.market_db),
            "benchmark_security_id": config.benchmark_company_id,
        },
        "dataset": {
            "row_unit": "company_execution_session_opportunity",
            "source_event_count": source_event_count,
            "mapped_company_count": len(mapped_company_ids),
            "selected_event_count": len(all_events),
            "raw_trigger_event_count": len(trigger_events),
            "aggregated_trigger_event_count": aggregated_trigger_event_count,
            "training_row_count": len(rows),
            "mean_trigger_events_per_opportunity": mean_triggers_per_opportunity,
            "max_trigger_events_per_opportunity": max_triggers_per_opportunity,
            "decision_start": min(row.decision_time for row in rows).isoformat(),
            "decision_end": max(row.decision_time for row in rows).isoformat(),
            "positive_alpha_threshold": config.positive_alpha_threshold,
            "target_horizon": config.target_horizon,
        },
        "training_config": _jsonable(effective_training_config),
        "backtest_config": _jsonable(backtest_config or BacktestConfig()),
        "walk_forward_summary": _jsonable(summary),
        "years": year_reports,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    return HistoricalExperimentResult(
        source_event_count=source_event_count,
        mapped_company_count=len(mapped_company_ids),
        event_count=len(all_events),
        trigger_count=len(trigger_events),
        aggregated_trigger_event_count=aggregated_trigger_event_count,
        training_row_count=len(rows),
        tested_years=tuple(split.test_year for split in splits),
        output_dir=output,
    )


def _mapped_company_ids(market_store: DuckDbMarketStore) -> tuple[str, ...]:
    with market_store._connect() as connection:
        rows = connection.execute(
            "SELECT DISTINCT company_id FROM company_security_map ORDER BY company_id"
        ).fetchall()
    return tuple(str(row[0]) for row in rows)


def _write_training_rows(path: Path, rows: tuple[TrainingRow, ...]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True, allow_nan=False))
            handle.write("\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value):
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
        return None
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the point-in-time LightGBM historical trading experiment."
    )
    parser.add_argument("--events-db", type=Path, required=True)
    parser.add_argument("--market-db", type=Path, required=True)
    parser.add_argument(
        "--benchmark-security-id",
        "--benchmark-company-id",
        dest="benchmark_security_id",
        required=True,
        help="Security ID for the benchmark series (legacy --benchmark-company-id is accepted).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-test-year", type=int)
    parser.add_argument("--min-train-rows", type=int, default=100)
    parser.add_argument("--min-validation-rows", type=int, default=20)
    parser.add_argument("--starting-capital", type=float, default=10_000.0)
    parser.add_argument("--allocation-pct", type=float, default=0.02)
    parser.add_argument("--max-open-positions", type=int, default=15)
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_historical_experiment(
        HistoricalExperimentConfig(
            events_db=args.events_db,
            market_db=args.market_db,
            benchmark_company_id=args.benchmark_security_id,
            output_dir=args.output_dir,
            first_test_year=args.first_test_year,
            min_train_rows=args.min_train_rows,
            min_validation_rows=args.min_validation_rows,
        ),
        backtest_config=BacktestConfig(
            starting_capital=args.starting_capital,
            allocation_pct=args.allocation_pct,
            max_open_positions=args.max_open_positions,
            round_trip_cost_bps=args.round_trip_cost_bps,
        ),
    )
    print(
        json.dumps(
            {
                "source_event_count": result.source_event_count,
                "mapped_company_count": result.mapped_company_count,
                "selected_event_count": result.event_count,
                "raw_trigger_event_count": result.trigger_count,
                "aggregated_trigger_event_count": result.aggregated_trigger_event_count,
                "opportunity_row_count": result.training_row_count,
                "training_row_count": result.training_row_count,
                "tested_years": result.tested_years,
                "output_dir": str(result.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
