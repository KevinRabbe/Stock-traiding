from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from math import isclose
from pathlib import Path

from stock_trading.engine import (
    FeatureSnapshot,
    FixedAllocationPortfolioPolicy,
    PassThroughOpportunityRiskPolicy,
    PassThroughPortfolioRiskPolicy,
)
from stock_trading.market import DuckDbMarketStore
from stock_trading.market.execution_time import decision_market_date
from stock_trading.ml.multi_horizon import build_multi_horizon_targets
from stock_trading.ml.system_context import augment_system_context_features
from stock_trading.ml.walk_forward import annual_walk_forward_splits
from stock_trading.research import (
    HistoricalCandidate,
    HistoricalOutcome,
    HistoricalStrategyBacktester,
    HistoricalYearResult,
    summarize_historical_years,
)
from stock_trading.strategies import (
    V5StrategyConfig,
    build_v5_strategy_from_saved_models,
)

from .lightgbm_diagnostics import _load_training_rows
from .lightgbm_validation_rank import _json_safe


@dataclass(frozen=True, slots=True)
class V5EngineReplayResult:
    source_training_row_count: int
    complete_multi_horizon_row_count: int
    model_years: tuple[int, ...]
    output_path: Path


def run_v5_engine_replay(
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
) -> V5EngineReplayResult:
    """Prove the generic strategy/research architecture reproduces V5 exactly."""

    root = Path(experiment_dir)
    rows_path = root / "training_rows.jsonl"
    v5_report_path = root / "profit_target_v5_backtest.json"
    models_root = root / "profit_models_v5"
    for required in (rows_path, v5_report_path, models_root):
        if not required.exists():
            raise FileNotFoundError(f"missing V5 engine replay prerequisite: {required}")

    v5_payload = json.loads(v5_report_path.read_text(encoding="utf-8"))
    expected_years = {int(item["year"]): item for item in v5_payload["years"]}
    source_rows = _load_training_rows(rows_path)
    market_store = DuckDbMarketStore(market_db)
    market_store.enable_read_cache(max_series=market_read_cache_series)
    targets = build_multi_horizon_targets(
        source_rows,
        market_store,
        benchmark_security_id=benchmark_security_id,
        horizons=(5, 20, 60),
        verify_existing_20d=True,
    )
    complete_rows = tuple(row for row in source_rows if row.event_id in targets)
    rows = augment_system_context_features(complete_rows)
    splits = annual_walk_forward_splits(rows)
    if not splits:
        raise ValueError("no walk-forward splits available for V5 engine replay")

    strategy_config = V5StrategyConfig(
        validation_top_fraction=validation_top_fraction,
        alpha_rank_weight=alpha_rank_weight,
        calibration_window_days=calibration_window_days,
        min_expected_return=round_trip_cost_bps / 10_000.0,
        max_expected_downside=max_expected_downside,
    )
    # V5's legacy portfolio has no independent gross-exposure gate; its bound is
    # the 15-position cap at 2% target allocation. Keep 100% here so this replay
    # proves strategy/engine identity rather than introducing a new risk policy.
    portfolio_policy = FixedAllocationPortfolioPolicy(
        allocation_pct=allocation_pct,
        max_open_positions=max_open_positions,
        max_gross_exposure_pct=1.0,
        one_position_per_company=True,
    )
    backtester = HistoricalStrategyBacktester(
        starting_capital=starting_capital,
        round_trip_cost_bps=round_trip_cost_bps,
    )
    year_results: list[HistoricalYearResult] = []
    year_reports: list[dict] = []

    for split in splits:
        validation_candidates = _feature_snapshots(split.validation_rows, market_store)
        strategy = build_v5_strategy_from_saved_models(
            models_root,
            model_year=split.test_year,
            validation_candidates=validation_candidates,
            config=strategy_config,
        )
        historical_candidates = _historical_candidates(
            split.test_rows,
            targets,
            market_store,
        )
        result = backtester.run(
            strategy=strategy,
            candidates=historical_candidates,
            opportunity_risk=PassThroughOpportunityRiskPolicy(),
            portfolio_policy=portfolio_policy,
            portfolio_risk=PassThroughPortfolioRiskPolicy(),
        )
        expected = expected_years.get(split.test_year)
        if expected is None:
            raise RuntimeError(f"V5 report missing year {split.test_year}")
        if len(result.trades) != int(expected["trades"]):
            raise RuntimeError(
                f"engine V5 replay trade count diverged in {split.test_year}: "
                f"{len(result.trades)} != {expected['trades']}"
            )
        if not isclose(
            result.total_return,
            float(expected["return"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"engine V5 replay return diverged in {split.test_year}: "
                f"{result.total_return} != {expected['return']}"
            )

        year_results.append(HistoricalYearResult(split.test_year, result))
        year_reports.append(
            {
                "year": split.test_year,
                "trades": len(result.trades),
                "return": result.total_return,
                "profit_factor": result.profit_factor,
                "realized_drawdown": result.realized_max_drawdown,
                "trade_horizon_counts": {
                    str(horizon): count
                    for horizon, count in sorted(
                        Counter(trade.horizon_sessions for trade in result.trades).items()
                    )
                },
                "identity_verified_against_v5": True,
            }
        )

    summary = summarize_historical_years(tuple(year_results))
    observed = {
        "compounded_return": summary.scorecard.compounded_return,
        "profitable_year_rate": summary.scorecard.profitable_year_rate,
        "total_trades": summary.scorecard.total_trades,
        "average_trade_alpha": summary.scorecard.average_trade_alpha,
        "aggregate_profit_factor": summary.scorecard.profit_factor,
        "worst_realized_drawdown": summary.scorecard.worst_realized_drawdown,
    }
    _verify_observed_identity(observed, v5_payload["observed"])

    payload = _json_safe(
        {
            "schema_version": "strategy-engine-v5-exact-replay",
            "experiment_dir": str(root),
            "market_db": str(market_db),
            "benchmark_security_id": benchmark_security_id,
            "source_training_row_count": len(source_rows),
            "complete_multi_horizon_row_count": len(rows),
            "model_years": [split.test_year for split in splits],
            "strategy": asdict(strategy_config),
            "portfolio_policy": asdict(portfolio_policy),
            "architecture": {
                "generic_strategy_plugin": True,
                "generic_historical_backtester": True,
                "saved_models_reused": True,
                "predictor_retrained": False,
                "exact_v5_identity_verified": True,
            },
            "v5_baseline": v5_payload["observed"],
            "observed": observed,
            "concentration": {
                "best_year": summary.best_year,
                "compounded_return_excluding_best_year": (
                    summary.compounded_return_excluding_best_year
                ),
            },
            "market_cache_stats": market_store.read_cache_stats(),
            "years": year_reports,
        }
    )
    output_path = root / "strategy_engine_v5_replay.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return V5EngineReplayResult(
        source_training_row_count=len(source_rows),
        complete_multi_horizon_row_count=len(rows),
        model_years=tuple(split.test_year for split in splits),
        output_path=output_path,
    )


def _feature_snapshots(rows, market_store: DuckDbMarketStore) -> tuple[FeatureSnapshot, ...]:
    snapshots: list[FeatureSnapshot] = []
    for row in rows:
        security_id = market_store.security_for_company(
            row.company_id,
            decision_market_date(row.decision_time),
        )
        if security_id is None:
            raise RuntimeError(
                f"no PIT security mapping for {row.company_id} at {row.decision_time}"
            )
        snapshots.append(
            FeatureSnapshot(
                candidate_id=row.event_id,
                event_id=row.event_id,
                company_id=row.company_id,
                security_id=security_id,
                decision_time=row.decision_time,
                execution_date=row.execution_date,
                features=row.features,
            )
        )
    return tuple(snapshots)


def _historical_candidates(rows, targets, market_store) -> tuple[HistoricalCandidate, ...]:
    snapshots = _feature_snapshots(rows, market_store)
    rows_by_event = {row.event_id: row for row in rows}
    results: list[HistoricalCandidate] = []
    for snapshot in snapshots:
        row = rows_by_event[snapshot.event_id]
        by_horizon = targets[row.event_id]
        results.append(
            HistoricalCandidate(
                snapshot=snapshot,
                outcomes={
                    horizon: HistoricalOutcome(
                        horizon_sessions=horizon,
                        exit_date=target.exit_date,
                        stock_return=target.stock_return,
                        alpha=target.alpha,
                        downside=target.downside,
                    )
                    for horizon, target in by_horizon.items()
                },
            )
        )
    return tuple(results)


def _verify_observed_identity(observed: dict, expected: dict) -> None:
    if int(observed["total_trades"]) != int(expected["total_trades"]):
        raise RuntimeError("engine V5 replay diverged for total_trades")
    for field in (
        "compounded_return",
        "profitable_year_rate",
        "average_trade_alpha",
        "aggregate_profit_factor",
        "worst_realized_drawdown",
    ):
        left = observed[field]
        right = expected[field]
        if left is None or right is None:
            if left != right:
                raise RuntimeError(f"engine V5 replay diverged for {field}")
            continue
        if not isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                f"engine V5 replay diverged for {field}: {left} != {right}"
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay saved V5 models through the generic strategy engine/research "
            "architecture and require exact year-by-year V5 identity."
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
    result = run_v5_engine_replay(
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
