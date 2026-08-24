from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from stock_trading.engine import (
    BasicOpportunityRiskPolicy,
    FeatureSnapshot,
    FixedAllocationPortfolioPolicy,
    Opportunity,
    PortfolioSnapshot,
    PassThroughPortfolioRiskPolicy,
)
from stock_trading.ml import LightGbmTrainer, ProfitLightGbmTrainer
from stock_trading.ml.multi_horizon import multi_horizon_maturity_dates, row_for_horizon
from stock_trading.ml.walk_forward import annual_walk_forward_splits
from stock_trading.research import HistoricalCandidate
from stock_trading.research.execution_realism import ExecutionRealisticHistoricalBacktester
from stock_trading.research.strategy_factory import (
    StrategyVariantSpec,
    apply_feature_profile,
    training_window_rows,
)
from stock_trading.strategies.v5_adaptive_horizon import (
    V5AdaptiveHorizonStrategy,
    V5CalibrationState,
    V5HorizonModels,
    V5StrategyConfig,
)

from . import lightgbm_strategy_factory as base_factory
from . import lightgbm_strategy_factory_executable as executable
from .lightgbm_validation_rank import _json_safe


SCHEMA_VERSION = "v5-strict-profit-proof-v1"
CURRENT_V5_STRATEGY_ID = "lightgbm-v5-adaptive-horizon"
DEFAULT_MAX_GROSS_EXPOSURE_PCT = 0.30
DEFAULT_MARKET_QUALITY_MANIFEST = Path("data/manifests/market_quality_verified.json")


@dataclass(frozen=True, slots=True)
class V5StrictProfitProofResult:
    output_path: Path
    model_years: tuple[int, ...]
    total_trades: int
    verdict: str


@dataclass(frozen=True, slots=True)
class _PreparedProof:
    strategies: Mapping[int, V5AdaptiveHorizonStrategy]
    candidates: tuple[HistoricalCandidate, ...]
    model_years: tuple[int, ...]
    split_diagnostics: tuple[dict, ...]
    source_row_count: int
    quality_removed_row_count: int
    pit_liquidity_removed_row_count: int
    executable_row_count: int


class _AnnualV5StrategyRouter:
    """Route one chronological replay into the model frozen for that test year."""

    def __init__(
        self,
        strategies: Mapping[int, V5AdaptiveHorizonStrategy],
        *,
        strategy_id: str = CURRENT_V5_STRATEGY_ID,
    ) -> None:
        if not strategies:
            raise ValueError("annual V5 router requires at least one strategy")
        self._strategies = dict(strategies)
        self._strategy_id = strategy_id
        for year, strategy in self._strategies.items():
            if year <= 0:
                raise ValueError("annual V5 strategy year must be positive")
            if strategy.strategy_id != strategy_id:
                raise ValueError("annual V5 strategy_id mismatch")

    @property
    def strategy_id(self) -> str:
        return self._strategy_id

    def evaluate(
        self,
        candidates: tuple[FeatureSnapshot, ...],
        portfolio: PortfolioSnapshot,
    ) -> tuple[Opportunity, ...]:
        if not candidates:
            return ()
        execution_years = {item.execution_date.year for item in candidates}
        if len(execution_years) != 1:
            raise ValueError("historical V5 batch crosses model years")
        year = next(iter(execution_years))
        strategy = self._strategies.get(year)
        if strategy is None:
            raise ValueError(f"no strict V5 strategy for execution year {year}")
        return strategy.evaluate(candidates, portfolio)


def current_v5_profit_proof_spec() -> StrategyVariantSpec:
    """Exact structural/training settings used by the legacy V5 PAPER champion."""

    return StrategyVariantSpec(
        variant_id=CURRENT_V5_STRATEGY_ID,
        feature_profile="full",
        training_window_years=None,
        tree_profile="baseline",
        horizons=(5, 20, 60),
        alpha_rank_weight=0.25,
        seed=42,
        validation_top_fraction=0.05,
        calibration_window_days=365,
        max_expected_downside=0.06,
    )


def run_v5_profit_proof(
    experiment_dir: str | Path,
    *,
    runtime_dir: str | Path = "data/runtime",
    market_db: str | Path | None = None,
    benchmark_security_id: str | None = None,
    starting_capital: float = 10_000.0,
    allocation_pct: float = 0.02,
    max_open_positions: int = 15,
    max_gross_exposure_pct: float = DEFAULT_MAX_GROSS_EXPOSURE_PCT,
    round_trip_cost_bps: float = 20.0,
    max_trailing_adv_participation_pct: float = 0.01,
    max_entry_day_participation_pct: float = 0.01,
    market_quality_manifest: str | Path = DEFAULT_MARKET_QUALITY_MANIFEST,
    min_train_rows: int = 100,
    market_read_cache_series: int = 200,
) -> V5StrictProfitProofResult:
    """Falsify or support the current V5 design under stricter historical replay.

    This is deliberately not a parameter search. It retrains exactly one strategy
    design: the current V5 PAPER champion structure. Training/validation eligibility
    is fenced by the latest required 5/20/60-session outcome date, execution must fit
    verified PIT/entry-day liquidity, and one continuous portfolio carries capital,
    positions, slot usage and the 30% gross-exposure cap across year boundaries.
    """

    if starting_capital <= 0:
        raise ValueError("starting_capital must be > 0")
    if not 0.0 < allocation_pct <= 1.0:
        raise ValueError("allocation_pct must be in (0, 1]")
    if max_open_positions <= 0:
        raise ValueError("max_open_positions must be > 0")
    if not 0.0 < max_gross_exposure_pct <= 1.0:
        raise ValueError("max_gross_exposure_pct must be in (0, 1]")
    if round_trip_cost_bps < 0:
        raise ValueError("round_trip_cost_bps must be >= 0")
    if min_train_rows <= 0 or market_read_cache_series <= 0:
        raise ValueError("training/cache limits must be > 0")
    for name, value in (
        ("max_trailing_adv_participation_pct", max_trailing_adv_participation_pct),
        ("max_entry_day_participation_pct", max_entry_day_participation_pct),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")

    root = Path(experiment_dir)
    runtime_root = Path(runtime_dir)
    if not (root / "training_rows.jsonl").exists():
        raise FileNotFoundError(f"missing training rows: {root / 'training_rows.jsonl'}")

    resolved_market_db, resolved_benchmark = _resolve_market_inputs(
        runtime_root,
        market_db=market_db,
        benchmark_security_id=benchmark_security_id,
    )
    quality_manifest = Path(market_quality_manifest)
    if not quality_manifest.exists():
        raise FileNotFoundError(f"missing market quality manifest: {quality_manifest}")

    spec = current_v5_profit_proof_spec()
    profitable_threshold = round_trip_cost_bps / 10_000.0
    if abs(profitable_threshold - 0.002) > 1e-12:
        raise ValueError(
            "strict current-V5 proof requires the live 20 bps round-trip cost floor"
        )

    prepared = executable._prepare_executable_data(
        root,
        market_db=resolved_market_db,
        benchmark_security_id=resolved_benchmark,
        market_quality_manifest=quality_manifest,
        market_read_cache_series=market_read_cache_series,
    )
    context = executable._ExecutableWorkerContext(
        rows=prepared.rows,
        targets=prepared.targets,
        security_ids=prepared.security_ids,
        entry_liquidity=prepared.entry_liquidity,
        invalid_target_keys=prepared.invalid_target_keys,
        starting_capital=starting_capital,
        allocation_pct=allocation_pct,
        max_open_positions=max_open_positions,
        round_trip_cost_bps=round_trip_cost_bps,
        min_train_rows=min_train_rows,
        max_trailing_adv_participation_pct=max_trailing_adv_participation_pct,
        max_entry_day_participation_pct=max_entry_day_participation_pct,
    )
    proof = _prepare_strict_v5_replay(spec, context)

    router = _AnnualV5StrategyRouter(proof.strategies)
    portfolio_policy = FixedAllocationPortfolioPolicy(
        allocation_pct=allocation_pct,
        max_open_positions=max_open_positions,
        max_gross_exposure_pct=max_gross_exposure_pct,
        one_position_per_company=True,
    )
    backtester = ExecutionRealisticHistoricalBacktester(
        starting_capital=starting_capital,
        round_trip_cost_bps=round_trip_cost_bps,
    )
    backtest, execution_diagnostics = backtester.run(
        strategy=router,
        candidates=proof.candidates,
        opportunity_risk=BasicOpportunityRiskPolicy(
            max_expected_downside=spec.max_expected_downside,
            min_expected_return=0.0,
            min_probability_positive=0.0,
        ),
        portfolio_policy=portfolio_policy,
        portfolio_risk=PassThroughPortfolioRiskPolicy(),
        entry_liquidity=prepared.entry_liquidity,
        max_entry_day_participation_pct=max_entry_day_participation_pct,
    )

    trades = tuple(backtest.trades)
    average_trade_alpha = (
        sum(item.alpha for item in trades) / len(trades) if trades else None
    )
    average_net_trade_return = (
        sum(item.net_return for item in trades) / len(trades) if trades else None
    )
    win_rate = (
        sum(item.net_return > 0 for item in trades) / len(trades) if trades else None
    )
    pnl_by_entry_year: dict[int, float] = defaultdict(float)
    trades_by_entry_year: Counter[int] = Counter()
    horizons: Counter[int] = Counter()
    for trade in trades:
        pnl_by_entry_year[trade.entry_date.year] += trade.pnl
        trades_by_entry_year[trade.entry_date.year] += 1
        horizons[trade.horizon_sessions] += 1
    best_entry_year = (
        max(pnl_by_entry_year, key=lambda year: (pnl_by_entry_year[year], -year))
        if pnl_by_entry_year
        else None
    )
    net_profit = backtest.ending_capital - backtest.starting_capital
    net_profit_excluding_best_entry_year = (
        net_profit - pnl_by_entry_year[best_entry_year]
        if best_entry_year is not None
        else None
    )

    scorecard = {
        "starting_capital": backtest.starting_capital,
        "ending_capital": backtest.ending_capital,
        "net_profit": net_profit,
        "total_return": backtest.total_return,
        "profit_factor": backtest.profit_factor,
        "realized_max_drawdown": backtest.realized_max_drawdown,
        "total_trades": len(trades),
        "win_rate": win_rate,
        "average_net_trade_return": average_net_trade_return,
        "average_trade_alpha": average_trade_alpha,
        "best_entry_year": best_entry_year,
        "best_entry_year_pnl": (
            pnl_by_entry_year[best_entry_year] if best_entry_year is not None else None
        ),
        "net_profit_excluding_best_entry_year": net_profit_excluding_best_entry_year,
        "trade_horizon_counts": {
            str(horizon): int(count) for horizon, count in sorted(horizons.items())
        },
    }
    viability = minimum_viability(scorecard)
    legacy = _load_legacy_comparison(root)

    payload = _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_dir": str(root),
            "runtime_dir": str(runtime_root),
            "market_db": str(resolved_market_db),
            "benchmark_security_id": resolved_benchmark,
            "verdict": viability["verdict"],
            "viability": viability,
            "proof_scope": {
                "question": "Would the current V5 design have produced positive historical PnL under stricter live-like constraints?",
                "single_fixed_strategy_design": True,
                "parameter_search": False,
                "predictors_retrained_from_scratch_each_test_year": True,
                "same_strategy_class_as_runtime": True,
                "full_horizon_maturity_required": True,
                "maturity_fence": "latest_requested_horizon_exit_before_test_year",
                "realized_outcomes_hidden_from_strategy": True,
                "execution_realism_enabled": True,
                "continuous_portfolio_across_test_years": True,
                "current_paper_thresholds_changed": False,
                "live_forward_proof": False,
            },
            "strategy_spec": spec.as_json(),
            "portfolio_policy": asdict(portfolio_policy),
            "execution_policy": {
                "round_trip_cost_bps": round_trip_cost_bps,
                "entry_price": "adjusted_open_on_exact_execution_session",
                "exit_price": "adjusted_close_on_exact_selected_horizon_session",
                "trailing_adv_feature": "market.avg_dollar_volume_20d",
                "max_trailing_adv_participation_pct": max_trailing_adv_participation_pct,
                "entry_day_notional": "raw_open_x_raw_daily_volume",
                "max_entry_day_participation_pct": max_entry_day_participation_pct,
                "full_fill_required": True,
                "market_quality_manifest": str(quality_manifest),
            },
            "data": {
                "source_row_count": proof.source_row_count,
                "quality_removed_row_count": proof.quality_removed_row_count,
                "pit_liquidity_removed_row_count": proof.pit_liquidity_removed_row_count,
                "executable_row_count": proof.executable_row_count,
                "model_years": list(proof.model_years),
                "candidate_count": len(proof.candidates),
                "verified_quality_exclusion_count": prepared.quality_exclusion_count,
                "invalid_target_count": len(prepared.invalid_target_keys),
                "market_cache_stats": prepared.market_cache_stats,
            },
            "execution_diagnostics": {
                "rejected_entry_liquidity": execution_diagnostics.rejected_entry_liquidity,
                "rejected_cash": backtest.rejected_cash,
            },
            "scorecard": scorecard,
            "entry_year_diagnostics": [
                {
                    "year": year,
                    "trade_count": int(trades_by_entry_year[year]),
                    "realized_pnl": pnl_by_entry_year[year],
                }
                for year in sorted(trades_by_entry_year)
            ],
            "walk_forward_splits": list(proof.split_diagnostics),
            "legacy_v5_comparison": legacy,
        }
    )
    output_path = root / "v5_strict_profit_proof.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return V5StrictProfitProofResult(
        output_path=output_path,
        model_years=proof.model_years,
        total_trades=len(trades),
        verdict=str(viability["verdict"]),
    )


def _prepare_strict_v5_replay(
    spec: StrategyVariantSpec,
    context: executable._ExecutableWorkerContext,
) -> _PreparedProof:
    executable_rows, quality_removed, liquidity_removed = executable._filter_executable_rows(
        context.rows,
        spec,
        context,
    )
    rows = apply_feature_profile(executable_rows, spec.feature_profile)
    maturity_dates = multi_horizon_maturity_dates(
        rows,
        context.targets,
        horizons=spec.horizons,
    )
    splits = annual_walk_forward_splits(rows, maturity_dates=maturity_dates)
    if not splits:
        raise ValueError("no maturity-safe annual walk-forward splits for strict V5 proof")

    profitable_threshold = context.round_trip_cost_bps / 10_000.0
    profit_trainer = ProfitLightGbmTrainer(spec.training_config)
    alpha_trainer = LightGbmTrainer(spec.training_config)
    strategies: dict[int, V5AdaptiveHorizonStrategy] = {}
    candidates: list[HistoricalCandidate] = []
    split_diagnostics: list[dict] = []

    for split in splits:
        train_rows = training_window_rows(
            split.train_rows,
            test_year=split.test_year,
            window_years=spec.training_window_years,
        )
        if len(train_rows) < context.min_train_rows:
            raise ValueError(
                f"strict V5 has only {len(train_rows)} training rows for test year "
                f"{split.test_year}"
            )

        models: dict[int, V5HorizonModels] = {}
        for horizon in spec.horizons:
            train_h = tuple(
                row_for_horizon(row, context.targets[row.event_id][horizon])
                for row in train_rows
            )
            validation_h = tuple(
                row_for_horizon(row, context.targets[row.event_id][horizon])
                for row in split.validation_rows
            )
            models[horizon] = V5HorizonModels(
                profit=profit_trainer.train(
                    train_h,
                    validation_h,
                    profitable_return_threshold=profitable_threshold,
                ),
                alpha=alpha_trainer.train(train_h, validation_h),
            )

        validation_candidates = base_factory._feature_snapshots(
            split.validation_rows,
            context.security_ids,
        )
        strategy_config = V5StrategyConfig(
            strategy_id=spec.variant_id,
            horizons=spec.horizons,
            validation_top_fraction=spec.validation_top_fraction,
            alpha_rank_weight=spec.alpha_rank_weight,
            calibration_window_days=spec.calibration_window_days,
            min_expected_return=profitable_threshold,
            max_expected_downside=spec.max_expected_downside,
        )
        calibration = V5CalibrationState.from_validation(
            validation_candidates,
            models,
            strategy_config,
        )
        strategy = V5AdaptiveHorizonStrategy(models, calibration, strategy_config)
        strategies[split.test_year] = strategy
        current_candidates = base_factory._historical_candidates(
            split.test_rows,
            context.targets,
            context.security_ids,
        )
        candidates.extend(current_candidates)
        split_diagnostics.append(
            {
                "test_year": split.test_year,
                "train_count": len(train_rows),
                "validation_count": len(split.validation_rows),
                "test_count": len(split.test_rows),
                "latest_training_maturity_before_test_year": True,
            }
        )

    materialized = tuple(sorted(candidates, key=lambda item: (
        item.snapshot.execution_date,
        item.snapshot.decision_time,
        item.snapshot.candidate_id,
    )))
    if len({item.snapshot.candidate_id for item in materialized}) != len(materialized):
        raise RuntimeError("strict V5 proof produced duplicate candidate identities")
    return _PreparedProof(
        strategies=strategies,
        candidates=materialized,
        model_years=tuple(sorted(strategies)),
        split_diagnostics=tuple(split_diagnostics),
        source_row_count=len(context.rows),
        quality_removed_row_count=quality_removed,
        pit_liquidity_removed_row_count=liquidity_removed,
        executable_row_count=len(executable_rows),
    )


def minimum_viability(scorecard: Mapping[str, object]) -> dict:
    """Apply only pre-existing project-level profitability/robustness standards."""

    alpha = scorecard.get("average_trade_alpha")
    excluding_best = scorecard.get("net_profit_excluding_best_entry_year")
    checks = {
        "positive_net_return": float(scorecard["total_return"]) > 0.0,
        "profit_factor_above_one": float(scorecard["profit_factor"]) > 1.0,
        "minimum_75_trades": int(scorecard["total_trades"]) >= 75,
        "max_drawdown_at_most_5pct": float(scorecard["realized_max_drawdown"]) <= 0.05,
        "positive_average_trade_alpha": alpha is not None and float(alpha) > 0.0,
        "positive_pnl_without_best_entry_year": (
            excluding_best is not None and float(excluding_best) > 0.0
        ),
    }
    passed = all(checks.values())
    return {
        "verdict": "passes_minimum_viability" if passed else "fails_minimum_viability",
        "passes": passed,
        "checks": checks,
        "interpretation": (
            "Historical viability survives the stricter replay; live forward evidence is still required."
            if passed
            else "The current V5 design does not clear the project's minimum historical viability bar under the stricter replay."
        ),
    }


def _resolve_market_inputs(
    runtime_dir: Path,
    *,
    market_db: str | Path | None,
    benchmark_security_id: str | None,
) -> tuple[Path, str]:
    if market_db is not None and benchmark_security_id is not None:
        return Path(market_db), benchmark_security_id.strip()

    path = runtime_dir / "paper_runtime.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid PAPER runtime config: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported PAPER runtime config schema")

    resolved_market_db = Path(market_db) if market_db is not None else Path(str(payload.get("market_db") or ""))
    resolved_benchmark = (
        benchmark_security_id.strip()
        if benchmark_security_id is not None
        else str(payload.get("benchmark_security_id") or "").strip()
    )
    if not str(resolved_market_db):
        raise ValueError("PAPER runtime config is missing market_db")
    if not resolved_benchmark:
        raise ValueError("PAPER runtime config is missing benchmark_security_id")
    return resolved_market_db, resolved_benchmark


def _load_legacy_comparison(root: Path) -> dict:
    path = root / "strategy_engine_v5_replay.json"
    if not path.exists():
        return {
            "available": False,
            "reason": "strategy_engine_v5_replay.json not found",
            "legacy_full_horizon_maturity_required": False,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid legacy V5 replay: {path}") from exc
    if payload.get("schema_version") != "strategy-engine-v5-exact-replay":
        raise ValueError("unexpected legacy V5 replay schema")
    observed = payload.get("observed")
    if not isinstance(observed, dict):
        raise ValueError("legacy V5 replay is missing observed scorecard")
    return {
        "available": True,
        "path": str(path),
        "legacy_full_horizon_maturity_required": False,
        "legacy_maturity_method": "stored_20d_exit_date_boundary",
        "observed": observed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain and replay exactly the current V5 PAPER strategy design under "
            "strict full-horizon maturity, execution realism, and the live 30% gross cap."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, default=Path("data/runtime"))
    parser.add_argument("--market-db", type=Path)
    parser.add_argument("--benchmark-security-id")
    parser.add_argument("--starting-capital", type=float, default=10_000.0)
    parser.add_argument("--allocation-pct", type=float, default=0.02)
    parser.add_argument("--max-open-positions", type=int, default=15)
    parser.add_argument("--max-gross-exposure-pct", type=float, default=0.30)
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    parser.add_argument("--max-trailing-adv-participation-pct", type=float, default=0.01)
    parser.add_argument("--max-entry-day-participation-pct", type=float, default=0.01)
    parser.add_argument(
        "--market-quality-manifest",
        type=Path,
        default=DEFAULT_MARKET_QUALITY_MANIFEST,
    )
    parser.add_argument("--min-train-rows", type=int, default=100)
    parser.add_argument("--market-read-cache-series", type=int, default=200)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_v5_profit_proof(
        args.experiment_dir,
        runtime_dir=args.runtime_dir,
        market_db=args.market_db,
        benchmark_security_id=args.benchmark_security_id,
        starting_capital=args.starting_capital,
        allocation_pct=args.allocation_pct,
        max_open_positions=args.max_open_positions,
        max_gross_exposure_pct=args.max_gross_exposure_pct,
        round_trip_cost_bps=args.round_trip_cost_bps,
        max_trailing_adv_participation_pct=args.max_trailing_adv_participation_pct,
        max_entry_day_participation_pct=args.max_entry_day_participation_pct,
        market_quality_manifest=args.market_quality_manifest,
        min_train_rows=args.min_train_rows,
        market_read_cache_series=args.market_read_cache_series,
    )
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "viability": payload["viability"],
                "proof_scope": payload["proof_scope"],
                "strategy_spec": payload["strategy_spec"],
                "portfolio_policy": payload["portfolio_policy"],
                "data": payload["data"],
                "execution_diagnostics": payload["execution_diagnostics"],
                "scorecard": payload["scorecard"],
                "entry_year_diagnostics": payload["entry_year_diagnostics"],
                "legacy_v5_comparison": payload["legacy_v5_comparison"],
                "output_path": str(result.output_path),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
