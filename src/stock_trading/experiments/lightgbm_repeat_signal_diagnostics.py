from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stock_trading.backtest import (
    BacktestConfig,
    FixedAllocationBacktester,
    FixedAllocationTrancheBacktester,
    summarize_walk_forward,
)
from stock_trading.backtest.portfolio import BacktestResult, ScoredCandidate
from stock_trading.ml import ProfitLightGbmModelBundle, TrainingRow
from stock_trading.ml.walk_forward import WalkForwardResult, annual_walk_forward_splits

from .lightgbm_diagnostics import _load_training_rows
from .lightgbm_profit import _average, _predict_profit_matrix
from .lightgbm_profit_tranches import _selected_candidates
from .lightgbm_validation_rank import _json_safe


@dataclass(frozen=True, slots=True)
class RepeatSignalDiagnosticsResult:
    training_row_count: int
    model_years: tuple[int, ...]
    output_path: Path


@dataclass(frozen=True, slots=True)
class _SignalObservation:
    candidate: ScoredCandidate
    overlap_ordinal: int
    days_since_previous_signal: int | None
    stock_return_delta_vs_previous: float | None
    alpha_delta_vs_previous: float | None
    score_delta_vs_previous: float | None


@dataclass(frozen=True, slots=True)
class _AcceptanceTrace:
    accepted_event_ids: tuple[str, ...]
    rejection_reason_by_event_id: dict[str, str]


def run_repeat_signal_diagnostics(
    experiment_dir: str | Path,
    *,
    validation_top_fraction: float = 0.05,
    max_company_tranches: int = 3,
    max_expected_downside: float = 0.06,
    starting_capital: float = 10_000.0,
    allocation_pct: float = 0.02,
    max_open_positions: int = 15,
    round_trip_cost_bps: float = 20.0,
) -> RepeatSignalDiagnosticsResult:
    """Diagnose repeated same-company signals without tuning the trading policy.

    Saved annual profit models are replayed with the same validation-only score
    threshold used by the profit-target experiment. Signal quality is measured
    before portfolio capacity is applied, then the accepted trade sets from the
    one-position and bounded-tranche portfolio mechanics are compared.

    ``overlap_ordinal`` is the number of simultaneously active selected signals
    for that company including the current signal. It resets to one once all
    prior selected 20-day opportunities for that company have exited.
    """

    if not 0 < validation_top_fraction < 1:
        raise ValueError("validation_top_fraction must be in (0, 1)")
    if max_company_tranches <= 1:
        raise ValueError("max_company_tranches must be > 1 for repeat diagnostics")
    if max_expected_downside < 0:
        raise ValueError("max_expected_downside must be >= 0")
    if max_open_positions <= 0:
        raise ValueError("max_open_positions must be > 0")

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
    single_backtester = FixedAllocationBacktester(config)
    tranche_backtester = FixedAllocationTrancheBacktester(
        config,
        max_company_tranches=max_company_tranches,
    )

    all_signal_observations: list[_SignalObservation] = []
    all_added_candidates: list[ScoredCandidate] = []
    all_capacity_displaced: list[ScoredCandidate] = []
    all_company_limit_displaced: list[ScoredCandidate] = []
    single_results: list[WalkForwardResult] = []
    tranche_results: list[WalkForwardResult] = []
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
        eligible = tuple(
            candidate
            for candidate in selected
            if candidate.prediction.expected_downside_20d <= max_expected_downside
        )

        observations = _classify_overlap_signals(eligible)
        all_signal_observations.extend(observations)

        single_portfolio = single_backtester.run(selected)
        tranche_portfolio = tranche_backtester.run(selected)
        single_results.append(
            WalkForwardResult(
                test_year=year,
                train_count=len(split.train_rows),
                validation_count=len(split.validation_rows),
                test_count=len(split.test_rows),
                backtest=single_portfolio,
            )
        )
        tranche_results.append(
            WalkForwardResult(
                test_year=year,
                train_count=len(split.train_rows),
                validation_count=len(split.validation_rows),
                test_count=len(split.test_rows),
                backtest=tranche_portfolio,
            )
        )

        single_trace = _trace_acceptance(
            eligible,
            max_open_positions=max_open_positions,
            max_company_tranches=1,
        )
        tranche_trace = _trace_acceptance(
            eligible,
            max_open_positions=max_open_positions,
            max_company_tranches=max_company_tranches,
        )
        _assert_trace_matches_portfolio(single_trace, single_portfolio, "single")
        _assert_trace_matches_portfolio(tranche_trace, tranche_portfolio, "tranche")

        candidate_by_event_id = {candidate.row.event_id: candidate for candidate in eligible}
        single_ids = set(single_trace.accepted_event_ids)
        tranche_ids = set(tranche_trace.accepted_event_ids)
        added_ids = tranche_ids - single_ids
        lost_ids = single_ids - tranche_ids
        added = [candidate_by_event_id[event_id] for event_id in sorted(added_ids)]
        capacity_displaced = [
            candidate_by_event_id[event_id]
            for event_id in sorted(lost_ids)
            if tranche_trace.rejection_reason_by_event_id.get(event_id) == "capacity"
        ]
        company_limit_displaced = [
            candidate_by_event_id[event_id]
            for event_id in sorted(lost_ids)
            if tranche_trace.rejection_reason_by_event_id.get(event_id) == "company_limit"
        ]
        all_added_candidates.extend(added)
        all_capacity_displaced.extend(capacity_displaced)
        all_company_limit_displaced.extend(company_limit_displaced)

        year_reports.append(
            {
                "year": year,
                "validation_score_threshold": score_threshold,
                "test_selected": len(selected),
                "eligible_after_downside_gate": len(eligible),
                "signal_quality_by_overlap_ordinal": _ordinal_summaries(observations),
                "repeat_signal_quality_by_gap_days": _gap_summaries(observations),
                "single_position": _portfolio_summary(single_portfolio),
                "bounded_tranches": _portfolio_summary(tranche_portfolio),
                "portfolio_set_difference": {
                    "added_by_tranches": _candidate_summary(added),
                    "single_trades_displaced_by_tranche_capacity": _candidate_summary(
                        capacity_displaced
                    ),
                    "single_trades_lost_to_tranche_company_limit": _candidate_summary(
                        company_limit_displaced
                    ),
                },
            }
        )

    payload = _json_safe(
        {
            "schema_version": "repeat-signal-diagnostics-v1",
            "experiment_dir": str(root),
            "training_row_count": len(rows),
            "model_years": list(model_years),
            "policy_replayed": {
                "target": "absolute_stock_return_after_costs",
                "score_cutoff_source": "preceding validation year only",
                "validation_top_fraction": validation_top_fraction,
                "max_expected_downside": max_expected_downside,
                "single_position_company_limit": 1,
                "tranche_company_limit": max_company_tranches,
                "allocation_pct_per_slot": allocation_pct,
                "max_open_slots": max_open_positions,
                "round_trip_cost_bps": round_trip_cost_bps,
                "starting_capital": starting_capital,
            },
            "definitions": {
                "overlap_ordinal": (
                    "count of eligible selected same-company 20-day signals still active at "
                    "the current entry, including the current signal; resets to 1 after prior "
                    "selected signals exit"
                ),
                "added_by_tranches": (
                    "trade accepted by bounded-tranche mechanics but not by one-position mechanics"
                ),
                "single_trades_displaced_by_tranche_capacity": (
                    "trade accepted by one-position mechanics but rejected by bounded-tranche "
                    "mechanics because earlier extra tranches filled the portfolio slot cap"
                ),
            },
            "aggregate": {
                "signal_quality_by_overlap_ordinal": _ordinal_summaries(
                    all_signal_observations
                ),
                "repeat_signal_quality_by_gap_days": _gap_summaries(
                    all_signal_observations
                ),
                "single_position_portfolio": _walk_forward_summary(single_results),
                "bounded_tranche_portfolio": _walk_forward_summary(tranche_results),
                "portfolio_set_difference": {
                    "added_by_tranches": _candidate_summary(all_added_candidates),
                    "single_trades_displaced_by_tranche_capacity": _candidate_summary(
                        all_capacity_displaced
                    ),
                    "single_trades_lost_to_tranche_company_limit": _candidate_summary(
                        all_company_limit_displaced
                    ),
                },
            },
            "years": year_reports,
        }
    )
    output_path = root / "repeat_signal_diagnostics.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return RepeatSignalDiagnosticsResult(
        training_row_count=len(rows),
        model_years=model_years,
        output_path=output_path,
    )


def _classify_overlap_signals(
    candidates: tuple[ScoredCandidate, ...] | list[ScoredCandidate],
) -> list[_SignalObservation]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.row.execution_date,
            -candidate.prediction.opportunity_score,
            candidate.row.event_id,
        ),
    )
    active_exit_dates: dict[str, list] = {}
    previous: dict[str, ScoredCandidate] = {}
    observations: list[_SignalObservation] = []

    for candidate in ordered:
        row = candidate.row
        active = [
            exit_date
            for exit_date in active_exit_dates.get(row.company_id, [])
            if exit_date > row.execution_date
        ]
        prior = previous.get(row.company_id)
        ordinal = len(active) + 1
        is_repeat = ordinal > 1 and prior is not None
        observations.append(
            _SignalObservation(
                candidate=candidate,
                overlap_ordinal=ordinal,
                days_since_previous_signal=(
                    (row.execution_date - prior.row.execution_date).days
                    if is_repeat
                    else None
                ),
                stock_return_delta_vs_previous=(
                    row.stock_return_20d - prior.row.stock_return_20d
                    if is_repeat
                    else None
                ),
                alpha_delta_vs_previous=(
                    row.alpha_20d - prior.row.alpha_20d if is_repeat else None
                ),
                score_delta_vs_previous=(
                    candidate.prediction.opportunity_score
                    - prior.prediction.opportunity_score
                    if is_repeat
                    else None
                ),
            )
        )
        active.append(row.exit_date_20d)
        active_exit_dates[row.company_id] = active
        previous[row.company_id] = candidate

    return observations


def _trace_acceptance(
    candidates: tuple[ScoredCandidate, ...] | list[ScoredCandidate],
    *,
    max_open_positions: int,
    max_company_tranches: int,
) -> _AcceptanceTrace:
    by_entry: dict = {}
    for candidate in candidates:
        by_entry.setdefault(candidate.row.execution_date, []).append(candidate)

    open_candidates: list[ScoredCandidate] = []
    accepted: list[str] = []
    rejected: dict[str, str] = {}
    for current_date in sorted(by_entry):
        open_candidates = [
            candidate
            for candidate in open_candidates
            if candidate.row.exit_date_20d > current_date
        ]
        company_counts: dict[str, int] = {}
        for candidate in open_candidates:
            company_id = candidate.row.company_id
            company_counts[company_id] = company_counts.get(company_id, 0) + 1

        daily = sorted(
            by_entry[current_date],
            key=lambda candidate: (
                -candidate.prediction.opportunity_score,
                candidate.row.event_id,
            ),
        )
        for candidate in daily:
            company_id = candidate.row.company_id
            if company_counts.get(company_id, 0) >= max_company_tranches:
                rejected[candidate.row.event_id] = "company_limit"
                continue
            if len(open_candidates) >= max_open_positions:
                rejected[candidate.row.event_id] = "capacity"
                continue
            open_candidates.append(candidate)
            company_counts[company_id] = company_counts.get(company_id, 0) + 1
            accepted.append(candidate.row.event_id)

    return _AcceptanceTrace(tuple(accepted), rejected)


def _assert_trace_matches_portfolio(
    trace: _AcceptanceTrace,
    portfolio: BacktestResult,
    label: str,
) -> None:
    traced = set(trace.accepted_event_ids)
    actual = {trade.event_id for trade in portfolio.trades}
    if traced != actual:
        raise RuntimeError(
            f"{label} diagnostic acceptance trace diverged from portfolio backtester"
        )


def _ordinal_summaries(observations: list[_SignalObservation]) -> dict:
    buckets = {
        "first_active_signal": [item for item in observations if item.overlap_ordinal == 1],
        "second_overlapping_signal": [
            item for item in observations if item.overlap_ordinal == 2
        ],
        "third_or_later_overlapping_signal": [
            item for item in observations if item.overlap_ordinal >= 3
        ],
    }
    return {name: _signal_summary(items) for name, items in buckets.items()}


def _gap_summaries(observations: list[_SignalObservation]) -> dict:
    repeats = [item for item in observations if item.days_since_previous_signal is not None]
    buckets = {
        "0_to_7_days": [
            item for item in repeats if item.days_since_previous_signal <= 7
        ],
        "8_to_14_days": [
            item for item in repeats if 8 <= item.days_since_previous_signal <= 14
        ],
        "15_to_21_days": [
            item for item in repeats if 15 <= item.days_since_previous_signal <= 21
        ],
        "22_plus_days": [
            item for item in repeats if item.days_since_previous_signal >= 22
        ],
    }
    return {name: _signal_summary(items) for name, items in buckets.items()}


def _signal_summary(observations: list[_SignalObservation]) -> dict:
    candidates = [item.candidate for item in observations]
    gaps = [
        float(item.days_since_previous_signal)
        for item in observations
        if item.days_since_previous_signal is not None
    ]
    return {
        **_candidate_summary(candidates),
        "average_days_since_previous_signal": _average(gaps),
        "median_days_since_previous_signal": _quantile(gaps, 0.50),
        "average_incremental_stock_return_vs_previous_signal": _average(
            [
                item.stock_return_delta_vs_previous
                for item in observations
                if item.stock_return_delta_vs_previous is not None
            ]
        ),
        "average_incremental_alpha_vs_previous_signal": _average(
            [
                item.alpha_delta_vs_previous
                for item in observations
                if item.alpha_delta_vs_previous is not None
            ]
        ),
        "average_score_change_vs_previous_signal": _average(
            [
                item.score_delta_vs_previous
                for item in observations
                if item.score_delta_vs_previous is not None
            ]
        ),
    }


def _candidate_summary(candidates: list[ScoredCandidate]) -> dict:
    return {
        "count": len(candidates),
        "average_stock_return_20d": _average(
            [candidate.row.stock_return_20d for candidate in candidates]
        ),
        "average_alpha_20d": _average([candidate.row.alpha_20d for candidate in candidates]),
        "average_opportunity_score": _average(
            [candidate.prediction.opportunity_score for candidate in candidates]
        ),
        "average_expected_downside_20d": _average(
            [candidate.prediction.expected_downside_20d for candidate in candidates]
        ),
        "profitable_after_cost_rate": (
            sum(candidate.row.stock_return_20d > 0.002 for candidate in candidates)
            / len(candidates)
            if candidates
            else None
        ),
    }


def _portfolio_summary(portfolio: BacktestResult) -> dict:
    return {
        "trades": len(portfolio.trades),
        "return": portfolio.total_return,
        "profit_factor": portfolio.profit_factor,
        "average_trade_stock_return": _average(
            [trade.gross_return for trade in portfolio.trades]
        ),
        "average_trade_alpha": _average([trade.alpha_20d for trade in portfolio.trades]),
        "realized_drawdown": portfolio.realized_max_drawdown,
        "rejected_company_limit": portfolio.rejected_duplicate_company,
        "rejected_capacity": portfolio.rejected_capacity,
    }


def _walk_forward_summary(results: list[WalkForwardResult]) -> dict:
    summary = summarize_walk_forward(results)
    trades = [trade for result in results for trade in result.backtest.trades]
    return {
        "compounded_return": summary.compounded_return,
        "profitable_year_rate": summary.profitable_year_rate,
        "total_trades": summary.total_trades,
        "average_trade_stock_return": _average([trade.gross_return for trade in trades]),
        "average_trade_alpha": summary.average_trade_alpha,
        "aggregate_profit_factor": summary.aggregate_profit_factor,
        "worst_realized_drawdown": summary.worst_realized_drawdown,
    }


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose repeated same-company profit-model signals and tranche capacity "
            "displacement without changing or retraining the saved policy."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--validation-top-fraction", type=float, default=0.05)
    parser.add_argument("--max-company-tranches", type=int, default=3)
    parser.add_argument("--max-expected-downside", type=float, default=0.06)
    parser.add_argument("--starting-capital", type=float, default=10_000.0)
    parser.add_argument("--allocation-pct", type=float, default=0.02)
    parser.add_argument("--max-open-positions", type=int, default=15)
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_repeat_signal_diagnostics(
        args.experiment_dir,
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
                "policy_replayed": payload["policy_replayed"],
                "aggregate": payload["aggregate"],
                "years": payload["years"],
                "output_path": str(result.output_path),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
