from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

from stock_trading.engine import FeatureSnapshot, StrategyRegistry, StrategyStage
from stock_trading.ml.online_calibration import RollingScoreHistory
from stock_trading.strategies.v5_adaptive_horizon import (
    V5AdaptiveHorizonStrategy,
    V5CalibrationState,
)


@dataclass(frozen=True, slots=True)
class HorizonDecisionDiagnostic:
    horizon_sessions: int
    expected_return: float
    expected_alpha: float
    expected_downside: float
    probability_positive: float
    raw_profit_score: float
    profit_percentile: float
    alpha_percentile: float
    combined_signal: float
    eligible: bool
    eligibility_reasons: tuple[str, ...]
    required_feature_count: int
    missing_feature_count: int
    missing_feature_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CandidateDecisionDiagnostic:
    candidate_id: str
    company_id: str
    security_id: str
    execution_date: date
    chosen_horizon: int | None
    final_percentile: float
    rank_threshold: float
    emitted: bool
    rejection_reason: str
    horizons: tuple[HorizonDecisionDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class StrategyDecisionDiagnostics:
    strategy_id: str
    candidate_count: int
    emitted_opportunity_count: int
    decisions: tuple[CandidateDecisionDiagnostic, ...]

    @property
    def emitted_candidate_ids(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self.decisions if item.emitted)


def diagnose_strategy(
    strategy: V5AdaptiveHorizonStrategy,
    candidates: tuple[FeatureSnapshot, ...],
) -> StrategyDecisionDiagnostics:
    """Mirror one V5-style strategy decision without mutating calibration state.

    Current runtime strategies all use ``V5AdaptiveHorizonStrategy`` regardless of
    whether they are the legacy V5 PAPER champion or frozen factory challengers.
    The diagnostic deliberately calls rolling percentile calibration with
    ``update=False`` and verifies the complete calibration snapshot is unchanged.
    """

    if not isinstance(strategy, V5AdaptiveHorizonStrategy):
        raise TypeError(
            f"decision diagnostics are not implemented for {type(strategy).__name__}"
        )
    if not candidates:
        return StrategyDecisionDiagnostics(strategy.strategy_id, 0, 0, ())

    execution_dates = {item.execution_date for item in candidates}
    if len(execution_dates) != 1:
        raise ValueError("current decision diagnostics require one execution session")
    execution_date = next(iter(execution_dates))
    before = _calibration_snapshot(strategy.calibration)

    batch = tuple(candidates)
    batch_size = len(batch)
    signals: dict[int, tuple[float, ...]] = {}
    expected_returns: dict[int, tuple[float, ...]] = {}
    expected_alphas: dict[int, tuple[float, ...]] = {}
    expected_downsides: dict[int, tuple[float, ...]] = {}
    probabilities: dict[int, tuple[float, ...]] = {}
    profit_scores: dict[int, tuple[float, ...]] = {}
    profit_percentiles: dict[int, tuple[float, ...]] = {}
    alpha_percentiles: dict[int, tuple[float, ...]] = {}
    eligible: dict[int, tuple[bool, ...]] = {}
    missing_features: dict[int, tuple[tuple[str, ...], ...]] = {}
    required_feature_counts: dict[int, int] = {}

    for horizon in strategy.config.horizons:
        horizon_models = strategy.models[horizon]
        profit_predictions = tuple(
            horizon_models.profit.predict(dict(candidate.features))
            for candidate in batch
        )
        alpha_predictions = tuple(
            horizon_models.alpha.predict(dict(candidate.features)).expected_alpha_20d
            for candidate in batch
        )
        raw_profit_scores = tuple(item.profit_score for item in profit_predictions)
        raw_profit_percentiles = strategy.calibration.profit_histories[horizon].percentiles(
            execution_date,
            raw_profit_scores,
            ineligible_percentile=0.5,
            update=False,
        )
        raw_alpha_percentiles = strategy.calibration.alpha_histories[horizon].percentiles(
            execution_date,
            alpha_predictions,
            ineligible_percentile=0.5,
            update=False,
        )
        signals[horizon] = tuple(
            (1.0 - strategy.config.alpha_rank_weight) * raw_profit_percentiles[index]
            + strategy.config.alpha_rank_weight * raw_alpha_percentiles[index]
            for index in range(batch_size)
        )
        expected_returns[horizon] = tuple(
            item.expected_stock_return_20d for item in profit_predictions
        )
        expected_alphas[horizon] = alpha_predictions
        expected_downsides[horizon] = tuple(
            item.expected_downside_20d for item in profit_predictions
        )
        probabilities[horizon] = tuple(
            item.probability_profitable_return for item in profit_predictions
        )
        profit_scores[horizon] = raw_profit_scores
        profit_percentiles[horizon] = raw_profit_percentiles
        alpha_percentiles[horizon] = raw_alpha_percentiles
        eligible[horizon] = tuple(
            expected_returns[horizon][index] >= strategy.config.min_expected_return
            and expected_downsides[horizon][index] <= strategy.config.max_expected_downside
            for index in range(batch_size)
        )

        required = tuple(
            sorted(
                set(horizon_models.profit.feature_schema.names)
                | set(horizon_models.alpha.feature_schema.names)
            )
        )
        required_feature_counts[horizon] = len(required)
        missing_features[horizon] = tuple(
            tuple(
                name
                for name in required
                if name not in candidate.features or candidate.features[name] is None
            )
            for candidate in batch
        )

    chosen: list[int | None] = []
    chosen_signals: list[float] = []
    any_eligible: list[bool] = []
    for index in range(batch_size):
        horizons = [
            horizon
            for horizon in sorted(strategy.config.horizons)
            if eligible[horizon][index]
        ]
        horizon = (
            max(
                horizons,
                key=lambda value: (
                    signals[value][index],
                    expected_returns[value][index],
                    -value,
                ),
            )
            if horizons
            else None
        )
        chosen.append(horizon)
        any_eligible.append(horizon is not None)
        chosen_signals.append(signals[horizon][index] if horizon is not None else 0.0)

    final_percentiles = strategy.calibration.final_history.percentiles(
        execution_date,
        chosen_signals,
        eligible=any_eligible,
        ineligible_percentile=0.0,
        update=False,
    )

    decisions: list[CandidateDecisionDiagnostic] = []
    for index, candidate in enumerate(batch):
        horizon_diagnostics: list[HorizonDecisionDiagnostic] = []
        for horizon in strategy.config.horizons:
            reasons: list[str] = []
            if expected_returns[horizon][index] < strategy.config.min_expected_return:
                reasons.append("expected_return_below_minimum")
            if expected_downsides[horizon][index] > strategy.config.max_expected_downside:
                reasons.append("expected_downside_above_maximum")
            missing = missing_features[horizon][index]
            horizon_diagnostics.append(
                HorizonDecisionDiagnostic(
                    horizon_sessions=horizon,
                    expected_return=expected_returns[horizon][index],
                    expected_alpha=expected_alphas[horizon][index],
                    expected_downside=expected_downsides[horizon][index],
                    probability_positive=probabilities[horizon][index],
                    raw_profit_score=profit_scores[horizon][index],
                    profit_percentile=profit_percentiles[horizon][index],
                    alpha_percentile=alpha_percentiles[horizon][index],
                    combined_signal=signals[horizon][index],
                    eligible=eligible[horizon][index],
                    eligibility_reasons=tuple(reasons),
                    required_feature_count=required_feature_counts[horizon],
                    missing_feature_count=len(missing),
                    missing_feature_names=missing,
                )
            )

        selected_horizon = chosen[index]
        final_percentile = final_percentiles[index]
        emitted = (
            selected_horizon is not None
            and final_percentile >= strategy.config.rank_threshold
        )
        if selected_horizon is None:
            rejection_reason = "no_eligible_horizon"
        elif not emitted:
            rejection_reason = "below_final_rank_threshold"
        else:
            rejection_reason = "emitted"
        decisions.append(
            CandidateDecisionDiagnostic(
                candidate_id=candidate.candidate_id,
                company_id=candidate.company_id,
                security_id=candidate.security_id,
                execution_date=candidate.execution_date,
                chosen_horizon=selected_horizon,
                final_percentile=final_percentile,
                rank_threshold=strategy.config.rank_threshold,
                emitted=emitted,
                rejection_reason=rejection_reason,
                horizons=tuple(horizon_diagnostics),
            )
        )

    after = _calibration_snapshot(strategy.calibration)
    if after != before:
        raise RuntimeError("read-only strategy diagnostics mutated calibration state")
    resolved = tuple(decisions)
    return StrategyDecisionDiagnostics(
        strategy_id=strategy.strategy_id,
        candidate_count=len(batch),
        emitted_opportunity_count=sum(1 for item in resolved if item.emitted),
        decisions=resolved,
    )


def diagnose_registry(
    registry: StrategyRegistry,
    candidates: tuple[FeatureSnapshot, ...],
) -> tuple[StrategyDecisionDiagnostics, ...]:
    strategies = [registry.active()]
    strategies.extend(
        registry.loaded_challenger_strategies(stages=(StrategyStage.SHADOW,))
    )
    return tuple(
        diagnose_strategy(strategy, candidates)  # type: ignore[arg-type]
        for strategy in strategies
    )


def rewind_registry_calibration_before(
    registry: StrategyRegistry,
    execution_date: date,
) -> None:
    """Rewind loaded mutable overlays to the state visible before one session.

    ``RollingScoreHistory`` itself ignores same-date observations when ranking a
    batch, so retaining only entries strictly before ``execution_date`` exactly
    reconstructs the reference population for a recent completed receipt. This is
    intended for read-only diagnosis only; callers must never save the rewound
    strategies back into runtime state.
    """

    strategies = [registry.active()]
    strategies.extend(
        registry.loaded_challenger_strategies(stages=(StrategyStage.SHADOW,))
    )
    for strategy in strategies:
        if not isinstance(strategy, V5AdaptiveHorizonStrategy):
            raise TypeError(
                f"calibration rewind is not implemented for {type(strategy).__name__}"
            )
        strategy.calibration = V5CalibrationState(
            profit_histories={
                horizon: _history_before(
                    strategy.calibration.profit_histories[horizon],
                    execution_date,
                    strategy.config.calibration_window_days,
                )
                for horizon in strategy.config.horizons
            },
            alpha_histories={
                horizon: _history_before(
                    strategy.calibration.alpha_histories[horizon],
                    execution_date,
                    strategy.config.calibration_window_days,
                )
                for horizon in strategy.config.horizons
            },
            final_history=_history_before(
                strategy.calibration.final_history,
                execution_date,
                strategy.config.calibration_window_days,
            ),
        )


class FileStrategyDecisionDiagnosticStore:
    """Immutable-style one-file audit for the prediction/gating decisions of a batch."""

    SCHEMA_VERSION = 1

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def write(
        self,
        *,
        batch_id: str,
        as_of: datetime,
        target_execution_date: date,
        diagnostics: tuple[StrategyDecisionDiagnostics, ...],
    ) -> Path:
        if not batch_id.startswith("batch_"):
            raise ValueError("invalid diagnostic batch_id")
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "batch_id": batch_id,
            "as_of": as_of.isoformat(),
            "target_execution_date": target_execution_date.isoformat(),
            "strategies": diagnostics_payload(diagnostics),
        }
        path = self.root / f"{batch_id}.json"
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid strategy decision diagnostic: {path}") from exc
            if existing != payload:
                raise ValueError(f"strategy decision diagnostic changed for {batch_id}")
            return path
        _atomic_json_write(path, payload)
        return path


def diagnostics_payload(
    diagnostics: Iterable[StrategyDecisionDiagnostics],
) -> list[dict]:
    return [_json_compatible(asdict(strategy)) for strategy in diagnostics]


def _json_compatible(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    return value


def validate_diagnostic_counts(
    diagnostics: tuple[StrategyDecisionDiagnostics, ...],
    *,
    champion_strategy_id: str,
    champion_opportunity_count: int,
    shadow_opportunity_counts: dict[str, int],
) -> None:
    actual = {champion_strategy_id: champion_opportunity_count, **shadow_opportunity_counts}
    expected = {item.strategy_id: item.emitted_opportunity_count for item in diagnostics}
    if expected != actual:
        raise RuntimeError(
            "read-only decision diagnostics disagree with actual strategy opportunity counts: "
            f"expected={expected} actual={actual}"
        )


def _calibration_snapshot(state: V5CalibrationState) -> tuple:
    return (
        tuple(
            (horizon, state.profit_histories[horizon].snapshot())
            for horizon in sorted(state.profit_histories)
        ),
        tuple(
            (horizon, state.alpha_histories[horizon].snapshot())
            for horizon in sorted(state.alpha_histories)
        ),
        state.final_history.snapshot(),
    )


def _history_before(
    history: RollingScoreHistory,
    execution_date: date,
    window_days: int,
) -> RollingScoreHistory:
    result = RollingScoreHistory(window_days=window_days)
    result.seed((day, score) for day, score in history.snapshot() if day < execution_date)
    return result


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
