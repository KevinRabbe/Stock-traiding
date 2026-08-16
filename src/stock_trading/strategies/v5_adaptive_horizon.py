from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from stock_trading.engine import FeatureSnapshot, Opportunity, PortfolioSnapshot
from stock_trading.ml.lightgbm_models import (
    LightGbmModelBundle,
    ProfitLightGbmModelBundle,
)
from stock_trading.ml.online_calibration import RollingScoreHistory
from stock_trading.ml.score_calibration import static_score_percentiles


@dataclass(frozen=True, slots=True)
class V5HorizonModels:
    profit: ProfitLightGbmModelBundle
    alpha: LightGbmModelBundle


@dataclass(frozen=True, slots=True)
class V5StrategyConfig:
    strategy_id: str = "lightgbm-v5-adaptive-horizon"
    horizons: tuple[int, ...] = (5, 20, 60)
    validation_top_fraction: float = 0.05
    alpha_rank_weight: float = 0.25
    calibration_window_days: int = 365
    min_expected_return: float = 0.002
    max_expected_downside: float = 0.06

    def __post_init__(self) -> None:
        if not self.strategy_id.strip():
            raise ValueError("strategy_id must not be empty")
        if not self.horizons or any(value <= 0 for value in self.horizons):
            raise ValueError("horizons must contain positive session counts")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must be unique")
        if not 0.0 < self.validation_top_fraction < 1.0:
            raise ValueError("validation_top_fraction must be in (0, 1)")
        if not 0.0 <= self.alpha_rank_weight <= 1.0:
            raise ValueError("alpha_rank_weight must be in [0, 1]")
        if self.calibration_window_days <= 0:
            raise ValueError("calibration_window_days must be > 0")
        if self.max_expected_downside < 0:
            raise ValueError("max_expected_downside must be >= 0")

    @property
    def rank_threshold(self) -> float:
        return 1.0 - self.validation_top_fraction


@dataclass(slots=True)
class V5CalibrationState:
    profit_histories: dict[int, RollingScoreHistory]
    alpha_histories: dict[int, RollingScoreHistory]
    final_history: RollingScoreHistory

    @classmethod
    def from_validation(
        cls,
        candidates: tuple[FeatureSnapshot, ...],
        models: Mapping[int, V5HorizonModels],
        config: V5StrategyConfig,
    ) -> "V5CalibrationState":
        _validate_models(models, config)
        if not candidates:
            raise ValueError("validation candidates must not be empty")

        profit_histories = {
            horizon: RollingScoreHistory(window_days=config.calibration_window_days)
            for horizon in config.horizons
        }
        alpha_histories = {
            horizon: RollingScoreHistory(window_days=config.calibration_window_days)
            for horizon in config.horizons
        }
        signals: dict[int, tuple[float, ...]] = {}
        expected_returns: dict[int, tuple[float, ...]] = {}
        eligible: dict[int, tuple[bool, ...]] = {}

        for horizon in config.horizons:
            horizon_models = models[horizon]
            profit_predictions = tuple(
                horizon_models.profit.predict(dict(candidate.features))
                for candidate in candidates
            )
            alpha_predictions = tuple(
                horizon_models.alpha.predict(dict(candidate.features)).expected_alpha_20d
                for candidate in candidates
            )
            profit_scores = tuple(item.profit_score for item in profit_predictions)
            profit_percentiles = static_score_percentiles(profit_scores)
            alpha_percentiles = static_score_percentiles(alpha_predictions)
            signals[horizon] = tuple(
                (1.0 - config.alpha_rank_weight) * float(profit_percentiles[index])
                + config.alpha_rank_weight * float(alpha_percentiles[index])
                for index in range(len(candidates))
            )
            expected_returns[horizon] = tuple(
                item.expected_stock_return_20d for item in profit_predictions
            )
            eligible[horizon] = tuple(
                item.expected_stock_return_20d >= config.min_expected_return
                and item.expected_downside_20d <= config.max_expected_downside
                for item in profit_predictions
            )
            profit_histories[horizon].seed(
                (candidate.execution_date, score)
                for candidate, score in zip(candidates, profit_scores, strict=True)
            )
            alpha_histories[horizon].seed(
                (candidate.execution_date, score)
                for candidate, score in zip(candidates, alpha_predictions, strict=True)
            )

        final_history = RollingScoreHistory(window_days=config.calibration_window_days)
        final_seed: list[tuple] = []
        for index, candidate in enumerate(candidates):
            horizon = _choose_horizon(
                index,
                config.horizons,
                signals,
                expected_returns,
                eligible,
            )
            if horizon is not None:
                final_seed.append((candidate.execution_date, signals[horizon][index]))
        final_history.seed(final_seed)
        return cls(
            profit_histories=profit_histories,
            alpha_histories=alpha_histories,
            final_history=final_history,
        )


class V5AdaptiveHorizonStrategy:
    """Production/replay adapter for the saved V5 opportunity logic.

    The strategy consumes generic PIT ``FeatureSnapshot`` objects, predicts all
    configured holding horizons, performs the same two-stage rolling calibration
    used by V5, chooses the strongest cost/downside-eligible horizon, and emits
    only final-rank-qualified opportunities. It has no portfolio or broker authority.
    """

    def __init__(
        self,
        models: Mapping[int, V5HorizonModels],
        calibration: V5CalibrationState,
        config: V5StrategyConfig | None = None,
    ) -> None:
        self.config = config or V5StrategyConfig()
        _validate_models(models, self.config)
        self.models = dict(models)
        self.calibration = calibration

    @property
    def strategy_id(self) -> str:
        return self.config.strategy_id

    def evaluate(
        self,
        candidates: tuple[FeatureSnapshot, ...],
        portfolio: PortfolioSnapshot,
    ) -> tuple[Opportunity, ...]:
        del portfolio
        if not candidates:
            return ()

        by_date: dict[object, list[FeatureSnapshot]] = defaultdict(list)
        for candidate in candidates:
            by_date[candidate.execution_date].append(candidate)

        opportunities: list[Opportunity] = []
        for execution_date in sorted(by_date):
            batch = tuple(by_date[execution_date])
            batch_size = len(batch)
            signals: dict[int, tuple[float, ...]] = {}
            expected_returns: dict[int, tuple[float, ...]] = {}
            expected_downsides: dict[int, tuple[float, ...]] = {}
            probabilities: dict[int, tuple[float, ...]] = {}
            alpha_predictions: dict[int, tuple[float, ...]] = {}
            profit_scores: dict[int, tuple[float, ...]] = {}
            eligible: dict[int, tuple[bool, ...]] = {}

            for horizon in self.config.horizons:
                horizon_models = self.models[horizon]
                profit_prediction = tuple(
                    horizon_models.profit.predict(dict(candidate.features))
                    for candidate in batch
                )
                alpha_prediction = tuple(
                    horizon_models.alpha.predict(dict(candidate.features)).expected_alpha_20d
                    for candidate in batch
                )
                raw_profit_scores = tuple(item.profit_score for item in profit_prediction)
                profit_percentiles = self.calibration.profit_histories[horizon].percentiles(
                    execution_date,
                    raw_profit_scores,
                    ineligible_percentile=0.5,
                )
                alpha_percentiles = self.calibration.alpha_histories[horizon].percentiles(
                    execution_date,
                    alpha_prediction,
                    ineligible_percentile=0.5,
                )
                signals[horizon] = tuple(
                    (1.0 - self.config.alpha_rank_weight) * profit_percentiles[index]
                    + self.config.alpha_rank_weight * alpha_percentiles[index]
                    for index in range(batch_size)
                )
                expected_returns[horizon] = tuple(
                    item.expected_stock_return_20d for item in profit_prediction
                )
                expected_downsides[horizon] = tuple(
                    item.expected_downside_20d for item in profit_prediction
                )
                probabilities[horizon] = tuple(
                    item.probability_profitable_return for item in profit_prediction
                )
                alpha_predictions[horizon] = alpha_prediction
                profit_scores[horizon] = raw_profit_scores
                eligible[horizon] = tuple(
                    expected_returns[horizon][index] >= self.config.min_expected_return
                    and expected_downsides[horizon][index] <= self.config.max_expected_downside
                    for index in range(batch_size)
                )

            chosen: list[int | None] = []
            chosen_signals: list[float] = []
            any_eligible: list[bool] = []
            for index in range(batch_size):
                horizon = _choose_horizon(
                    index,
                    self.config.horizons,
                    signals,
                    expected_returns,
                    eligible,
                )
                chosen.append(horizon)
                any_eligible.append(horizon is not None)
                chosen_signals.append(signals[horizon][index] if horizon is not None else 0.0)

            final_percentiles = self.calibration.final_history.percentiles(
                execution_date,
                chosen_signals,
                eligible=any_eligible,
                ineligible_percentile=0.0,
            )

            for index, candidate in enumerate(batch):
                horizon = chosen[index]
                if horizon is None or final_percentiles[index] < self.config.rank_threshold:
                    continue
                opportunities.append(
                    Opportunity(
                        strategy_id=self.strategy_id,
                        candidate_id=candidate.candidate_id,
                        event_id=candidate.event_id,
                        company_id=candidate.company_id,
                        security_id=candidate.security_id,
                        execution_date=candidate.execution_date,
                        score=final_percentiles[index],
                        expected_return=expected_returns[horizon][index],
                        expected_alpha=alpha_predictions[horizon][index],
                        expected_downside=expected_downsides[horizon][index],
                        probability_positive=probabilities[horizon][index],
                        horizon_sessions=horizon,
                        metadata={
                            "raw_profit_score": profit_scores[horizon][index],
                            "combined_signal_before_final_calibration": chosen_signals[index],
                            "final_percentile": final_percentiles[index],
                        },
                    )
                )
        return tuple(opportunities)


def load_v5_horizon_models(
    models_root: str | Path,
    *,
    model_year: int,
    horizons: tuple[int, ...] = (5, 20, 60),
) -> dict[int, V5HorizonModels]:
    root = Path(models_root) / str(model_year)
    return {
        horizon: V5HorizonModels(
            profit=ProfitLightGbmModelBundle.load(root / f"{horizon}d" / "profit"),
            alpha=LightGbmModelBundle.load(root / f"{horizon}d" / "alpha"),
        )
        for horizon in horizons
    }


def build_v5_strategy_from_saved_models(
    models_root: str | Path,
    *,
    model_year: int,
    validation_candidates: tuple[FeatureSnapshot, ...],
    config: V5StrategyConfig | None = None,
) -> V5AdaptiveHorizonStrategy:
    resolved = config or V5StrategyConfig()
    models = load_v5_horizon_models(
        models_root,
        model_year=model_year,
        horizons=resolved.horizons,
    )
    calibration = V5CalibrationState.from_validation(
        validation_candidates,
        models,
        resolved,
    )
    return V5AdaptiveHorizonStrategy(models, calibration, resolved)


def _choose_horizon(
    index: int,
    horizons: tuple[int, ...],
    signals: Mapping[int, tuple[float, ...]],
    expected_returns: Mapping[int, tuple[float, ...]],
    eligible: Mapping[int, tuple[bool, ...]],
) -> int | None:
    candidates = [horizon for horizon in sorted(horizons) if eligible[horizon][index]]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda horizon: (
            signals[horizon][index],
            expected_returns[horizon][index],
            -horizon,
        ),
    )


def _validate_models(
    models: Mapping[int, V5HorizonModels],
    config: V5StrategyConfig,
) -> None:
    if set(models) != set(config.horizons):
        raise ValueError("model horizons do not match V5 strategy config")
