from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from stock_trading.engine import FeatureSnapshot, PortfolioSnapshot
from stock_trading.strategies.v5_adaptive_horizon import (
    V5AdaptiveHorizonStrategy,
    V5CalibrationState,
    V5HorizonModels,
    V5StrategyConfig,
)


class _ProfitModel:
    def __init__(self, horizon: int):
        self.horizon = horizon

    def predict(self, features):
        horizon = self.horizon
        return SimpleNamespace(
            expected_stock_return_20d=float(features[f"return_{horizon}"]),
            expected_downside_20d=float(features.get(f"downside_{horizon}", 0.01)),
            probability_profitable_return=float(features.get(f"probability_{horizon}", 0.8)),
            profit_score=float(features[f"profit_score_{horizon}"]),
        )


class _AlphaModel:
    def __init__(self, horizon: int):
        self.horizon = horizon

    def predict(self, features):
        return SimpleNamespace(
            expected_alpha_20d=float(features[f"alpha_{self.horizon}"])
        )


def _models():
    return {
        horizon: V5HorizonModels(
            profit=_ProfitModel(horizon),
            alpha=_AlphaModel(horizon),
        )
        for horizon in (5, 20, 60)
    }


def _candidate(candidate_id: str, execution_date: date, values: dict) -> FeatureSnapshot:
    return FeatureSnapshot(
        candidate_id=candidate_id,
        event_id=f"evt-{candidate_id}",
        company_id=f"company-{candidate_id}",
        security_id=f"security-{candidate_id}",
        decision_time=datetime.combine(
            execution_date - timedelta(days=1),
            datetime.min.time(),
            tzinfo=timezone.utc,
        ),
        execution_date=execution_date,
        features=values,
    )


def _validation_candidates() -> tuple[FeatureSnapshot, ...]:
    raw_scores = (0.1, 0.3, 0.5, 0.7, 0.9)
    rows = []
    for index, score in enumerate(raw_scores):
        features = {}
        for horizon in (5, 20, 60):
            features[f"profit_score_{horizon}"] = score
            features[f"alpha_{horizon}"] = score
            features[f"return_{horizon}"] = 0.03
            features[f"downside_{horizon}"] = 0.01
        rows.append(
            _candidate(
                f"validation-{index}",
                date(2024, 12, 1) + timedelta(days=index),
                features,
            )
        )
    return tuple(rows)


def _test_features(*, signal_5: float, signal_20: float, signal_60: float) -> dict:
    return {
        "profit_score_5": signal_5,
        "alpha_5": signal_5,
        "return_5": 0.03,
        "downside_5": 0.01,
        "profit_score_20": signal_20,
        "alpha_20": signal_20,
        "return_20": 0.04,
        "downside_20": 0.02,
        "profit_score_60": signal_60,
        "alpha_60": signal_60,
        "return_60": 0.05,
        "downside_60": 0.03,
    }


def test_v5_adapter_chooses_horizon_and_emits_generic_opportunity() -> None:
    config = V5StrategyConfig(validation_top_fraction=0.25)
    models = _models()
    calibration = V5CalibrationState.from_validation(
        _validation_candidates(),
        models,
        config,
    )
    strategy = V5AdaptiveHorizonStrategy(models, calibration, config)
    candidate = _candidate(
        "test",
        date(2025, 1, 2),
        _test_features(signal_5=0.6, signal_20=0.8, signal_60=0.4),
    )
    portfolio = PortfolioSnapshot(
        as_of=datetime(2025, 1, 1, 12, tzinfo=timezone.utc),
        equity=10_000.0,
        cash=10_000.0,
        gross_exposure_pct=0.0,
    )

    opportunities = strategy.evaluate((candidate,), portfolio)

    assert len(opportunities) == 1
    opportunity = opportunities[0]
    assert opportunity.strategy_id == config.strategy_id
    assert opportunity.candidate_id == "test"
    assert opportunity.horizon_sessions == 20
    assert opportunity.expected_return == pytest.approx(0.04)
    assert opportunity.expected_alpha == pytest.approx(0.8)
    assert opportunity.expected_downside == pytest.approx(0.02)
    assert opportunity.score >= config.rank_threshold


def test_v5_adapter_same_day_candidates_share_only_prior_calibration_history() -> None:
    config = V5StrategyConfig(validation_top_fraction=0.25)
    models = _models()
    calibration = V5CalibrationState.from_validation(
        _validation_candidates(),
        models,
        config,
    )
    strategy = V5AdaptiveHorizonStrategy(models, calibration, config)
    execution_date = date(2025, 1, 2)
    first = _candidate(
        "first",
        execution_date,
        _test_features(signal_5=0.6, signal_20=0.80, signal_60=0.4),
    )
    second = _candidate(
        "second",
        execution_date,
        _test_features(signal_5=0.6, signal_20=0.85, signal_60=0.4),
    )
    portfolio = PortfolioSnapshot(
        as_of=datetime(2025, 1, 1, 12, tzinfo=timezone.utc),
        equity=10_000.0,
        cash=10_000.0,
        gross_exposure_pct=0.0,
    )

    opportunities = strategy.evaluate((first, second), portfolio)

    assert len(opportunities) == 2
    # Both 0.80 and 0.85 lie between the same two prior validation scores.
    # They therefore receive the same PIT final percentile; neither sees the other.
    assert opportunities[0].score == pytest.approx(opportunities[1].score)


def test_v5_adapter_rejects_all_horizons_when_downside_gate_fails() -> None:
    config = V5StrategyConfig(validation_top_fraction=0.25, max_expected_downside=0.06)
    models = _models()
    calibration = V5CalibrationState.from_validation(
        _validation_candidates(),
        models,
        config,
    )
    strategy = V5AdaptiveHorizonStrategy(models, calibration, config)
    features = _test_features(signal_5=0.9, signal_20=0.9, signal_60=0.9)
    for horizon in (5, 20, 60):
        features[f"downside_{horizon}"] = 0.20
    candidate = _candidate("unsafe", date(2025, 1, 2), features)
    portfolio = PortfolioSnapshot(
        as_of=datetime(2025, 1, 1, 12, tzinfo=timezone.utc),
        equity=10_000.0,
        cash=10_000.0,
        gross_exposure_pct=0.0,
    )

    assert strategy.evaluate((candidate,), portfolio) == ()
