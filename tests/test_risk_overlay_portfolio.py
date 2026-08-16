from types import SimpleNamespace

import pytest

from stock_trading.backtest.risk_overlay_portfolio import (
    RiskOverlayConfig,
    risk_overlay_allocation_pct,
)
from stock_trading.ml.lightgbm_models import OpportunityPrediction


def _candidate(*, downside: float, breadth: float = 1.0, benchmark_120d: float = 0.1, volatility_ratio: float = 1.0):
    prediction = OpportunityPrediction(
        expected_alpha_20d=0.05,
        expected_downside_20d=downside,
        probability_positive_alpha=0.8,
        opportunity_score=0.99,
    )
    row = SimpleNamespace(
        features={
            "system.regime.benchmark_trend_breadth": breadth,
            "market.benchmark_return_120d": benchmark_120d,
            "system.volatility.ratio_20_60": volatility_ratio,
        }
    )
    return SimpleNamespace(prediction=prediction, row=row)


def test_risk_overlay_never_upsizes_above_v5_base() -> None:
    config = RiskOverlayConfig()
    decision = risk_overlay_allocation_pct(
        _candidate(downside=0.0),
        config,
        max_correlation=None,
    )

    assert decision.allocation_pct == pytest.approx(config.base_allocation_pct)
    assert decision.allocation_pct <= config.base_allocation_pct


def test_risk_overlay_downsizes_high_predicted_downside() -> None:
    config = RiskOverlayConfig()
    low_risk = risk_overlay_allocation_pct(
        _candidate(downside=0.005),
        config,
        max_correlation=0.1,
    )
    high_risk = risk_overlay_allocation_pct(
        _candidate(downside=0.06),
        config,
        max_correlation=0.1,
    )

    assert config.min_allocation_pct <= high_risk.allocation_pct < low_risk.allocation_pct
    assert low_risk.allocation_pct <= config.base_allocation_pct


def test_risk_overlay_uses_market_regime_without_future_outcomes() -> None:
    config = RiskOverlayConfig()
    supportive = risk_overlay_allocation_pct(
        _candidate(
            downside=0.005,
            breadth=1.0,
            benchmark_120d=0.10,
            volatility_ratio=0.8,
        ),
        config,
        max_correlation=0.1,
    )
    hostile = risk_overlay_allocation_pct(
        _candidate(
            downside=0.005,
            breadth=0.0,
            benchmark_120d=-0.10,
            volatility_ratio=1.6,
        ),
        config,
        max_correlation=0.1,
    )

    assert hostile.regime_multiplier < supportive.regime_multiplier
    assert hostile.volatility_multiplier < supportive.volatility_multiplier
    assert hostile.allocation_pct < supportive.allocation_pct


def test_risk_overlay_penalizes_high_existing_correlation() -> None:
    config = RiskOverlayConfig()
    diversified = risk_overlay_allocation_pct(
        _candidate(downside=0.005),
        config,
        max_correlation=0.2,
    )
    crowded = risk_overlay_allocation_pct(
        _candidate(downside=0.005),
        config,
        max_correlation=0.95,
    )

    assert crowded.correlation_multiplier < diversified.correlation_multiplier
    assert crowded.allocation_pct < diversified.allocation_pct


def test_risk_overlay_config_rejects_minimum_above_base() -> None:
    with pytest.raises(ValueError, match="allocation percentages"):
        RiskOverlayConfig(min_allocation_pct=0.03, base_allocation_pct=0.02)
