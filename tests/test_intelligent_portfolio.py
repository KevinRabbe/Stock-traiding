from datetime import date, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from stock_trading.backtest.intelligent_portfolio import (
    PortfolioIntelligenceConfig,
    dynamic_allocation_pct,
    trailing_return_correlation,
)
from stock_trading.ml.lightgbm_models import OpportunityPrediction


class _FakeMarketStore:
    def __init__(self, series):
        self.series = series

    def bars_before(self, security_id, day, limit):
        bars = [bar for bar in self.series[security_id] if bar.date < day]
        return bars[-limit:]


def _bars(returns):
    current = 100.0
    result = [
        SimpleNamespace(date=date(2024, 1, 1), adj_close=Decimal(str(current)))
    ]
    for index, value in enumerate(returns, start=1):
        current *= 1.0 + value
        result.append(
            SimpleNamespace(
                date=date(2024, 1, 1) + timedelta(days=index),
                adj_close=Decimal(str(current)),
            )
        )
    return result


def _prediction(score, downside):
    return OpportunityPrediction(
        expected_alpha_20d=0.05,
        expected_downside_20d=downside,
        probability_positive_alpha=0.8,
        opportunity_score=score,
    )


def test_dynamic_allocation_rewards_quality_and_penalizes_risk_and_correlation() -> None:
    config = PortfolioIntelligenceConfig()
    strong = dynamic_allocation_pct(
        _prediction(0.999, 0.005),
        config,
        max_correlation=0.2,
    )
    weak_correlated = dynamic_allocation_pct(
        _prediction(0.951, 0.055),
        config,
        max_correlation=0.9,
    )

    assert config.min_allocation_pct <= weak_correlated < strong <= config.max_allocation_pct


def test_trailing_return_correlation_uses_only_aligned_prior_returns() -> None:
    returns = [0.01, -0.02, 0.015, 0.005, -0.01, 0.02]
    store = _FakeMarketStore(
        {
            "same-a": _bars(returns),
            "same-b": _bars(returns),
            "opposite": _bars([-value for value in returns]),
        }
    )
    cutoff = date(2024, 1, 20)

    positive = trailing_return_correlation(
        store,
        "same-a",
        "same-b",
        cutoff,
        lookback_sessions=6,
        min_observations=4,
    )
    negative = trailing_return_correlation(
        store,
        "same-a",
        "opposite",
        cutoff,
        lookback_sessions=6,
        min_observations=4,
    )

    assert positive == pytest.approx(1.0)
    assert negative is not None and negative < -0.99


def test_portfolio_config_rejects_correlated_cap_below_single_position_cap() -> None:
    with pytest.raises(ValueError, match="max_correlated_exposure_pct"):
        PortfolioIntelligenceConfig(
            max_allocation_pct=0.04,
            max_correlated_exposure_pct=0.03,
        )
