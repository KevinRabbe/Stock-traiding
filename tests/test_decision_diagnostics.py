from datetime import date, datetime, timezone
from types import SimpleNamespace

from stock_trading.engine import FeatureSnapshot, PortfolioSnapshot
from stock_trading.ml.dataset import FeatureSchema
from stock_trading.ml.online_calibration import RollingScoreHistory
from stock_trading.strategies.v5_adaptive_horizon import (
    V5AdaptiveHorizonStrategy,
    V5CalibrationState,
    V5StrategyConfig,
)
from stock_trading.live.decision_diagnostics import diagnose_strategy


UTC = timezone.utc


class _ProfitModel:
    feature_schema = FeatureSchema(("x", "missing"))

    def __init__(self, expected_return: float, downside: float, score: float) -> None:
        self.expected_return = expected_return
        self.downside = downside
        self.score = score

    def predict(self, features):
        del features
        return SimpleNamespace(
            expected_stock_return_20d=self.expected_return,
            expected_downside_20d=self.downside,
            probability_profitable_return=0.6,
            profit_score=self.score,
        )


class _AlphaModel:
    feature_schema = FeatureSchema(("x", "missing"))

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def predict(self, features):
        del features
        return SimpleNamespace(expected_alpha_20d=self.alpha)


def _strategy(*, expected_return: float, downside: float, score: float, alpha: float):
    config = V5StrategyConfig(
        strategy_id="diag-strategy",
        horizons=(5,),
        validation_top_fraction=0.05,
        alpha_rank_weight=0.25,
        calibration_window_days=365,
        min_expected_return=0.002,
        max_expected_downside=0.06,
    )
    profit_history = RollingScoreHistory(window_days=365)
    alpha_history = RollingScoreHistory(window_days=365)
    final_history = RollingScoreHistory(window_days=365)
    prior_day = date(2026, 8, 18)
    profit_history.seed(((prior_day, 1.0),))
    alpha_history.seed(((prior_day, 1.0),))
    final_history.seed(((prior_day, 0.9),))
    calibration = V5CalibrationState(
        profit_histories={5: profit_history},
        alpha_histories={5: alpha_history},
        final_history=final_history,
    )
    model = SimpleNamespace(
        profit=_ProfitModel(expected_return, downside, score),
        alpha=_AlphaModel(alpha),
    )
    return V5AdaptiveHorizonStrategy({5: model}, calibration, config)


def _candidate():
    return FeatureSnapshot(
        candidate_id="opportunity:company-a:2026-08-19",
        event_id="opportunity:company-a:2026-08-19",
        company_id="company-a",
        security_id="security-a",
        decision_time=datetime(2026, 8, 18, 18, 0, tzinfo=UTC),
        execution_date=date(2026, 8, 19),
        features={"x": 1.0},
    )


def _portfolio():
    return PortfolioSnapshot(
        as_of=datetime(2026, 8, 18, 18, 0, tzinfo=UTC),
        equity=10_000.0,
        cash=10_000.0,
        gross_exposure_pct=0.0,
    )


def test_diagnostics_explain_no_eligible_horizon_without_mutating_state() -> None:
    strategy = _strategy(expected_return=-0.01, downside=0.01, score=-0.02, alpha=-0.01)
    before = strategy.calibration.profit_histories[5].snapshot()

    diagnostic = diagnose_strategy(strategy, (_candidate(),))

    assert strategy.calibration.profit_histories[5].snapshot() == before
    assert diagnostic.emitted_opportunity_count == 0
    decision = diagnostic.decisions[0]
    assert decision.rejection_reason == "no_eligible_horizon"
    assert decision.chosen_horizon is None
    horizon = decision.horizons[0]
    assert horizon.eligible is False
    assert horizon.eligibility_reasons == ("expected_return_below_minimum",)
    assert horizon.missing_feature_names == ("missing",)
    assert horizon.missing_feature_count == 1

    assert strategy.evaluate((_candidate(),), _portfolio()) == ()


def test_diagnostics_explain_final_rank_rejection_and_match_strategy_count() -> None:
    strategy = _strategy(expected_return=0.02, downside=0.01, score=0.0, alpha=0.0)

    diagnostic = diagnose_strategy(strategy, (_candidate(),))
    actual = strategy.evaluate((_candidate(),), _portfolio())

    assert diagnostic.emitted_opportunity_count == len(actual) == 0
    decision = diagnostic.decisions[0]
    assert decision.chosen_horizon == 5
    assert decision.final_percentile < decision.rank_threshold
    assert decision.rejection_reason == "below_final_rank_threshold"
    assert decision.horizons[0].eligible is True
