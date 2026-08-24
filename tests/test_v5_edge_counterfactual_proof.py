from datetime import date

from stock_trading.experiments.v5_edge_counterfactual_proof import (
    _clone_score_history,
    _fresh_strategy,
    _scorecards_match,
)
from stock_trading.ml.online_calibration import RollingScoreHistory
from stock_trading.strategies.v5_adaptive_horizon import (
    V5AdaptiveHorizonStrategy,
    V5CalibrationState,
    V5HorizonModels,
    V5StrategyConfig,
)


def _history() -> RollingScoreHistory:
    history = RollingScoreHistory(window_days=365)
    history.seed(((date(2020, 1, 1), 0.25), (date(2020, 1, 2), 0.75)))
    return history


def _strategy() -> V5AdaptiveHorizonStrategy:
    config = V5StrategyConfig(horizons=(5,))
    calibration = V5CalibrationState(
        profit_histories={5: _history()},
        alpha_histories={5: _history()},
        final_history=_history(),
    )
    return V5AdaptiveHorizonStrategy(
        {5: V5HorizonModels(profit=object(), alpha=object())},
        calibration,
        config,
    )


def test_clone_score_history_is_independent() -> None:
    source = _history()
    clone = _clone_score_history(source)

    assert clone.snapshot() == source.snapshot()
    clone.percentiles(date(2020, 1, 3), (0.5,))

    assert clone.snapshot() != source.snapshot()
    assert source.snapshot() == (
        (date(2020, 1, 1), 0.25),
        (date(2020, 1, 2), 0.75),
    )


def test_fresh_strategy_clones_all_calibration_histories() -> None:
    source = _strategy()
    clone = _fresh_strategy(source)

    assert clone is not source
    assert clone.models == source.models
    assert clone.config == source.config
    assert (
        clone.calibration.profit_histories[5].snapshot()
        == source.calibration.profit_histories[5].snapshot()
    )

    clone.calibration.final_history.percentiles(date(2020, 1, 3), (0.5,))
    assert (
        clone.calibration.final_history.snapshot()
        != source.calibration.final_history.snapshot()
    )


def test_scorecards_match_detects_counterfactual_drift() -> None:
    baseline = {
        "starting_capital": 10000.0,
        "ending_capital": 10425.0,
        "net_profit": 425.0,
        "total_return": 0.0425,
        "profit_factor": 1.31,
        "realized_max_drawdown": 0.039,
        "total_trades": 205,
        "win_rate": 0.527,
        "average_net_trade_return": 0.01036,
        "average_trade_alpha": -0.00081,
        "net_profit_excluding_best_entry_year": 111.25,
    }
    assert _scorecards_match(baseline, dict(baseline))

    changed = dict(baseline)
    changed["total_trades"] = 192
    assert not _scorecards_match(baseline, changed)
