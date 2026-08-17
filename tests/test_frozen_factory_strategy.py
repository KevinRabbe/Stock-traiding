from __future__ import annotations

from datetime import date

import pytest

from stock_trading.ml.online_calibration import RollingScoreHistory
from stock_trading.strategies.frozen_factory import (
    _calibration_payload,
    _restore_calibration,
)
from stock_trading.strategies.v5_adaptive_horizon import (
    V5CalibrationState,
    V5StrategyConfig,
)


def _history(*values: tuple[date, float], window_days: int = 365) -> RollingScoreHistory:
    history = RollingScoreHistory(window_days=window_days)
    history.seed(values)
    return history


def test_frozen_calibration_round_trip_preserves_exact_history() -> None:
    config = V5StrategyConfig(
        strategy_id="factory-test",
        horizons=(5, 20),
        calibration_window_days=365,
    )
    calibration = V5CalibrationState(
        profit_histories={
            5: _history((date(2025, 1, 2), 0.1), (date(2025, 2, 3), 0.2)),
            20: _history((date(2025, 1, 2), 0.3)),
        },
        alpha_histories={
            5: _history((date(2025, 1, 2), -0.2)),
            20: _history((date(2025, 1, 2), 0.4), (date(2025, 3, 4), 0.5)),
        },
        final_history=_history((date(2025, 1, 2), 0.7)),
    )

    payload = _calibration_payload(calibration, config)
    restored = _restore_calibration(payload, config)

    assert restored.profit_histories[5].snapshot() == calibration.profit_histories[5].snapshot()
    assert restored.profit_histories[20].snapshot() == calibration.profit_histories[20].snapshot()
    assert restored.alpha_histories[5].snapshot() == calibration.alpha_histories[5].snapshot()
    assert restored.alpha_histories[20].snapshot() == calibration.alpha_histories[20].snapshot()
    assert restored.final_history.snapshot() == calibration.final_history.snapshot()


def test_frozen_calibration_rejects_window_mismatch() -> None:
    config = V5StrategyConfig(strategy_id="factory-test", horizons=(20,), calibration_window_days=365)
    calibration = V5CalibrationState(
        profit_histories={20: _history((date(2025, 1, 2), 0.1))},
        alpha_histories={20: _history((date(2025, 1, 2), 0.2))},
        final_history=_history((date(2025, 1, 2), 0.3)),
    )
    payload = _calibration_payload(calibration, config)
    payload["window_days"] = 30

    with pytest.raises(ValueError, match="calibration window"):
        _restore_calibration(payload, config)
