from __future__ import annotations

import json
from datetime import date

from stock_trading.live.runtime_strategy_state import FileRuntimeStrategyStateStore
from stock_trading.ml.online_calibration import RollingScoreHistory
from stock_trading.strategies.v5_adaptive_horizon import (
    V5AdaptiveHorizonStrategy,
    V5CalibrationState,
    V5StrategyConfig,
)


def _strategy(score: float) -> V5AdaptiveHorizonStrategy:
    strategy = object.__new__(V5AdaptiveHorizonStrategy)
    strategy.config = V5StrategyConfig(
        strategy_id="test-strategy",
        horizons=(5,),
        calibration_window_days=365,
    )
    profit = RollingScoreHistory(window_days=365)
    alpha = RollingScoreHistory(window_days=365)
    final = RollingScoreHistory(window_days=365)
    day = date(2026, 8, 20)
    profit.seed(((day, score),))
    alpha.seed(((day, score + 1.0),))
    final.seed(((day, score + 2.0),))
    strategy.calibration = V5CalibrationState(
        profit_histories={5: profit},
        alpha_histories={5: alpha},
        final_history=final,
    )
    strategy.models = {}
    return strategy


def test_evaluated_checkpoint_restores_calibration_without_claiming_completion(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"artifact":"a"}\n', encoding="utf-8")
    store = FileRuntimeStrategyStateStore(tmp_path / "state")
    checkpointed = _strategy(1.5)

    path = store.save_checkpoint(
        checkpointed,
        manifest,
        evaluated_batch_id="batch_a",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["evaluated_batch_id"] == "batch_a"
    assert "completed_batch_id" not in payload

    restored = _strategy(99.0)
    assert store.restore(restored, manifest) is True
    assert restored.calibration.profit_histories[5].snapshot() == (
        (date(2026, 8, 20), 1.5),
    )

    completed_path = store.save(
        restored,
        manifest,
        completed_batch_id="batch_a",
    )
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    assert completed["completed_batch_id"] == "batch_a"
    assert "evaluated_batch_id" not in completed
