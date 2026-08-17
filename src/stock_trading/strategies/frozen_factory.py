from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from stock_trading.engine import (
    load_strategy_artifact_manifest,
    verify_strategy_artifact_manifest,
)
from stock_trading.ml.lightgbm_models import LightGbmModelBundle, ProfitLightGbmModelBundle
from stock_trading.ml.online_calibration import RollingScoreHistory

from .v5_adaptive_horizon import (
    V5AdaptiveHorizonStrategy,
    V5CalibrationState,
    V5HorizonModels,
    V5StrategyConfig,
)


FROZEN_FACTORY_SCHEMA = "frozen-factory-strategy-v1"


def write_frozen_factory_strategy(
    root: str | Path,
    *,
    strategy_id: str,
    model_year: int,
    models: Mapping[int, V5HorizonModels],
    calibration: V5CalibrationState,
    config: V5StrategyConfig,
    source: Mapping[str, Any],
    training: Mapping[str, Any],
) -> Path:
    """Persist one immutable-ready factory strategy directory.

    The directory is not considered trusted merely because this function wrote it;
    callers must build and verify a StrategyArtifactManifest over the completed
    directory before registering it for SHADOW/PAPER/LIVE use.
    """

    base = Path(root)
    if not strategy_id.strip():
        raise ValueError("strategy_id must not be empty")
    if model_year <= 0:
        raise ValueError("model_year must be positive")
    if config.strategy_id != strategy_id:
        raise ValueError("strategy_id does not match strategy config")
    if set(models) != set(config.horizons):
        raise ValueError("model horizons do not match strategy config")
    if base.exists() and any(base.iterdir()):
        raise FileExistsError(f"frozen strategy directory is not empty: {base}")
    base.mkdir(parents=True, exist_ok=True)

    models_root = base / "models"
    for horizon in config.horizons:
        horizon_models = models[horizon]
        horizon_models.profit.save(models_root / f"{horizon}d" / "profit")
        horizon_models.alpha.save(models_root / f"{horizon}d" / "alpha")

    payload = {
        "schema_version": FROZEN_FACTORY_SCHEMA,
        "strategy_id": strategy_id,
        "model_year": model_year,
        "config": {
            **asdict(config),
            "horizons": list(config.horizons),
        },
        "calibration": _calibration_payload(calibration, config),
        "source": dict(source),
        "training": dict(training),
    }
    metadata_path = base / "strategy.json"
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def load_frozen_factory_strategy(root: str | Path) -> V5AdaptiveHorizonStrategy:
    base = Path(root)
    metadata_path = base / "strategy.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid frozen factory metadata: {metadata_path}") from exc

    try:
        if payload["schema_version"] != FROZEN_FACTORY_SCHEMA:
            raise ValueError("unsupported frozen factory strategy schema")
        strategy_id = str(payload["strategy_id"])
        model_year = int(payload["model_year"])
        if model_year <= 0:
            raise ValueError("invalid frozen model_year")
        raw_config = dict(payload["config"])
        raw_config["horizons"] = tuple(int(item) for item in raw_config["horizons"])
        config = V5StrategyConfig(**raw_config)
        if config.strategy_id != strategy_id:
            raise ValueError("frozen strategy_id/config mismatch")
        models_root = base / "models"
        models = {
            horizon: V5HorizonModels(
                profit=ProfitLightGbmModelBundle.load(
                    models_root / f"{horizon}d" / "profit"
                ),
                alpha=LightGbmModelBundle.load(models_root / f"{horizon}d" / "alpha"),
            )
            for horizon in config.horizons
        }
        calibration = _restore_calibration(payload["calibration"], config)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid frozen factory strategy: {metadata_path}") from exc
    return V5AdaptiveHorizonStrategy(models, calibration, config)


def load_frozen_factory_strategy_from_manifest(
    manifest_path: str | Path,
) -> V5AdaptiveHorizonStrategy:
    """Verify an immutable manifest before loading its strategy plugin."""

    manifest = load_strategy_artifact_manifest(manifest_path)
    verify_strategy_artifact_manifest(manifest)
    strategy = load_frozen_factory_strategy(manifest.root)
    if strategy.strategy_id != manifest.strategy_id:
        raise ValueError("artifact manifest strategy_id does not match frozen strategy")
    return strategy


def _calibration_payload(
    calibration: V5CalibrationState,
    config: V5StrategyConfig,
) -> dict[str, Any]:
    if set(calibration.profit_histories) != set(config.horizons):
        raise ValueError("profit calibration horizons do not match strategy config")
    if set(calibration.alpha_histories) != set(config.horizons):
        raise ValueError("alpha calibration horizons do not match strategy config")
    return {
        "window_days": config.calibration_window_days,
        "profit_histories": {
            str(horizon): _history_payload(calibration.profit_histories[horizon])
            for horizon in config.horizons
        },
        "alpha_histories": {
            str(horizon): _history_payload(calibration.alpha_histories[horizon])
            for horizon in config.horizons
        },
        "final_history": _history_payload(calibration.final_history),
    }


def _history_payload(history: RollingScoreHistory) -> list[list[Any]]:
    return [[day.isoformat(), float(score)] for day, score in history.snapshot()]


def _restore_calibration(
    payload: Mapping[str, Any],
    config: V5StrategyConfig,
) -> V5CalibrationState:
    window_days = int(payload["window_days"])
    if window_days != config.calibration_window_days:
        raise ValueError("calibration window does not match strategy config")

    raw_profit = payload["profit_histories"]
    raw_alpha = payload["alpha_histories"]
    if set(raw_profit) != {str(item) for item in config.horizons}:
        raise ValueError("frozen profit calibration horizons do not match config")
    if set(raw_alpha) != {str(item) for item in config.horizons}:
        raise ValueError("frozen alpha calibration horizons do not match config")

    profit_histories = {
        horizon: _restore_history(raw_profit[str(horizon)], window_days)
        for horizon in config.horizons
    }
    alpha_histories = {
        horizon: _restore_history(raw_alpha[str(horizon)], window_days)
        for horizon in config.horizons
    }
    final_history = _restore_history(payload["final_history"], window_days)
    return V5CalibrationState(
        profit_histories=profit_histories,
        alpha_histories=alpha_histories,
        final_history=final_history,
    )


def _restore_history(values: Any, window_days: int) -> RollingScoreHistory:
    history = RollingScoreHistory(window_days=window_days)
    try:
        history.seed(
            (date.fromisoformat(str(item[0])), float(item[1]))
            for item in values
        )
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("invalid frozen calibration history") from exc
    return history
