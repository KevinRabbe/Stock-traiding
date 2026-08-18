from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from stock_trading.engine import StrategyRegistry, StrategyStage
from stock_trading.ml.online_calibration import RollingScoreHistory
from stock_trading.strategies.v5_adaptive_horizon import (
    V5AdaptiveHorizonStrategy,
    V5CalibrationState,
)


_SCHEMA_VERSION = 1


class FileRuntimeStrategyStateStore:
    """Mutable forward-only calibration overlays for immutable strategy artifacts.

    Frozen manifests remain the authority for model weights and initial calibration.
    This store persists only the rolling calibration state accumulated by completed
    forward PAPER/SHADOW cycles. Each overlay is bound to the exact artifact-manifest
    bytes; replacing an artifact therefore makes an old overlay fail closed instead
    of silently crossing strategy generations.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def restore_registry(self, registry: StrategyRegistry) -> tuple[str, ...]:
        restored: list[str] = []
        strategies = [registry.active()]
        strategies.extend(
            registry.loaded_challenger_strategies(stages=(StrategyStage.SHADOW,))
        )
        for strategy in strategies:
            record = registry.record(strategy.strategy_id)
            if not record.artifact_ref:
                raise ValueError(
                    f"active strategy {strategy.strategy_id} has no artifact manifest"
                )
            if self.restore(strategy, record.artifact_ref):
                restored.append(strategy.strategy_id)
        return tuple(sorted(restored))

    def save_registry(
        self,
        registry: StrategyRegistry,
        *,
        completed_batch_id: str,
    ) -> tuple[Path, ...]:
        if not completed_batch_id.strip():
            raise ValueError("completed_batch_id must not be empty")
        paths: list[Path] = []
        strategies = [registry.active()]
        strategies.extend(
            registry.loaded_challenger_strategies(stages=(StrategyStage.SHADOW,))
        )
        for strategy in strategies:
            record = registry.record(strategy.strategy_id)
            if not record.artifact_ref:
                raise ValueError(
                    f"active strategy {strategy.strategy_id} has no artifact manifest"
                )
            paths.append(
                self.save(
                    strategy,
                    record.artifact_ref,
                    completed_batch_id=completed_batch_id,
                )
            )
        return tuple(paths)

    def restore(self, strategy, artifact_manifest_path: str | Path) -> bool:
        path = self._path(strategy.strategy_id)
        if not path.exists():
            return False
        payload = _read_json(path)
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(f"unsupported runtime strategy state schema: {path}")
        if payload.get("strategy_id") != strategy.strategy_id:
            raise ValueError("runtime strategy state strategy_id mismatch")
        expected_manifest = _manifest_digest(artifact_manifest_path)
        if payload.get("artifact_manifest_sha256") != expected_manifest:
            raise ValueError(
                f"runtime strategy state artifact identity mismatch for {strategy.strategy_id}"
            )
        _restore_calibration(strategy, payload.get("calibration"))
        return True

    def save(
        self,
        strategy,
        artifact_manifest_path: str | Path,
        *,
        completed_batch_id: str,
    ) -> Path:
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "strategy_id": strategy.strategy_id,
            "artifact_manifest_sha256": _manifest_digest(artifact_manifest_path),
            "completed_batch_id": completed_batch_id,
            "calibration": _calibration_payload(strategy),
        }
        path = self._path(strategy.strategy_id)
        _atomic_json_write(path, payload)
        return path

    def _path(self, strategy_id: str) -> Path:
        safe = strategy_id.replace("/", "_").replace("\\", "_")
        return self.root / f"{safe}.json"


def _calibration_payload(strategy) -> dict[str, Any]:
    if not isinstance(strategy, V5AdaptiveHorizonStrategy):
        raise TypeError(
            f"runtime calibration persistence is not implemented for {type(strategy).__name__}"
        )
    config = strategy.config
    state = strategy.calibration
    if set(state.profit_histories) != set(config.horizons):
        raise ValueError("runtime profit calibration horizons do not match strategy")
    if set(state.alpha_histories) != set(config.horizons):
        raise ValueError("runtime alpha calibration horizons do not match strategy")
    return {
        "window_days": config.calibration_window_days,
        "horizons": list(config.horizons),
        "profit_histories": {
            str(horizon): _history_payload(state.profit_histories[horizon])
            for horizon in config.horizons
        },
        "alpha_histories": {
            str(horizon): _history_payload(state.alpha_histories[horizon])
            for horizon in config.horizons
        },
        "final_history": _history_payload(state.final_history),
    }


def _restore_calibration(strategy, payload: Any) -> None:
    if not isinstance(strategy, V5AdaptiveHorizonStrategy):
        raise TypeError(
            f"runtime calibration persistence is not implemented for {type(strategy).__name__}"
        )
    if not isinstance(payload, dict):
        raise ValueError("invalid runtime calibration payload")
    config = strategy.config
    try:
        window_days = int(payload["window_days"])
        horizons = tuple(int(item) for item in payload["horizons"])
        raw_profit = dict(payload["profit_histories"])
        raw_alpha = dict(payload["alpha_histories"])
        raw_final = payload["final_history"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid runtime calibration payload") from exc
    if window_days != config.calibration_window_days:
        raise ValueError("runtime calibration window differs from strategy config")
    if horizons != tuple(config.horizons):
        raise ValueError("runtime calibration horizons differ from strategy config")
    expected_keys = {str(item) for item in config.horizons}
    if set(raw_profit) != expected_keys or set(raw_alpha) != expected_keys:
        raise ValueError("runtime calibration history horizons differ from strategy config")

    strategy.calibration = V5CalibrationState(
        profit_histories={
            horizon: _restore_history(raw_profit[str(horizon)], window_days)
            for horizon in config.horizons
        },
        alpha_histories={
            horizon: _restore_history(raw_alpha[str(horizon)], window_days)
            for horizon in config.horizons
        },
        final_history=_restore_history(raw_final, window_days),
    )


def _history_payload(history: RollingScoreHistory) -> list[list[Any]]:
    return [[day.isoformat(), float(score)] for day, score in history.snapshot()]


def _restore_history(values: Any, window_days: int) -> RollingScoreHistory:
    history = RollingScoreHistory(window_days=window_days)
    try:
        history.seed(
            (date.fromisoformat(str(item[0])), float(item[1]))
            for item in values
        )
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("invalid runtime calibration history") from exc
    return history


def _manifest_digest(path: str | Path) -> str:
    manifest = Path(path)
    try:
        content = manifest.read_bytes()
    except OSError as exc:
        raise FileNotFoundError(f"missing strategy artifact manifest: {manifest}") from exc
    return hashlib.sha256(content).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid runtime strategy state: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"runtime strategy state must be an object: {path}")
    return payload


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
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
