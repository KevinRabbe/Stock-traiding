from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile

from .contracts import EngineCycleResult, StrategyStage
from .registry import (
    StrategyMetadataStore,
    StrategyRecord,
    StrategyRegistrySnapshot,
    StrategyScorecard,
)


_STRATEGY_SCHEMA_VERSION = 1


class FileStrategyMetadataStore(StrategyMetadataStore):
    """Atomic durable champion/challenger metadata store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> StrategyRegistrySnapshot | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid strategy registry metadata at {self.path}") from exc
        if not isinstance(payload, dict):
            raise ValueError("strategy registry metadata must be an object")
        if payload.get("schema_version") != _STRATEGY_SCHEMA_VERSION:
            raise ValueError("unsupported strategy registry schema")
        records_payload = payload.get("records", ())
        if not isinstance(records_payload, list):
            raise ValueError("strategy registry records must be a list")
        records = tuple(_record_from_json(item) for item in records_payload)
        ids = [record.strategy_id for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate strategy metadata record")
        champion_id = payload.get("champion_id")
        if champion_id is not None and champion_id not in set(ids):
            raise ValueError("persisted champion does not exist in strategy records")
        return StrategyRegistrySnapshot(
            champion_id=str(champion_id) if champion_id is not None else None,
            records=records,
        )

    def save(self, snapshot: StrategyRegistrySnapshot) -> None:
        payload = {
            "schema_version": _STRATEGY_SCHEMA_VERSION,
            "champion_id": snapshot.champion_id,
            "records": [_record_to_json(record) for record in snapshot.records],
        }
        _atomic_json_write(self.path, payload)


class JsonlEngineAuditObserver:
    """Append-only durable audit journal for every completed engine cycle."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def record(self, result: EngineCycleResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "schema_version": 1,
            "kind": "engine_cycle",
            "payload": _json_value(result),
        }
        encoded = json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ) + "\n"
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())


def _record_to_json(record: StrategyRecord) -> dict:
    scorecard = record.scorecard
    scorecard_json = None
    if scorecard is not None:
        scorecard_json = {
            "compounded_return": scorecard.compounded_return,
            "profit_factor": (
                "inf" if math.isinf(scorecard.profit_factor) else scorecard.profit_factor
            ),
            "worst_realized_drawdown": scorecard.worst_realized_drawdown,
            "total_trades": scorecard.total_trades,
            "profitable_year_rate": scorecard.profitable_year_rate,
            "average_trade_alpha": scorecard.average_trade_alpha,
        }
    return {
        "strategy_id": record.strategy_id,
        "stage": record.stage.value,
        "artifact_ref": record.artifact_ref,
        "scorecard": scorecard_json,
        "selection_score": record.selection_score,
        "notes": record.notes,
    }


def _record_from_json(payload: dict) -> StrategyRecord:
    if not isinstance(payload, dict):
        raise ValueError("invalid strategy registry record")
    try:
        scorecard_payload = payload.get("scorecard")
        scorecard = None
        if scorecard_payload is not None:
            if not isinstance(scorecard_payload, dict):
                raise ValueError("invalid strategy scorecard")
            profit_factor_raw = scorecard_payload["profit_factor"]
            profit_factor = (
                math.inf if profit_factor_raw == "inf" else float(profit_factor_raw)
            )
            scorecard = StrategyScorecard(
                compounded_return=float(scorecard_payload["compounded_return"]),
                profit_factor=profit_factor,
                worst_realized_drawdown=float(
                    scorecard_payload["worst_realized_drawdown"]
                ),
                total_trades=int(scorecard_payload["total_trades"]),
                profitable_year_rate=float(scorecard_payload["profitable_year_rate"]),
                average_trade_alpha=(
                    float(scorecard_payload["average_trade_alpha"])
                    if scorecard_payload.get("average_trade_alpha") is not None
                    else None
                ),
            )
        return StrategyRecord(
            strategy_id=str(payload["strategy_id"]),
            stage=StrategyStage(str(payload["stage"])),
            artifact_ref=(
                str(payload["artifact_ref"])
                if payload.get("artifact_ref") is not None
                else None
            ),
            scorecard=scorecard,
            selection_score=(
                float(payload["selection_score"])
                if payload.get("selection_score") is not None
                else None
            ),
            notes=str(payload.get("notes") or ""),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid strategy registry record") from exc


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
            newline="\n",
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _json_value(value):
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value
