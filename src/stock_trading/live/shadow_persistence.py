from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

from .service import ShadowStrategyResult


class JsonlShadowAuditObserver:
    """Append-only, fsync-backed evidence journal for SHADOW decisions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def record(
        self,
        as_of,
        results: tuple[ShadowStrategyResult, ...],
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "schema_version": 1,
            "kind": "shadow_cycle",
            "as_of": _json_value(as_of),
            "results": [_json_value(asdict(item)) for item in results],
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


def _json_value(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value
