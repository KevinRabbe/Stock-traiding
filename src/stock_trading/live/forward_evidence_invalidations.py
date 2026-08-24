from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from stock_trading.core import as_utc


@dataclass(frozen=True, slots=True)
class ForwardEvidenceInvalidation:
    batch_id: str
    evidence_source: str
    reason: str
    invalidated_at: datetime

    def __post_init__(self) -> None:
        if not self.batch_id.startswith("batch_"):
            raise ValueError("invalid forward evidence batch_id")
        if not self.evidence_source.strip():
            raise ValueError("forward evidence invalidation requires evidence_source")
        if not self.reason.strip():
            raise ValueError("forward evidence invalidation requires reason")
        object.__setattr__(self, "invalidated_at", as_utc(self.invalidated_at))


class FileForwardEvidenceInvalidationStore:
    """Durable audit receipts for diagnostics excluded from forward evaluation."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> tuple[ForwardEvidenceInvalidation, ...]:
        if not self.path.exists():
            return ()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid forward evidence invalidation store: {self.path}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("unsupported forward evidence invalidation schema")
        values = payload.get("invalidations")
        if not isinstance(values, list):
            raise ValueError("forward evidence invalidations must be a list")
        result: list[ForwardEvidenceInvalidation] = []
        for item in values:
            if not isinstance(item, dict):
                raise ValueError("forward evidence invalidation must be an object")
            try:
                invalidated_at = datetime.fromisoformat(
                    str(item["invalidated_at"]).replace("Z", "+00:00")
                )
                result.append(
                    ForwardEvidenceInvalidation(
                        batch_id=str(item["batch_id"]),
                        evidence_source=str(item["evidence_source"]),
                        reason=str(item["reason"]),
                        invalidated_at=invalidated_at,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid forward evidence invalidation entry") from exc
        batch_ids = [item.batch_id for item in result]
        if len(batch_ids) != len(set(batch_ids)):
            raise ValueError("duplicate forward evidence invalidation batch_id")
        return tuple(sorted(result, key=lambda item: item.batch_id))

    def invalidated_batch_ids(self) -> frozenset[str]:
        return frozenset(item.batch_id for item in self.load())

    def add_many(self, values: tuple[ForwardEvidenceInvalidation, ...]) -> int:
        if not values:
            return 0
        existing = {item.batch_id: item for item in self.load()}
        added = 0
        for item in values:
            previous = existing.get(item.batch_id)
            if previous is not None:
                if (
                    previous.evidence_source != item.evidence_source
                    or previous.reason != item.reason
                ):
                    raise ValueError(
                        f"forward evidence invalidation changed for {item.batch_id}"
                    )
                continue
            existing[item.batch_id] = item
            added += 1
        if added:
            self._save(tuple(sorted(existing.values(), key=lambda item: item.batch_id)))
        return added

    def _save(self, values: tuple[ForwardEvidenceInvalidation, ...]) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "invalidations": [
                {
                    **asdict(item),
                    "invalidated_at": item.invalidated_at.isoformat(),
                }
                for item in values
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=self.path.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
