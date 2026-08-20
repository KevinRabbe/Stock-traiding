from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from stock_trading.core import as_utc


@dataclass(frozen=True, slots=True)
class PaperRuntimeBatchCommit:
    """Durable proof that one runtime batch crossed the PAPER broker boundary."""

    batch_id: str
    committed_at: datetime
    submitted_order_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "committed_at", as_utc(self.committed_at))
        if not self.batch_id.strip():
            raise ValueError("PAPER runtime batch commit batch_id must not be empty")
        if len(self.submitted_order_ids) != len(set(self.submitted_order_ids)):
            raise ValueError("PAPER runtime batch commit contains duplicate order IDs")


class FilePaperRuntimeBatchCommitStore:
    """Atomic append-by-identity broker completion records for crash recovery."""

    SCHEMA_VERSION = 1

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @classmethod
    def for_ledger(cls, ledger) -> "FilePaperRuntimeBatchCommitStore":
        ledger_path = Path(ledger.path)
        return cls(
            ledger_path.parent / f"{ledger_path.stem}_runtime_batch_commits"
        )

    def load(self, batch_id: str) -> PaperRuntimeBatchCommit | None:
        path = self._path(batch_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid PAPER runtime batch commit: {path}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("unsupported PAPER runtime batch commit schema")
        try:
            return PaperRuntimeBatchCommit(
                batch_id=str(payload["batch_id"]),
                committed_at=datetime.fromisoformat(str(payload["committed_at"])),
                submitted_order_ids=tuple(
                    str(item) for item in payload.get("submitted_order_ids", ())
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid PAPER runtime batch commit: {path}") from exc

    def write(self, commit: PaperRuntimeBatchCommit) -> Path:
        path = self._path(commit.batch_id)
        existing = self.load(commit.batch_id)
        if existing is not None:
            if existing.submitted_order_ids != commit.submitted_order_ids:
                raise ValueError("PAPER runtime batch commit changed submitted order IDs")
            return path
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            **asdict(commit),
            "committed_at": commit.committed_at.isoformat(),
            "submitted_order_ids": list(commit.submitted_order_ids),
        }
        _atomic_json_write(path, payload)
        return path

    def _path(self, batch_id: str) -> Path:
        if not batch_id.startswith("batch_"):
            raise ValueError("invalid PAPER runtime batch_id")
        return self.root / f"{batch_id}.json"


def _atomic_json_write(path: Path, payload: dict) -> None:
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
