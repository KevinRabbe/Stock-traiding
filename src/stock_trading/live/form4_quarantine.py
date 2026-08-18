from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from stock_trading.core import as_utc


@dataclass(frozen=True, order=True, slots=True)
class QuarantinedForm4Filing:
    accepted_at: datetime
    cik: str
    accession_number: str
    raw_artifact_id: str
    error_type: str
    error_message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted_at", as_utc(self.accepted_at))
        for name in (
            "cik",
            "accession_number",
            "raw_artifact_id",
            "error_type",
            "error_message",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"quarantined filing {name} must not be empty")
            object.__setattr__(self, name, value)


class FileForm4Quarantine:
    """Atomic durable quarantine for SEC Form 4 documents we cannot normalize.

    The raw artifact remains in the immutable raw store. This file records why a
    specific accession could not be parsed before the current-event queue advances
    its source watermark. Re-recording the same accession/raw artifact is
    idempotent, which keeps crash recovery safe.

    A quarantine record can be resolved only after a separate recovery path has
    durably persisted the replacement raw artifact, normalized events, queue state,
    and recovery audit record. Removing it then means "no longer unresolved"; it
    does not erase the original raw artifact or the recovery audit trail.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> tuple[QuarantinedForm4Filing, ...]:
        if not self.path.exists():
            return ()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid Form 4 quarantine at {self.path}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("unsupported Form 4 quarantine schema")
        try:
            records = tuple(
                QuarantinedForm4Filing(
                    accepted_at=datetime.fromisoformat(str(item["accepted_at"])),
                    cik=str(item["cik"]),
                    accession_number=str(item["accession_number"]),
                    raw_artifact_id=str(item["raw_artifact_id"]),
                    error_type=str(item["error_type"]),
                    error_message=str(item["error_message"]),
                )
                for item in payload.get("records", ())
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid Form 4 quarantine at {self.path}") from exc
        identities = [(item.cik, item.accession_number) for item in records]
        if len(identities) != len(set(identities)):
            raise ValueError("Form 4 quarantine contains duplicate accessions")
        return tuple(sorted(records))

    def record(self, item: QuarantinedForm4Filing) -> bool:
        records = {(record.cik, record.accession_number): record for record in self.load()}
        key = (item.cik, item.accession_number)
        existing = records.get(key)
        if existing is not None:
            if existing.raw_artifact_id != item.raw_artifact_id:
                raise ValueError(
                    "quarantined SEC accession changed raw artifact identity: "
                    f"{item.accession_number}"
                )
            return False
        records[key] = item
        self._save(tuple(sorted(records.values())))
        return True

    def resolve(self, cik: str, accession_number: str) -> bool:
        """Remove one now-recovered accession from the active quarantine."""

        key = (str(cik).strip(), str(accession_number).strip())
        if not all(key):
            raise ValueError("quarantine resolution identity must not be empty")
        records = self.load()
        kept = tuple(
            record
            for record in records
            if (record.cik, record.accession_number) != key
        )
        if len(kept) == len(records):
            return False
        self._save(kept)
        return True

    def _save(self, records: tuple[QuarantinedForm4Filing, ...]) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "records": [
                {
                    "accepted_at": item.accepted_at.isoformat(),
                    "cik": item.cik,
                    "accession_number": item.accession_number,
                    "raw_artifact_id": item.raw_artifact_id,
                    "error_type": item.error_type,
                    "error_message": item.error_message,
                }
                for item in records
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
