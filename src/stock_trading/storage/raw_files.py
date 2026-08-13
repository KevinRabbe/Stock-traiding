import json
import os
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from stock_trading.core import RawRecord, Source


_CONTENT_EXTENSIONS = {
    "application/json": ".json",
    "application/xml": ".xml",
    "application/zip": ".zip",
    "text/csv": ".csv",
    "text/plain": ".txt",
}


class FileRawStore:
    """Immutable content-addressed storage for raw source artifacts."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def put(self, record: RawRecord) -> Path:
        source_dir = self.root / record.source.value / record.fetched_at.strftime("%Y/%m")
        source_dir.mkdir(parents=True, exist_ok=True)

        extension = _CONTENT_EXTENSIONS.get(record.content_type, ".bin")
        content_path = source_dir / f"{record.artifact_id}{extension}"
        metadata_path = source_dir / f"{record.artifact_id}.metadata.json"

        payload = record.content if isinstance(record.content, bytes) else record.content.encode("utf-8")
        self._write_once(content_path, payload)

        metadata = {
            "artifact_id": record.artifact_id,
            "source": record.source.value,
            "source_record_id": record.source_record_id,
            "fetched_at": record.fetched_at.isoformat(),
            "content_type": record.content_type,
            "sha256": record.sha256,
        }
        self._write_metadata_once(metadata_path, metadata)
        return content_path

    def latest(self, source: Source, source_record_id: str) -> RawRecord | None:
        """Return the newest immutable snapshot already stored for a source record.

        This is intentionally keyed by the provider's source record id rather
        than only the content hash so interrupted historical backfills can
        resume without downloading completed source records again.
        """

        source_root = self.root / source.value
        if not source_root.exists():
            return None

        matches: list[tuple[datetime, Path, dict[str, str]]] = []
        for metadata_path in source_root.glob("**/*.metadata.json"):
            metadata = self._read_metadata(metadata_path)
            if metadata.get("source") != source.value:
                continue
            if metadata.get("source_record_id") != source_record_id:
                continue
            try:
                fetched_at = datetime.fromisoformat(metadata["fetched_at"])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"invalid raw artifact metadata at {metadata_path}") from exc
            matches.append((fetched_at, metadata_path, metadata))

        if not matches:
            return None

        _, metadata_path, metadata = max(matches, key=lambda item: item[0])
        extension = _CONTENT_EXTENSIONS.get(metadata["content_type"], ".bin")
        content_path = metadata_path.with_name(f"{metadata['artifact_id']}{extension}")
        if not content_path.exists():
            raise ValueError(f"raw artifact content missing for {metadata_path}")

        content = content_path.read_bytes()
        return RawRecord(
            source=source,
            source_record_id=metadata["source_record_id"],
            fetched_at=metadata["fetched_at"],
            content_type=metadata["content_type"],
            content=content,
            sha256=metadata["sha256"],
        )

    @staticmethod
    def _read_metadata(path: Path) -> dict[str, str]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid raw artifact metadata at {path}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"invalid raw artifact metadata at {path}")
        return value

    @classmethod
    def _write_metadata_once(cls, path: Path, metadata: dict[str, str]) -> None:
        if path.exists():
            existing = cls._read_metadata(path)
            # fetched_at describes this HTTP retrieval, not content identity.
            # A rerun may fetch the exact same immutable artifact later in the
            # same month; preserve the original metadata instead of treating
            # the new retrieval timestamp as a content collision.
            stable_keys = (
                "artifact_id",
                "source",
                "source_record_id",
                "content_type",
                "sha256",
            )
            if any(existing.get(key) != metadata.get(key) for key in stable_keys):
                raise ValueError(f"immutable raw artifact collision at {path}")
            return

        cls._write_once(
            path,
            json.dumps(metadata, sort_keys=True, indent=2).encode("utf-8"),
        )

    @staticmethod
    def _write_once(path: Path, content: bytes) -> None:
        if path.exists():
            existing = path.read_bytes()
            if existing != content:
                raise ValueError(f"immutable raw artifact collision at {path}")
            return

        with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
            temp.write(content)
            temp.flush()
            os.fsync(temp.fileno())
            temporary_path = Path(temp.name)
        os.replace(temporary_path, path)
