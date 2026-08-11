import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from stock_trading.core import RawRecord


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
        self._write_once(
            metadata_path,
            json.dumps(metadata, sort_keys=True, indent=2).encode("utf-8"),
        )
        return content_path

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
