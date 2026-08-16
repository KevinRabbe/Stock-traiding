from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    path: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class StrategyArtifactManifest:
    strategy_id: str
    root: str
    files: tuple[ArtifactFile, ...]
    manifest_sha256: str


def build_strategy_artifact_manifest(
    strategy_id: str,
    root: str | Path,
) -> StrategyArtifactManifest:
    base = Path(root)
    if not strategy_id.strip():
        raise ValueError("strategy_id must not be empty")
    if not base.is_dir():
        raise FileNotFoundError(f"strategy artifact directory does not exist: {base}")
    files = tuple(
        ArtifactFile(
            path=path.relative_to(base).as_posix(),
            sha256=_sha256_file(path),
            size=path.stat().st_size,
        )
        for path in sorted(path for path in base.rglob("*") if path.is_file())
    )
    if not files:
        raise ValueError("strategy artifact directory contains no files")
    digest = _manifest_digest(strategy_id, files)
    return StrategyArtifactManifest(
        strategy_id=strategy_id,
        root=str(base),
        files=files,
        manifest_sha256=digest,
    )


def verify_strategy_artifact_manifest(
    manifest: StrategyArtifactManifest,
    *,
    root: str | Path | None = None,
) -> None:
    base = Path(root) if root is not None else Path(manifest.root)
    if _manifest_digest(manifest.strategy_id, manifest.files) != manifest.manifest_sha256:
        raise ValueError("strategy artifact manifest digest is invalid")
    for item in manifest.files:
        path = base / Path(item.path)
        if not path.is_file():
            raise FileNotFoundError(f"strategy artifact file missing: {item.path}")
        if path.stat().st_size != item.size or _sha256_file(path) != item.sha256:
            raise ValueError(f"strategy artifact file changed: {item.path}")


def write_strategy_artifact_manifest(
    manifest: StrategyArtifactManifest,
    destination: str | Path,
) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_strategy_artifact_manifest(path: str | Path) -> StrategyArtifactManifest:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        files = tuple(ArtifactFile(**item) for item in payload["files"])
        manifest = StrategyArtifactManifest(
            strategy_id=str(payload["strategy_id"]),
            root=str(payload["root"]),
            files=files,
            manifest_sha256=str(payload["manifest_sha256"]),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid strategy artifact manifest: {source}") from exc
    if _manifest_digest(manifest.strategy_id, manifest.files) != manifest.manifest_sha256:
        raise ValueError("strategy artifact manifest digest is invalid")
    return manifest


def _manifest_digest(strategy_id: str, files: tuple[ArtifactFile, ...]) -> str:
    canonical = {
        "strategy_id": strategy_id,
        "files": [asdict(item) for item in files],
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
