from __future__ import annotations

import errno
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO


class FileRuntimeLock:
    """Non-blocking cross-process lock for one authoritative runtime directory.

    The lock file is intentionally persistent. Ownership comes from the operating
    system byte/file lock held by the open descriptor, not from file existence, so
    a crash releases authority automatically without a stale-lock cleanup race.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle: BinaryIO | None = None

    @property
    def acquired(self) -> bool:
        return self._handle is not None

    def acquire(self) -> bool:
        if self._handle is not None:
            raise RuntimeError("runtime lock is already acquired by this object")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        handle = os.fdopen(fd, "r+b", buffering=0)
        try:
            if os.fstat(handle.fileno()).st_size == 0:
                handle.write(b"\0")
                os.fsync(handle.fileno())
            handle.seek(0)
            if not _try_lock(handle):
                handle.close()
                return False

            self._handle = handle
            metadata = {
                "pid": os.getpid(),
                "acquired_at": datetime.now(timezone.utc).isoformat(),
            }
            encoded = (
                json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            handle.seek(0)
            handle.write(encoded)
            handle.truncate(len(encoded))
            os.fsync(handle.fileno())
            handle.seek(0)
            return True
        except Exception:
            if self._handle is handle:
                try:
                    _unlock(handle)
                finally:
                    self._handle = None
            handle.close()
            raise

    def holder(self) -> dict | None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        result: dict[str, object] = {}
        pid = payload.get("pid")
        acquired_at = payload.get("acquired_at")
        if isinstance(pid, int):
            result["pid"] = pid
        if isinstance(acquired_at, str) and acquired_at:
            result["acquired_at"] = acquired_at
        return result or None

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle.seek(0)
            _unlock(handle)
        finally:
            self._handle = None
            handle.close()

    def __enter__(self) -> FileRuntimeLock:
        if not self.acquire():
            raise RuntimeError("runtime lock is already held by another process")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.release()


def _try_lock(handle: BinaryIO) -> bool:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                return False
            raise
        return True

    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _unlock(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
