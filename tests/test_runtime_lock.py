from __future__ import annotations

import importlib
import os

import pytest

from stock_trading.live.run_current_pipeline import run_current_pipeline
from stock_trading.live.runtime_lock import FileRuntimeLock


def test_runtime_lock_excludes_second_holder_and_releases(tmp_path) -> None:
    path = tmp_path / "runtime" / "current_pipeline.lock"
    first = FileRuntimeLock(path)
    second = FileRuntimeLock(path)

    assert first.acquire() is True
    holder = first.holder()
    assert holder is not None
    assert holder["pid"] == os.getpid()
    assert isinstance(holder["acquired_at"], str)
    assert second.acquire() is False

    first.release()
    assert second.acquire() is True
    second.release()


def test_current_pipeline_fails_fast_when_runtime_is_busy(tmp_path) -> None:
    runtime_dir = tmp_path / "runtime"
    first = FileRuntimeLock(runtime_dir / "current_pipeline.lock")
    assert first.acquire() is True
    try:
        result = run_current_pipeline(
            data_root=tmp_path / "data",
            experiment_dir=tmp_path / "experiment",
            runtime_dir=runtime_dir,
        )
    finally:
        first.release()

    assert result["status"] == "runtime_busy"
    assert result["runtime_lock"]["acquired"] is False
    assert result["runtime_lock"]["holder"]["pid"] == os.getpid()
    assert result["poll"] is None
    assert result["paper_lifecycle"] is None
    assert result["cycle"] is None
    assert result["forward_outcomes"] is None


def test_current_pipeline_releases_lock_when_locked_body_raises(tmp_path, monkeypatch) -> None:
    module = importlib.import_module("stock_trading.live.run_current_pipeline")
    runtime_dir = tmp_path / "runtime"

    def _raise(**kwargs):
        del kwargs
        raise RuntimeError("synthetic pipeline failure")

    monkeypatch.setattr(module, "_run_current_pipeline_locked", _raise)
    with pytest.raises(RuntimeError, match="synthetic pipeline failure"):
        module.run_current_pipeline(runtime_dir=runtime_dir)

    replacement = FileRuntimeLock(runtime_dir / "current_pipeline.lock")
    assert replacement.acquire() is True
    replacement.release()
