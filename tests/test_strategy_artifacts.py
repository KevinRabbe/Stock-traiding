from __future__ import annotations

import pytest

from stock_trading.engine.artifacts import (
    build_strategy_artifact_manifest,
    load_strategy_artifact_manifest,
    verify_strategy_artifact_manifest,
    write_strategy_artifact_manifest,
)


def test_strategy_artifact_manifest_is_deterministic_and_detects_mutation(tmp_path) -> None:
    root = tmp_path / "models"
    (root / "20d").mkdir(parents=True)
    (root / "5d").mkdir(parents=True)
    (root / "20d" / "model.txt").write_text("twenty", encoding="utf-8")
    (root / "5d" / "model.txt").write_text("five", encoding="utf-8")

    first = build_strategy_artifact_manifest("v5", root)
    second = build_strategy_artifact_manifest("v5", root)

    assert first.manifest_sha256 == second.manifest_sha256
    assert [item.path for item in first.files] == ["20d/model.txt", "5d/model.txt"]
    verify_strategy_artifact_manifest(first)

    path = write_strategy_artifact_manifest(first, tmp_path / "manifest.json")
    loaded = load_strategy_artifact_manifest(path)
    assert loaded == first
    verify_strategy_artifact_manifest(loaded)

    (root / "5d" / "model.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        verify_strategy_artifact_manifest(loaded)


def test_strategy_artifact_manifest_requires_files(tmp_path) -> None:
    root = tmp_path / "empty"
    root.mkdir()

    with pytest.raises(ValueError, match="contains no files"):
        build_strategy_artifact_manifest("v5", root)
