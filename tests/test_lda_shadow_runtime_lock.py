from stock_trading.live.run_current_lda_shadow import run_current_lda_shadow
from stock_trading.live.runtime_lock import FileRuntimeLock


def test_lda_shadow_fails_busy_before_external_work(tmp_path) -> None:
    runtime_dir = tmp_path / "runtime"
    holder = FileRuntimeLock(runtime_dir / "current_pipeline.lock")
    assert holder.acquire() is True
    try:
        result = run_current_lda_shadow(
            data_root=tmp_path / "data",
            experiment_dir=tmp_path / "experiment",
            runtime_dir=runtime_dir,
        )
    finally:
        holder.release()

    assert result["status"] == "runtime_busy"
    assert result["authority"] == "shadow_only_no_paper"
    assert result["evidence_source"] == "lda_shadow"
    assert result["runtime_lock"]["acquired"] is False
    assert result["runtime_lock"]["holder"] is not None
    assert not (runtime_dir / "lda_shadow").exists()
