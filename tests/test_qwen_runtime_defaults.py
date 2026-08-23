from stock_trading.extraction import FileSemanticCache, QwenSemanticExtractor
from stock_trading.extraction.qwen import DEFAULT_QWEN_BASE_URL, DEFAULT_QWEN_MODEL
from stock_trading.live.run_current_lda_shadow import (
    DEFAULT_QWEN_BASE_URL as LDA_DEFAULT_QWEN_BASE_URL,
    DEFAULT_QWEN_MODEL as LDA_DEFAULT_QWEN_MODEL,
)


def test_qwen_defaults_match_validated_ollama_runtime(tmp_path) -> None:
    extractor = QwenSemanticExtractor(cache=FileSemanticCache(tmp_path), client=object())  # type: ignore[arg-type]
    assert DEFAULT_QWEN_BASE_URL == "http://127.0.0.1:11434/v1"
    assert DEFAULT_QWEN_MODEL == "qwen3.5:4b-q8_0"
    assert extractor.base_url == DEFAULT_QWEN_BASE_URL
    assert extractor.model == DEFAULT_QWEN_MODEL
    assert LDA_DEFAULT_QWEN_BASE_URL == DEFAULT_QWEN_BASE_URL
    assert LDA_DEFAULT_QWEN_MODEL == DEFAULT_QWEN_MODEL
