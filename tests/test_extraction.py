import json

import httpx
import pytest
from pydantic import ValidationError

from stock_trading.extraction import FileSemanticCache, QwenSemanticExtractor, SemanticResult


def test_semantic_schema_rejects_uncontrolled_topics() -> None:
    with pytest.raises(ValidationError, match="unknown semantic topics"):
        SemanticResult(
            topics=("MADE.UP.TOPIC",),
            direction="positive",
            novelty=0.5,
            importance=0.5,
            company_relevance=0.5,
            policy_relevance=0.5,
            confidence=0.8,
        )


def test_qwen_extraction_is_validated_and_cached(tmp_path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        assert body["model"] == "Qwen/Qwen3.5-4B"
        assert body["temperature"] == 0
        result = {
            "topics": ["DEFENSE.MISSILES", "GOVERNMENT.PROCUREMENT"],
            "direction": "positive",
            "novelty": 0.7,
            "importance": 0.9,
            "company_relevance": 0.95,
            "policy_relevance": 0.8,
            "confidence": 0.92,
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(result)}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    extractor = QwenSemanticExtractor(
        cache=FileSemanticCache(tmp_path / "semantic"),
        client=client,
    )

    first = extractor.extract(
        "Award modification for missile defense interceptors.",
        context="government contract",
    )
    second = extractor.extract(
        "Award modification for missile defense interceptors.",
        context="government contract",
    )

    assert calls == 1
    assert first == second
    assert first.importance == 0.9
    assert first.model == "Qwen/Qwen3.5-4B"
    assert first.extractor_version == "semantic-v1"


def test_qwen_retries_invalid_json_once(tmp_path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            content = "not json"
        else:
            content = json.dumps(
                {
                    "topics": ["TECH.SEMICONDUCTORS"],
                    "direction": "neutral",
                    "novelty": None,
                    "importance": 0.4,
                    "company_relevance": 0.9,
                    "policy_relevance": 0.7,
                    "confidence": 0.6,
                }
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    extractor = QwenSemanticExtractor(
        cache=FileSemanticCache(tmp_path / "semantic"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    annotation = extractor.extract("Semiconductor export control implementation.")

    assert calls == 2
    assert annotation.topics == ("TECH.SEMICONDUCTORS",)
