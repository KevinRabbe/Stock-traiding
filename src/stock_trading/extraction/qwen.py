import json
import os
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx

from stock_trading.core import SemanticAnnotation

from .semantic import ALLOWED_TOPICS, SemanticResult


class FileSemanticCache:
    """Content-addressed immutable cache for validated semantic results."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def get(self, key: str) -> SemanticResult | None:
        path = self._path(key)
        if not path.exists():
            return None
        return SemanticResult.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, key: str, result: SemanticResult) -> Path:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = result.model_dump_json(indent=2).encode("utf-8")
        if path.exists():
            existing = path.read_bytes()
            if existing != content:
                raise ValueError(f"semantic cache collision at {path}")
            return path

        with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
            temp.write(content)
            temp.flush()
            os.fsync(temp.fileno())
            temporary_path = Path(temp.name)
        os.replace(temporary_path, path)
        return path

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"


class QwenSemanticExtractor:
    """Strict semantic extraction through a local OpenAI-compatible Qwen service."""

    def __init__(
        self,
        *,
        cache: FileSemanticCache,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "Qwen/Qwen3.5-4B",
        extractor_version: str = "semantic-v1",
        timeout_seconds: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.cache = cache
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.extractor_version = extractor_version
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout_seconds)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "QwenSemanticExtractor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def extract(self, source_text: str, *, context: str = "") -> SemanticAnnotation:
        text = source_text.strip()
        if not text:
            raise ValueError("source_text must not be empty")

        key = self.cache_key(text, context=context)
        cached = self.cache.get(key)
        if cached is not None:
            return cached.to_annotation(
                model=self.model,
                extractor_version=self.extractor_version,
            )

        last_error: Exception | None = None
        for _attempt in range(2):
            try:
                result = self._infer(text, context=context)
                self.cache.put(key, result)
                return result.to_annotation(
                    model=self.model,
                    extractor_version=self.extractor_version,
                )
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc

        raise ValueError("Qwen semantic extraction failed validation twice") from last_error

    def cache_key(self, source_text: str, *, context: str = "") -> str:
        material = json.dumps(
            {
                "source_text": source_text.strip(),
                "context": context.strip(),
                "model": self.model,
                "extractor_version": self.extractor_version,
                "schema_version": "semantic-v1",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(material.encode("utf-8")).hexdigest()

    def _infer(self, source_text: str, *, context: str) -> SemanticResult:
        response = self._client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": self._system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": self._user_prompt(source_text, context=context),
                    },
                ],
            },
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Qwen response content must be text JSON")
        return SemanticResult.model_validate(json.loads(content))

    @staticmethod
    def _user_prompt(source_text: str, *, context: str) -> str:
        return (
            "Context:\n"
            f"{context.strip() or 'none'}\n\n"
            "Authoritative source text:\n"
            f"{source_text.strip()}"
        )

    @staticmethod
    def _system_prompt() -> str:
        topics = ", ".join(sorted(ALLOWED_TOPICS))
        return (
            "Extract trading-relevant semantics from the supplied authoritative text. "
            "Do not invent facts and do not infer any future price movement. Return only one JSON object "
            "with exactly these keys: topics, direction, novelty, importance, company_relevance, "
            "policy_relevance, confidence. topics must be an array using only the controlled topic IDs below. "
            "direction must be one of positive, negative, mixed, neutral, unknown. Score fields are numbers "
            "from 0 to 1 or null, except confidence which must always be 0 to 1. If uncertain, lower confidence.\n\n"
            f"Allowed topics: {topics}"
        )
