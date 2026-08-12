from hashlib import sha256

from .enums import EventType, Source


def deterministic_event_id(
    source: Source,
    source_record_id: str,
    event_type: EventType,
    event_index: int = 0,
) -> str:
    """Create a stable event ID from source identity and event position."""

    if event_index < 0:
        raise ValueError("event_index must be >= 0")
    canonical = "\x1f".join(
        (source.value, source_record_id, event_type.value, str(event_index))
    )
    return f"evt_{sha256(canonical.encode('utf-8')).hexdigest()[:32]}"


def content_sha256(content: bytes | str) -> str:
    """Hash raw source content using a canonical UTF-8 representation for text."""

    payload = content if isinstance(content, bytes) else content.encode("utf-8")
    return sha256(payload).hexdigest()


def raw_artifact_id(content_hash: str) -> str:
    """Create a stable raw artifact ID from a validated SHA-256 digest."""

    if len(content_hash) != 64 or any(c not in "0123456789abcdef" for c in content_hash):
        raise ValueError("content_hash must be a lowercase SHA-256 hex digest")
    return f"raw_{content_hash[:32]}"
