from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


LOCAL_SECRETS_RELATIVE_PATH = Path("local") / "secrets.json"


@dataclass(frozen=True, slots=True)
class TiingoCredentials:
    token: str
    email: str | None
    source: str


def load_tiingo_credentials(data_root: Path) -> TiingoCredentials:
    """Load Tiingo credentials without ever requiring secrets in source control.

    TIINGO_API_TOKEN wins when present so CI and temporary shells keep working.
    Otherwise the local-only data/local/secrets.json file is used.
    """

    env_token = os.environ.get("TIINGO_API_TOKEN", "").strip()
    env_email = os.environ.get("TIINGO_EMAIL", "").strip() or None
    if env_token:
        return TiingoCredentials(env_token, env_email, "environment")

    secrets_path = data_root / LOCAL_SECRETS_RELATIVE_PATH
    try:
        payload = json.loads(secrets_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(
            "TIINGO_API_TOKEN is not set and local Tiingo secrets were not found at "
            f"{secrets_path}. Copy data/local/secrets.example.json to secrets.json and "
            "fill in your local credentials."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in local Tiingo secrets file: {secrets_path}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Local Tiingo secrets must be a JSON object: {secrets_path}")

    token = str(payload.get("tiingo_api_token", "")).strip()
    email = str(payload.get("tiingo_email", "")).strip() or None
    if not token:
        raise RuntimeError(f"tiingo_api_token is missing or empty in {secrets_path}")

    return TiingoCredentials(token, email, str(secrets_path))
