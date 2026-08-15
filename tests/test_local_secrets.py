import json

import pytest

from stock_trading.local_secrets import load_tiingo_credentials


def test_tiingo_credentials_prefer_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TIINGO_API_TOKEN", " env-token ")
    monkeypatch.setenv("TIINGO_EMAIL", " env@example.com ")
    secrets_path = tmp_path / "local" / "secrets.json"
    secrets_path.parent.mkdir(parents=True)
    secrets_path.write_text(
        json.dumps(
            {
                "tiingo_api_token": "file-token",
                "tiingo_email": "file@example.com",
            }
        ),
        encoding="utf-8",
    )

    credentials = load_tiingo_credentials(tmp_path)

    assert credentials.token == "env-token"
    assert credentials.email == "env@example.com"
    assert credentials.source == "environment"


def test_tiingo_credentials_fall_back_to_local_file(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TIINGO_API_TOKEN", raising=False)
    monkeypatch.delenv("TIINGO_EMAIL", raising=False)
    secrets_path = tmp_path / "local" / "secrets.json"
    secrets_path.parent.mkdir(parents=True)
    secrets_path.write_text(
        json.dumps(
            {
                "tiingo_api_token": " local-token ",
                "tiingo_email": " local@example.com ",
            }
        ),
        encoding="utf-8",
    )

    credentials = load_tiingo_credentials(tmp_path)

    assert credentials.token == "local-token"
    assert credentials.email == "local@example.com"
    assert credentials.source == str(secrets_path)


def test_tiingo_credentials_explain_missing_local_file(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TIINGO_API_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="local Tiingo secrets were not found"):
        load_tiingo_credentials(tmp_path)


def test_tiingo_credentials_reject_empty_token(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TIINGO_API_TOKEN", raising=False)
    secrets_path = tmp_path / "local" / "secrets.json"
    secrets_path.parent.mkdir(parents=True)
    secrets_path.write_text(
        json.dumps({"tiingo_api_token": "", "tiingo_email": "user@example.com"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="tiingo_api_token is missing or empty"):
        load_tiingo_credentials(tmp_path)
