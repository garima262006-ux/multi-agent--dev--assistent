from pathlib import Path

import pytest

from app.config import get_settings


def test_settings_load_runtime_values_from_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1/")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("LLM_MOCK", "true")
    monkeypatch.setenv("MAX_RETRIES", "4")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://one.example, https://two.example")
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "generated"))

    settings = get_settings()

    assert settings.llm_api_key == "test-key"
    assert settings.llm_model == "test-model"
    assert settings.llm_base_url == "https://llm.example/v1"
    assert settings.llm_timeout_seconds == 12.5
    assert settings.llm_mock is True
    assert settings.max_retries == 4
    assert settings.log_level == "DEBUG"
    assert settings.cors_allow_origins == (
        "https://one.example",
        "https://two.example",
    )
    assert settings.workspace_dir == (tmp_path / "generated").resolve()


def test_settings_reject_invalid_retry_count(monkeypatch) -> None:
    monkeypatch.setenv("MAX_RETRIES", "-1")

    with pytest.raises(ValueError, match="MAX_RETRIES"):
        get_settings()
