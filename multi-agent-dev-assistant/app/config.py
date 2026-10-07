"""Central application settings loaded from environment variables and ``.env``."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    """Runtime configuration shared by the API, agents, and services."""

    llm_api_key: str
    llm_model: str
    llm_base_url: str
    llm_timeout_seconds: float
    llm_mock: bool
    max_retries: int
    log_level: str
    cors_allow_origins: tuple[str, ...]
    workspace_dir: Path


def get_settings() -> Settings:
    """Load and validate current settings, allowing tests to override the env."""
    load_dotenv()
    try:
        timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    except ValueError as exc:
        raise ValueError("LLM_TIMEOUT_SECONDS must be a number.") from exc
    if not 1 <= timeout_seconds <= 300:
        raise ValueError("LLM timeout must be between 1 and 300 seconds.")

    try:
        max_retries = int(os.getenv("MAX_RETRIES", "3"))
    except ValueError as exc:
        raise ValueError("MAX_RETRIES must be a non-negative integer.") from exc
    if max_retries < 0:
        raise ValueError("MAX_RETRIES must be a non-negative integer.")

    mock_value = os.getenv("LLM_MOCK", "false").strip().casefold()
    cors_origins = tuple(
        origin.strip()
        for origin in os.getenv(
            "CORS_ALLOW_ORIGINS",
            "http://localhost:3000,http://localhost:5173,"
            "http://127.0.0.1:3000,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    )
    workspace_value = Path(os.getenv("WORKSPACE_DIR", "workspace")).expanduser()
    workspace_dir = (
        workspace_value
        if workspace_value.is_absolute()
        else PROJECT_ROOT / workspace_value
    ).resolve()

    return Settings(
        llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
        llm_model=os.getenv("LLM_MODEL", "").strip(),
        llm_base_url=os.getenv("LLM_BASE_URL", "").strip().rstrip("/"),
        llm_timeout_seconds=timeout_seconds,
        llm_mock=mock_value in {"1", "true", "yes", "on"},
        max_retries=max_retries,
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        cors_allow_origins=cors_origins,
        workspace_dir=workspace_dir,
    )
