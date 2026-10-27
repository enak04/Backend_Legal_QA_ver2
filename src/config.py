"""
src/config.py - Loads and validates environment variables.

Loads configuration from .env using python-dotenv into a typed Settings dataclass.
Fails with a clear, beginner-friendly error message if any required configuration key is missing.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Strongly typed application configuration settings."""
    provider: str
    llm_api_key: str
    fast_model: str
    smart_model: str
    mongodb_uri: str
    db_name: str
    use_mock_service: bool
    service_url: str
    service_api_key: str
    max_questions_per_turn: int
    history_window: int
    max_collecting_turns: int
    session_ttl_days: int
    debug: bool
    app_api_key: str = "test_app_key"
    allowed_origins: tuple[str, ...] = ("*",)


def _str_to_bool(value: str | None, default: bool = False) -> bool:
    """Helper to convert a string environment variable into a boolean."""
    if value is None:
        return default
    return value.strip().lower() in ("true", "1", "yes", "y", "t")


def load_settings(env_path: str | Path | None = None) -> Settings:
    """
    Loads environment variables from .env and returns a typed Settings object.
    Raises ValueError with a helpful message if required settings are missing.
    """
    if env_path is not None:
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv(override=False)

    # Check for required keys
    # LLM_API_KEY is required for real LLM operations, but we allow fake/test mode
    provider = os.getenv("PROVIDER", "openai").strip()
    llm_api_key = os.getenv("LLM_API_KEY", "").strip()

    # If provider is not fake, LLM_API_KEY must not be empty
    if not llm_api_key and provider != "fake":
        raise ValueError(
            "Missing required configuration key: 'LLM_API_KEY'. "
            "Please add LLM_API_KEY=your_key to your .env file."
        )

    fast_model = os.getenv("FAST_MODEL", "gpt-4o-mini").strip()
    if not fast_model:
        raise ValueError("Missing required configuration key: 'FAST_MODEL'.")

    smart_model = os.getenv("SMART_MODEL", "gpt-4o").strip()
    if not smart_model:
        raise ValueError("Missing required configuration key: 'SMART_MODEL'.")

    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017").strip()
    if not mongodb_uri:
        raise ValueError("Missing required configuration key: 'MONGODB_URI'.")

    db_name = os.getenv("DB_NAME", "legal_intake_bot").strip()
    if not db_name:
        raise ValueError("Missing required configuration key: 'DB_NAME'.")

    return Settings(
        provider=provider,
        llm_api_key=llm_api_key,
        fast_model=fast_model,
        smart_model=smart_model,
        mongodb_uri=mongodb_uri,
        db_name=db_name,
        use_mock_service=_str_to_bool(os.getenv("USE_MOCK_SERVICE"), default=True),
        service_url=os.getenv("SERVICE_URL", "http://localhost:8001/legal-answer").strip(),
        service_api_key=os.getenv("SERVICE_API_KEY", "mock_service_key").strip(),
        max_questions_per_turn=int(os.getenv("MAX_QUESTIONS_PER_TURN", "2")),
        history_window=int(os.getenv("HISTORY_WINDOW", "10")),
        max_collecting_turns=int(os.getenv("MAX_COLLECTING_TURNS", "14")),
        session_ttl_days=int(os.getenv("SESSION_TTL_DAYS", "30")),
        debug=_str_to_bool(os.getenv("DEBUG"), default=False),
        app_api_key=os.getenv("APP_API_KEY", "test_app_key").strip() or "test_app_key",
        allowed_origins=tuple(
            o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
        ),
    )


# Cached settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Returns the cached Settings object, loading it on first call."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings
