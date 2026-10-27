"""
tests/test_config.py - Tests for configuration loading and validation.
"""

import os
from unittest.mock import patch
import pytest

from src.config import load_settings


def test_load_settings_success():
    """Verify loading valid settings returns a strongly typed Settings instance."""
    with patch.dict(os.environ, {
        "PROVIDER": "openai",
        "LLM_API_KEY": "test-key-123",
        "FAST_MODEL": "gpt-4o-mini",
        "SMART_MODEL": "gpt-4o",
        "MONGODB_URI": "mongodb://localhost:27017",
        "DB_NAME": "legal_intake_bot",
    }):
        settings = load_settings()
        assert settings.provider == "openai"
        assert settings.llm_api_key == "test-key-123"
        assert settings.fast_model == "gpt-4o-mini"
        assert settings.smart_model == "gpt-4o"
        assert settings.max_questions_per_turn == 2


def test_load_settings_missing_api_key_fails():
    """Verify that missing LLM_API_KEY raises a clear ValueError when provider is not fake."""
    with patch.dict(os.environ, {
        "PROVIDER": "openai",
        "LLM_API_KEY": "",
    }, clear=True):
        with pytest.raises(ValueError, match="Missing required configuration key: 'LLM_API_KEY'"):
            load_settings()


def test_load_settings_missing_fast_model_fails():
    """Verify that missing FAST_MODEL raises a clear ValueError."""
    with patch.dict(os.environ, {
        "PROVIDER": "fake",
        "FAST_MODEL": "",
    }, clear=True):
        with pytest.raises(ValueError, match="Missing required configuration key: 'FAST_MODEL'"):
            load_settings()
