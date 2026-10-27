"""
tests/conftest.py - Pytest configuration and environment fixtures for tests.
"""

import os
import pytest


@pytest.fixture(autouse=True, scope="session")
def setup_test_environment():
    """Sets environment variables so tests run safely without real credentials or external services."""
    os.environ["PROVIDER"] = "fake"
    os.environ["LLM_API_KEY"] = "mock_test_key_for_tests"
    os.environ["FAST_MODEL"] = "gpt-4o-mini"
    os.environ["SMART_MODEL"] = "gpt-4o"
    os.environ["MONGODB_URI"] = "mongodb://localhost:27017"
    os.environ["DB_NAME"] = "test_legal_intake_bot"
    os.environ["USE_MOCK_SERVICE"] = "true"
    os.environ["APP_API_KEY"] = "test_app_key"
    os.environ["ALLOWED_ORIGINS"] = "*"
