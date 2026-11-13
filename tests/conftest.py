"""
tests/conftest.py - Pytest configuration and environment fixtures for tests.

The test environment is set at MODULE IMPORT time (not in a fixture), because test modules
such as test_app.py import app.py during collection, and app.py calls get_settings() at import.
Setting the variables any later would let the cached settings pick up the developer's real .env
(e.g. USE_MOCK_SERVICE=false and a production SERVICE_URL), sending test traffic to real services.
"""

import os

import pytest

TEST_ENV = {
    "PROVIDER": "fake",
    "LLM_API_KEY": "mock_test_key_for_tests",
    "FAST_MODEL": "gpt-4o-mini",
    "SMART_MODEL": "gpt-4o",
    "MONGODB_URI": "mongodb://localhost:27017",
    "DB_NAME": "test_legal_intake_bot",
    "USE_MOCK_SERVICE": "true",
    "SERVICE_URL": "http://localhost:8001/legal-answer",
    "SERVICE_API_KEY": "mock_service_key",
    "APP_API_KEY": "test_app_key",
    "ALLOWED_ORIGINS": "*",
    "DIRECT_SERVICE_ANSWER": "false",
}

# Applied before any test module (and therefore app.py / src.config) is imported.
os.environ.update(TEST_ENV)


@pytest.fixture(autouse=True, scope="session")
def setup_test_environment():
    """Re-applies the test environment and clears any cached settings loaded before it."""
    os.environ.update(TEST_ENV)
    import src.config

    src.config._settings = None
    yield
