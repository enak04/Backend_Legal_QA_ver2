"""
tests/test_prompts.py - Tests for prompt loading and template rendering.
"""

import pytest

from src.prompts import load_prompt, render


def test_load_prompt():
    """Verify that load_prompt correctly reads prompt files and computes 8-character SHA256 version."""
    text, version = load_prompt("talker")
    assert isinstance(text, str)
    assert len(text) > 0
    assert isinstance(version, str)
    assert len(version) == 8


def test_load_prompt_missing_file():
    """Verify that load_prompt raises FileNotFoundError when a prompt does not exist."""
    with pytest.raises(FileNotFoundError):
        load_prompt("non_existent_prompt_xyz")


def test_render_success():
    """Verify that render replaces all placeholders cleanly."""
    template = "Hello, {{name}}! Welcome to {{place}}."
    result = render(template, name="Alice", place="Wonderland")
    assert result == "Hello, Alice! Welcome to Wonderland."


def test_render_missing_variable():
    """Verify that render raises ValueError if a placeholder is left unreplaced."""
    template = "Hello, {{name}}! Your code is {{code}}."
    with pytest.raises(ValueError, match="Missing variable"):
        render(template, name="Alice")


def test_render_extra_variable():
    """Verify that render raises ValueError if an extra variable is passed."""
    template = "Hello, {{name}}!"
    with pytest.raises(ValueError, match="Extra variable"):
        render(template, name="Alice", age=30)
