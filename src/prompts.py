"""
src/prompts.py - Prompt file loader and template renderer.

Handles:
1. load_prompt(name): reads prompts/<name>.txt and returns (text, version),
   where version is the first 8 characters of sha256 of the text.
2. render(template, **vars): replaces {{var}} placeholders.
   Raises ValueError if any {{...}} placeholder is left unreplaced or if
   any extra variable is provided that the template does not use.
"""

import hashlib
import re
from pathlib import Path
from typing import Any

# Base path to the prompts directory
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str, prompts_dir: Path | None = None) -> tuple[str, str]:
    """
    Reads prompts/<name>.txt and returns (text, version).
    Version is the first 8 hex characters of the SHA-256 hash of the prompt text.
    """
    directory = prompts_dir or PROMPTS_DIR
    filename = f"{name}.txt" if not name.endswith(".txt") else name
    filepath = directory / filename

    if not filepath.is_file():
        raise FileNotFoundError(f"Prompt file not found at: {filepath}")

    text = filepath.read_text(encoding="utf-8")
    version = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return text, version


def render(template: str, **vars: Any) -> str:
    """
    Replaces {{var}} placeholders in template with corresponding values in vars.

    Raises:
        ValueError: If extra variables are passed that the template does not use,
                    or if any {{...}} placeholder is left unreplaced.
    """
    # Find all placeholders in the template (e.g. {{case_summary}})
    placeholders = set(re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", template))
    passed_keys = set(vars.keys())

    # Check for extra variables passed that the template does not use
    extra_keys = passed_keys - placeholders
    if extra_keys:
        raise ValueError(
            f"Extra variable(s) passed that template does not use: {sorted(list(extra_keys))}"
        )

    # Perform text replacement
    rendered_text = template
    for key, value in vars.items():
        rendered_text = rendered_text.replace(f"{{{{{key}}}}}", str(value))

    # Check if any placeholders were left unreplaced
    remaining = set(re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", rendered_text))
    if remaining:
        raise ValueError(
            f"Missing variable(s) for placeholders in template: {sorted(list(remaining))}"
        )

    return rendered_text
