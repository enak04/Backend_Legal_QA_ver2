"""
src/talker.py - Prepares and executes the talker prompt to generate conversational replies.

Fills the talker prompt template with current context and placeholder values:
- phase: "COLLECTING"
- case_summary: "(nothing yet)"
- gaps: "Find out what happened and what they want help with"
- style_hint: "unknown"
- system_notes: ""
Wraps user messages in <user_message> tags for security and limits history to HISTORY_WINDOW messages.
"""

from collections.abc import AsyncGenerator
from typing import Any

from src.config import get_settings
from src.llm import BaseLLM, LLM
from src.prompts import load_prompt, render


def _format_conversation_history(messages: list[dict], window_size: int) -> list[dict]:
    """
    Limits messages to the last window_size items and wraps user text in <user_message> tags.
    """
    recent = messages[-window_size:] if len(messages) > window_size else messages
    formatted: list[dict] = []

    for msg in recent:
        role = msg.get("role", "user")
        text = msg.get("text", "")
        if role == "user":
            content = f"<user_message>\n{text}\n</user_message>"
        else:
            content = text
        formatted.append({"role": role, "content": content})

    return formatted


async def build_talker_reply(
    session: dict,
    messages: list[dict],
    llm: BaseLLM | None = None,
    stream: bool = False,
) -> str | AsyncGenerator[str, None]:
    """
    Loads and renders the talker prompt template, formats conversation history,
    and calls the SMART_MODEL tier via the LLM client.

    If stream=True, returns an async generator yielding text chunks.
    Otherwise, returns the complete text string.
    """
    settings = get_settings()
    client = llm or LLM()

    # Load and render talker system prompt template
    template_text, _ = load_prompt("talker")
    rendered_system = render(
        template_text,
        phase=session.get("status", "COLLECTING"),
        language=session.get("language", "en"),
        case_summary="(nothing yet)",
        gaps="Find out what happened and what they want help with",
        style_hint="unknown",
        system_notes="",
        max_questions=settings.max_questions_per_turn,
    )

    formatted_history = _format_conversation_history(
        messages=messages,
        window_size=settings.history_window,
    )

    session_id = session.get("_id")

    if stream:
        return client.stream(
            prompt_name="talker",
            system=rendered_system,
            messages=formatted_history,
            tier="smart",
            session_id=session_id,
        )
    else:
        return await client.text(
            prompt_name="talker",
            system=rendered_system,
            messages=formatted_history,
            tier="smart",
            session_id=session_id,
        )
