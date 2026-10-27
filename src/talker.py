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

from src.config import get_settings
from src.llm import LLM, BaseLLM
from src.prompts import load_prompt, render
from src.schemas import Gap


def format_gaps_for_talker(gaps: list[Gap] | list[dict] | str | None) -> str:
    """
    Formats the top gaps as a numbered text block for the talker prompt:
    1. Topic: <topic>
       Why it matters: <why_it_matters>
       Example answer: <example_answer>
    """
    if not gaps:
        return "(no gaps identified)"
    if isinstance(gaps, str):
        return gaps

    blocks: list[str] = []
    for idx, g in enumerate(gaps, 1):
        if isinstance(g, dict):
            topic = g.get("topic", "")
            why = g.get("why_it_matters", "")
            example = g.get("example_answer", "")
        else:
            topic = getattr(g, "topic", "")
            why = getattr(g, "why_it_matters", "")
            example = getattr(g, "example_answer", "")

        lines = [f"{idx}. Topic: {topic}"]
        if why:
            lines.append(f"   Why it matters: {why}")
        if example:
            lines.append(f"   Example answer: {example}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


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


from typing import Literal, overload


@overload
async def build_talker_reply(
    session: dict,
    messages: list[dict],
    llm: BaseLLM | None = None,
    stream: Literal[False] = False,
    case_summary: str | None = None,
    gaps: list[Gap] | list[dict] | str | None = None,
    language: str | None = None,
    style_hint: str | None = None,
    system_notes: str | None = None,
) -> str: ...


@overload
async def build_talker_reply(
    session: dict,
    messages: list[dict],
    llm: BaseLLM | None = None,
    stream: Literal[True] = ...,
    case_summary: str | None = None,
    gaps: list[Gap] | list[dict] | str | None = None,
    language: str | None = None,
    style_hint: str | None = None,
    system_notes: str | None = None,
) -> AsyncGenerator[str, None]: ...


@overload
async def build_talker_reply(
    session: dict,
    messages: list[dict],
    llm: BaseLLM | None = None,
    stream: bool = False,
    case_summary: str | None = None,
    gaps: list[Gap] | list[dict] | str | None = None,
    language: str | None = None,
    style_hint: str | None = None,
    system_notes: str | None = None,
) -> str | AsyncGenerator[str, None]: ...


async def build_talker_reply(
    session: dict,
    messages: list[dict],
    llm: BaseLLM | None = None,
    stream: bool = False,
    case_summary: str | None = None,
    gaps: list[Gap] | list[dict] | str | None = None,
    language: str | None = None,
    style_hint: str | None = None,
    system_notes: str | None = None,
) -> str | AsyncGenerator[str, None]:
    """
    Loads and renders the talker prompt template, formats conversation history,
    and calls the SMART_MODEL tier via the LLM client.

    If stream=True, returns an async generator yielding text chunks.
    Otherwise, returns the complete text string.
    """
    settings = get_settings()
    client = llm or LLM()

    # Determine summary, language, style hint, system notes, and gaps block
    summary_text = case_summary or "(nothing yet)"
    lang_text = language or session.get("language", "en")
    hint_text = style_hint or "unknown"
    notes_text = system_notes or ""
    gaps_text = format_gaps_for_talker(gaps)

    # Load and render talker system prompt template
    template_text, _ = load_prompt("talker")
    rendered_system = render(
        template_text,
        phase=session.get("status", "COLLECTING"),
        language=lang_text,
        case_summary=summary_text,
        gaps=gaps_text,
        style_hint=hint_text,
        system_notes=notes_text,
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

