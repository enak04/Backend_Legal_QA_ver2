"""
src/understand.py - Understand / note-taker step.

Builds the prompt from prompts/extractor.txt and calls the fast LLM tier
to extract intent, language, and case patch (UnderstandOutput).
"""

from datetime import datetime, timezone

from src.case_ops import render_case_json_for_prompts
from src.llm import BaseLLM
from src.prompts import load_prompt, render
from src.schemas import CaseFile, UnderstandOutput


def _format_recent_messages_for_prompt(messages: list[dict]) -> str:
    """Formats recent conversation turns for the extractor prompt."""
    if not messages:
        return "(no previous messages)"

    lines: list[str] = []
    for m in messages:
        sender = "User" if m.get("role") == "user" else "Assistant"
        lines.append(f"{sender}: {m.get('text', '')}")
    return "\n".join(lines)


async def understand(
    session: dict,
    new_message: str,
    llm: BaseLLM,
    recent_messages: list[dict] | None = None,
) -> UnderstandOutput:
    """
    Extracts user intent, language, and structured case patches from the newest user message.
    """
    # Load case file from session
    raw_case = session.get("case_file", {})
    if isinstance(raw_case, dict) and not raw_case:
        case_obj = CaseFile()
    else:
        case_obj = CaseFile.model_validate(raw_case)

    case_json_str = render_case_json_for_prompts(case_obj)
    recent_formatted = _format_recent_messages_for_prompt(recent_messages or [])
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Load and render extractor prompt
    template, _ = load_prompt("extractor")
    rendered_system = render(
        template,
        today=today_str,
        phase=session.get("status", "COLLECTING"),
        case_json=case_json_str,
        recent_messages=recent_formatted,
        new_message=new_message,
    )

    # Call fast LLM tier expecting UnderstandOutput
    return await llm.json(
        prompt_name="extractor",
        system=rendered_system,
        messages=[],
        schema=UnderstandOutput,
        tier="fast",
        session_id=session.get("_id"),
    )
