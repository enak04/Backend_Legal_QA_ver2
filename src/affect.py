"""
src/affect.py - Emotional affect and crisis signal detector.

Implements:
1. read_affect: Calls the fast LLM tier to analyze emotional state (distress, anger,
   confusion, shame, urgency, and crisis flags).
2. affect_to_style_hint: Pure code heuristics converting affect readings into style
   guidance for the talker prompt. Affect signals influence tone only; they never
   alter case facts.
"""

from src.llm import BaseLLM
from src.prompts import load_prompt, render
from src.schemas import AffectReading


def _format_recent_messages(messages: list[dict]) -> str:
    """Formats recent conversation turns for the affect prompt."""
    if not messages:
        return "(no previous messages)"

    lines: list[str] = []
    for m in messages:
        sender = "User" if m.get("role") == "user" else "Assistant"
        lines.append(f"{sender}: {m.get('text', '')}")
    return "\n".join(lines)


async def read_affect(
    session: dict,
    new_message: str,
    llm: BaseLLM,
    recent_messages: list[dict] | None = None,
) -> AffectReading:
    """
    Executes the affect LLM call using FAST_MODEL to detect emotional levels and crisis flags.
    """
    recent_formatted = _format_recent_messages(recent_messages or [])
    template, _ = load_prompt("affect")
    rendered_system = render(
        template,
        recent_messages=recent_formatted,
        new_message=new_message,
    )

    return await llm.json(
        prompt_name="affect",
        system=rendered_system,
        messages=[],
        schema=AffectReading,
        tier="fast",
        session_id=session.get("_id"),
    )


def affect_to_style_hint(affect: AffectReading) -> str:
    """
    Transforms affect readings into tone instructions for the conversational talker.
    Affect signals shape conversational tone only and never become case facts.
    """
    hints: list[str] = []

    # Distress guidance
    if affect.distress >= 2:
        d_hint = (
            "The person is distressed. Acknowledge their feelings first in one or two warm sentences. "
            "Ask only ONE gentle, easy question. Keep it very short."
        )
        if affect.distress == 3:
            d_hint += " Do not ask for dates, documents or amounts this turn."
        hints.append(d_hint)

    # Confusion guidance
    if affect.confusion >= 2:
        hints.append(
            "The person is confused. Explain why you are asking and give a simple example answer."
        )

    # Anger guidance
    if affect.anger >= 2:
        hints.append(
            "The person is angry. Let them feel heard in one sentence, do not argue, then steer to the facts."
        )

    # Shame guidance
    if affect.shame >= 2:
        hints.append(
            "The person feels ashamed. Be non-judgmental and normalize asking for help."
        )

    # Base style
    base_text = " ".join(hints) if hints else "Neutral and friendly."

    # Append style note if provided by the model
    if affect.style_note:
        note = affect.style_note.strip()
        if note:
            base_text = f"{base_text} {note}"

    return base_text.strip()
