"""
src/presenter.py - Formats and presents the external service answer in friendly chat.

Retells the legal service's answer in the shared conversational voice (prompts/voice.txt),
in the person's language. All legal content must come from the service answer.

Modes:
- initial:  first presentation of the main case answer
- followup: questions/reactions about the stored answer
- quick:    a mid-conversation question answered by the service, ending with the next intake question

Every mode is guarded using:
1. Pure code checks: rejects section/article numbers, case citations, currency amounts,
   or dates not present in the source answer.
2. Faithfulness LLM check (FAST tier) calling prompts/faithfulness.txt.
If problems are found, regenerates ONCE with problem instructions.
If it still fails, falls back to a plain template (initial/quick) or an honest
"I don't want to guess" reply (followup).
Records which path was used in session['last_presenter_path'] ('primary', 'regenerated', 'fallback').
The short disclaimer is included only with the first legal answer of a session (session['disclaimer_shown']).
"""

import logging
import re
import sys

from src.case_ops import render_case_summary
from src.config import get_settings
from src.llm import LLM, BaseLLM
from src.messages import DISCLAIMER_SHORT, FOLLOWUP_FALLBACK_REPLY
from src.prompts import load_prompt, render
from src.schemas import CaseFile, FaithfulnessReport, ServiceResponse

logger = logging.getLogger(__name__)

# Regex patterns for code guard check
SECTION_REGEX = re.compile(r"\b(?:section|sec\.?|article|art\.?|clause|§)\s*\d+[a-zA-Z0-9.-]*\b", re.IGNORECASE)
CASE_NAME_REGEX = re.compile(r"\b[A-Z][a-zA-Z0-9_'\-]+\s+(?:v\.|vs\.?)\s+[A-Z][a-zA-Z0-9_'\-]+\b")
CURRENCY_REGEX = re.compile(r"(?:[\$\€\£\₹]|CAD|USD|INR)\s*\d+(?:,\d{3})*(?:\.\d+)?|\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:dollars|euros|pounds|rupees)\b", re.IGNORECASE)
DATE_REGEX = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember))\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}|\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)


def check_code_guard(presenter_text: str, source_answer: str, allowed_context: str = "") -> list[str]:
    """
    Checks that any section/article numbers, case citations ('v.' or 'vs'),
    currency amounts, or dates in presenter_text also appear in source_answer
    (or in allowed_context, i.e. the person's own case facts, which may be repeated back).
    Returns a list of violation descriptions.
    """
    violations: list[str] = []
    source_lower = f"{source_answer}\n{allowed_context}".lower()

    # Section / Article numbers
    for match in SECTION_REGEX.findall(presenter_text):
        if match.lower() not in source_lower:
            violations.append(f"Section/article '{match}' is not in source answer")

    # Case citations
    for match in CASE_NAME_REGEX.findall(presenter_text):
        if match.lower() not in source_lower:
            violations.append(f"Case citation '{match}' is not in source answer")

    # Currency amounts
    for match in CURRENCY_REGEX.findall(presenter_text):
        if match.lower() not in source_lower:
            violations.append(f"Currency amount '{match}' is not in source answer")

    # Dates
    for match in DATE_REGEX.findall(presenter_text):
        if match.lower() not in source_lower:
            violations.append(f"Date '{match}' is not in source answer")

    return violations


PRESENTER_TEMPERATURE = 0.6

FALLBACK_INTRO = "Here's what the legal service came back with on your situation."


def fallback_template(
    service_response: ServiceResponse,
    disclaimer: str = DISCLAIMER_SHORT,
    intro: str = FALLBACK_INTRO,
) -> str:
    """
    Plain fallback that passes the service answer on verbatim, without template headings,
    followed by the disclaimer when one is given.
    """
    parts: list[str] = [intro] if intro else []
    if service_response.sections:
        for body_text in service_response.sections.values():
            if body_text and body_text.strip():
                parts.append(body_text.strip())
    elif service_response.answer:
        parts.append(service_response.answer.strip())
    else:
        parts.append("The answer came back empty, though. Want me to try again?")

    if disclaimer and disclaimer.strip():
        parts.append(disclaimer.strip())
    return "\n\n".join(parts)


async def check_faithfulness(
    service_answer: str,
    presenter_text: str,
    llm: BaseLLM,
    session_id: str | None = None,
    mode: str = "initial",
    case_summary: str = "",
) -> FaithfulnessReport:
    """Executes the FAST_MODEL faithfulness prompt check."""
    template_text, _ = load_prompt("faithfulness")
    rendered_system = render(
        template_text,
        service_answer=service_answer,
        presenter_text=presenter_text,
        mode=mode,
        case_summary=case_summary or "(none)",
    )
    return await llm.json(
        prompt_name="faithfulness",
        system=rendered_system,
        messages=[],
        schema=FaithfulnessReport,
        tier="fast",
        session_id=session_id,
    )


def _passes(code_issues: list[str], faith: FaithfulnessReport) -> bool:
    return not code_issues and faith.ok and not faith.unsupported_claims and not faith.omitted_critical


def _fallback(mode: str, service_response: ServiceResponse, disclaimer: str, next_question: str) -> str:
    if mode == "followup":
        return FOLLOWUP_FALLBACK_REPLY
    text = fallback_template(service_response, disclaimer)
    if mode == "quick" and next_question.strip():
        text = f"{text}\n\n{next_question.strip()}"
    return text


async def present(
    session: dict,
    service_response: ServiceResponse,
    mode: str = "initial",
    new_message: str = "",
    llm: BaseLLM | None = None,
    next_question: str = "",
) -> str:
    """
    Presents the legal service answer conversationally.
    Validates output with code guard and faithfulness checks in every mode.
    Regenerates ONCE on failure, or falls back.
    """
    client = llm or LLM()
    session_id = session.get("_id")
    source_answer = service_response.answer or ""

    raw_case = session.get("case_file", {})
    case_obj = CaseFile.model_validate(raw_case) if raw_case else CaseFile()
    case_summary = render_case_summary(case_obj)
    lang_text = case_obj.hard_core.language or session.get("language", "English")
    style_hint = session.get("style_hint") or "Calm and clear."
    # The person's own facts (and the next intake question) may be repeated back without being flagged
    allowed_context = f"{case_summary}\n{next_question}"

    # Short disclaimer only with the first legal answer (initial or quick) in the session
    disclaimer_text = ""
    if mode in ("initial", "quick") and not session.get("disclaimer_shown"):
        disclaimer_text = DISCLAIMER_SHORT

    def _done(text: str, path: str) -> str:
        session["last_presenter_path"] = path
        if disclaimer_text:
            session["disclaimer_shown"] = True
        return text

    settings = get_settings()
    if settings.direct_service_answer and mode == "initial":
        # Optional verbatim delivery (DIRECT_SERVICE_ANSWER=true)
        return _done(fallback_template(service_response, disclaimer_text), "direct_base_model")

    template_text, _ = load_prompt("presenter")
    rendered_system = render(
        template_text,
        mode=mode,
        language=lang_text,
        style_hint=style_hint,
        case_summary=case_summary,
        service_answer=source_answer,
        new_message=new_message,
        disclaimer=disclaimer_text or "(none)",
        next_question=next_question or "(none)",
    )

    # 1. Primary generation
    draft_1 = await client.text(
        prompt_name="presenter",
        system=rendered_system,
        messages=[],
        tier="smart",
        session_id=session_id,
        temperature=PRESENTER_TEMPERATURE,
    )

    code_issues_1 = check_code_guard(draft_1, source_answer, allowed_context)
    faith_1 = await check_faithfulness(source_answer, draft_1, client, session_id, mode=mode, case_summary=case_summary)
    if _passes(code_issues_1, faith_1):
        print(f"[Presenter Guard] mode={mode} path=primary", file=sys.stderr)
        return _done(draft_1, "primary")

    # 2. Problems detected: regenerate ONCE with problem instructions
    problems: list[str] = list(code_issues_1)
    problems.extend(f"Unsupported claim: {c}" for c in faith_1.unsupported_claims)
    problems.extend(f"Omitted critical detail: {o}" for o in faith_1.omitted_critical)

    correction_system = (
        f"{rendered_system}\n\n"
        "IMPORTANT CORRECTION:\n"
        f"Your previous attempt had these problems: {'; '.join(problems)}\n"
        "Stick strictly to the source answer. Do NOT add any laws, sections, cases, dates, amounts "
        "or predictions that are not in it."
    )
    print(f"[Presenter Guard] mode={mode} draft 1 failed ({len(problems)} issues). Regenerating...", file=sys.stderr)

    draft_2 = await client.text(
        prompt_name="presenter",
        system=correction_system,
        messages=[],
        tier="smart",
        session_id=session_id,
        temperature=PRESENTER_TEMPERATURE,
    )

    code_issues_2 = check_code_guard(draft_2, source_answer, allowed_context)
    faith_2 = await check_faithfulness(source_answer, draft_2, client, session_id, mode=mode, case_summary=case_summary)
    if _passes(code_issues_2, faith_2):
        print(f"[Presenter Guard] mode={mode} path=regenerated", file=sys.stderr)
        return _done(draft_2, "regenerated")

    # 3. Still failing: safe fallback
    print(f"[Presenter Guard] mode={mode} path=fallback", file=sys.stderr)
    return _done(_fallback(mode, service_response, disclaimer_text, next_question), "fallback")
