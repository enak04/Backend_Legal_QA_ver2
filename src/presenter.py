"""
src/presenter.py - Formats and presents the external service answer in friendly chat.

Translates and formats legal content from the service into conversational messages.
Guards the output using:
1. Pure code checks: rejects section/article numbers, case citations, currency amounts,
   or dates not present in the source answer.
2. Faithfulness LLM check (FAST tier) calling prompts/faithfulness.txt.
If problems are found, regenerates ONCE with problem instructions.
If it still fails, falls back to a plain formatted template with the disclaimer.
Logs which path was used ('primary', 'regenerated', 'fallback').
"""

import logging
import re
import sys

from src.case_ops import render_case_summary
from src.llm import LLM, BaseLLM
from src.messages import DISCLAIMER
from src.prompts import load_prompt, render
from src.schemas import CaseFile, FaithfulnessReport, ServiceResponse

logger = logging.getLogger(__name__)

# Regex patterns for code guard check
SECTION_REGEX = re.compile(r"\b(?:section|sec\.?|article|art\.?|clause|§)\s*\d+[a-zA-Z0-9.-]*\b", re.IGNORECASE)
CASE_NAME_REGEX = re.compile(r"\b[A-Z][a-zA-Z0-9_'\-]+\s+(?:v\.|vs\.?)\s+[A-Z][a-zA-Z0-9_'\-]+\b")
CURRENCY_REGEX = re.compile(r"(?:[\$\€\£\₹]|CAD|USD|INR)\s*\d+(?:,\d{3})*(?:\.\d+)?|\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:dollars|euros|pounds|rupees)\b", re.IGNORECASE)
DATE_REGEX = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember))\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}|\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)


def check_code_guard(presenter_text: str, source_answer: str) -> list[str]:
    """
    Checks that any section/article numbers, case citations ('v.' or 'vs'),
    currency amounts, or dates in presenter_text also appear in source_answer.
    Returns a list of violation descriptions.
    """
    violations: list[str] = []
    source_lower = source_answer.lower()

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


def fallback_template(service_response: ServiceResponse, disclaimer: str = DISCLAIMER) -> str:
    """
    Fallback plain template that prints the service answer's sections
    with light formatting and the disclaimer.
    """
    parts: list[str] = []
    if service_response.sections:
        for heading, body in service_response.sections.items():
            title = heading.replace("_", " ").title()
            parts.append(f"{title}:\n{body.strip()}")
    elif service_response.answer:
        parts.append(service_response.answer.strip())
    else:
        parts.append("Our legal service has processed your case.")

    parts.append(f"\n{disclaimer.strip()}")
    return "\n\n".join(parts)


async def check_faithfulness(
    service_answer: str,
    presenter_text: str,
    llm: BaseLLM,
    session_id: str | None = None,
) -> FaithfulnessReport:
    """Executes the FAST_MODEL faithfulness prompt check."""
    template_text, _ = load_prompt("faithfulness")
    rendered_system = render(
        template_text,
        service_answer=service_answer,
        presenter_text=presenter_text,
    )
    return await llm.json(
        prompt_name="faithfulness",
        system=rendered_system,
        messages=[],
        schema=FaithfulnessReport,
        tier="fast",
        session_id=session_id,
    )


async def present(
    session: dict,
    service_response: ServiceResponse,
    mode: str = "initial",
    new_message: str = "",
    llm: BaseLLM | None = None,
) -> str:
    """
    Presents the legal service answer in friendly chat.
    Validates output with code guard and faithfulness checks.
    Regenerates ONCE on failure, or falls back to template.
    Logs which path was used.
    """
    client = llm or LLM()
    session_id = session.get("_id")
    source_answer = service_response.answer or ""

    raw_case = session.get("case_file", {})
    case_obj = CaseFile.model_validate(raw_case) if raw_case else CaseFile()
    case_summary = render_case_summary(case_obj)
    lang_text = case_obj.hard_core.language or session.get("language", "English")
    style_hint = session.get("style_hint", "calm and clear")


    template_text, _ = load_prompt("presenter")
    rendered_system = render(
        template_text,
        mode=mode,
        language=lang_text,
        style_hint=style_hint,
        case_summary=case_summary,
        service_answer=source_answer,
        new_message=new_message,
        disclaimer=DISCLAIMER,
    )

    # 1. Primary generation
    draft_1 = await client.text(
        prompt_name="presenter",
        system=rendered_system,
        messages=[],
        tier="smart",
        session_id=session_id,
    )

    # Guard checks on draft 1
    code_issues_1 = check_code_guard(draft_1, source_answer)
    faith_1 = await check_faithfulness(source_answer, draft_1, client, session_id)

    if not code_issues_1 and faith_1.ok and not faith_1.unsupported_claims and not faith_1.omitted_critical:
        session["last_presenter_path"] = "primary"
        print("[Presenter Guard] Path: primary (Passed all checks)", file=sys.stderr)
        return draft_1

    # 2. Problems detected: Regenerate ONCE with problem instructions
    problems: list[str] = []
    if code_issues_1:
        problems.extend(code_issues_1)
    if faith_1.unsupported_claims:
        problems.extend([f"Unsupported claim: {c}" for c in faith_1.unsupported_claims])
    if faith_1.omitted_critical:
        problems.extend([f"Omitted critical detail: {o}" for o in faith_1.omitted_critical])

    fix_instruction = "Fix these problems: " + "; ".join(problems)
    correction_system = (
        f"{rendered_system}\n\n"
        f"IMPORTANT CORRECTION:\n"
        f"Your previous attempt had the following issues: {fix_instruction}\n"
        f"Strictly adhere to the source answer. Do NOT add any citations, laws, sections, dates, or amounts not in it."
    )

    print(f"[Presenter Guard] Draft 1 failed ({len(problems)} issues). Regenerating...", file=sys.stderr)

    draft_2 = await client.text(
        prompt_name="presenter",
        system=correction_system,
        messages=[],
        tier="smart",
        session_id=session_id,
    )

    # Guard checks on draft 2
    code_issues_2 = check_code_guard(draft_2, source_answer)
    faith_2 = await check_faithfulness(source_answer, draft_2, client, session_id)

    if not code_issues_2 and faith_2.ok and not faith_2.unsupported_claims and not faith_2.omitted_critical:
        session["last_presenter_path"] = "regenerated"
        print("[Presenter Guard] Path: regenerated (Passed on retry)", file=sys.stderr)
        return draft_2

    # 3. Still failed: Fall back to plain template
    session["last_presenter_path"] = "fallback"
    print("[Presenter Guard] Path: fallback (Template used)", file=sys.stderr)
    return fallback_template(service_response, DISCLAIMER)
