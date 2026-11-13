"""
src/confirm.py - Confirmation stage logic: generates recap summaries and builds external service queries.

Uses:
- confirm_summary.txt (SMART tier) to produce the 'here is what I understood' recap.
- rewriter.txt (SMART tier) to compile active case facts into a self-contained ServiceQuery.
- redact.py to strip PII (phone numbers, emails, long digit sequences) from the query.
"""

from src.case_ops import render_case_json_for_prompts, render_case_summary
from src.llm import LLM, BaseLLM
from src.prompts import load_prompt, render
from src.redact import redact_service_query
from src.schemas import CaseFile, ServiceQuery

# Recap is written in the shared voice; slightly higher than JSON calls so it reads naturally
RECAP_TEMPERATURE = 0.6


async def make_recap(
    session: dict,
    case: CaseFile,
    llm: BaseLLM | None = None,
) -> str:
    """
    Generates a warm, plain-language recap of what has been understood so far
    using the confirm_summary prompt (SMART_MODEL tier).
    """
    client = llm or LLM()
    session_id = session.get("_id")

    lang_text = case.hard_core.language or session.get("language", "English")
    summary_text = render_case_summary(case)
    assumptions_text = "\n".join(f"- {a}" for a in case.assumptions) if case.assumptions else "None"

    template_text, _ = load_prompt("confirm_summary")
    rendered_system = render(
        template_text,
        language=lang_text,
        case_summary=summary_text,
        assumptions=assumptions_text,
    )

    recap_text = await client.text(
        prompt_name="confirm_summary",
        system=rendered_system,
        messages=[],
        tier="smart",
        session_id=session_id,
        temperature=RECAP_TEMPERATURE,
    )

    return recap_text.strip()


async def build_service_query(
    case: CaseFile,
    llm: BaseLLM | None = None,
    session: dict | None = None,
) -> ServiceQuery:
    """
    Compiles active case items into a self-contained ServiceQuery for the external
    legal answering service using the rewriter prompt (SMART_MODEL tier).
    Strips phone numbers, emails, and long digit sequences via redact_service_query.
    """
    client = llm or LLM()
    session_id = session.get("_id") if session else None

    case_json = render_case_json_for_prompts(case)
    assumptions_text = "\n".join(f"- {a}" for a in case.assumptions) if case.assumptions else "None"
    questions_text = "\n".join(f"- {q}" for q in case.user_questions) if case.user_questions else "None"


    template_text, _ = load_prompt("rewriter")
    rendered_system = render(
        template_text,
        case_json=case_json,
        assumptions=assumptions_text,
        user_questions=questions_text,
    )

    raw_query = await client.json(
        prompt_name="rewriter",
        system=rendered_system,
        messages=[],
        schema=ServiceQuery,
        tier="smart",
        session_id=session_id,
    )

    # Strip PII before returning
    cleaned_query = redact_service_query(raw_query)
    return cleaned_query


def build_quick_question_query(case: CaseFile, question: str) -> ServiceQuery:
    """
    Builds a ServiceQuery for a single question asked mid-conversation, without an LLM call
    (to keep the turn fast). Includes a short context block from the case so far so the
    legal service can tailor the answer. PII is stripped before returning.
    """
    hc = case.hard_core
    context = render_case_summary(case)
    query_text = (
        f"Question: {question.strip()}\n\n"
        "Context (the conversation is still in progress, so these details may be incomplete):\n"
        f"{context}"
    )
    query = ServiceQuery(
        kind="quick_question",
        query_text=query_text,
        jurisdiction=hc.jurisdiction,
        legal_area=hc.legal_area_guess,
        user_goal=hc.user_goal,
        questions_to_answer=[question.strip()],
        user_language=hc.language or "English",
    )
    return redact_service_query(query)
