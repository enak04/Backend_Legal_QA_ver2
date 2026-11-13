"""
src/gaps.py - Gap analysis identifying missing information in the case file.

Calls the 'gap_analyzer' prompt on the FAST_MODEL tier to determine what
additional facts or core details are needed before legal advice can be given.
Injects service-requested items (pending_service_gaps) as prepended high-impact gaps.
"""


from src.case_ops import render_case_summary
from src.llm import LLM, BaseLLM
from src.prompts import load_prompt, render
from src.schemas import CaseFile, Gap, GapAnalysis


async def find_gaps(
    session: dict,
    case: CaseFile,
    llm: BaseLLM | None = None,
) -> GapAnalysis:
    """
    Identifies up to 3 gaps in the case file using the FAST_MODEL tier.

    If the session has pending_service_gaps (set when the external service
    replies 'needs_more_info'), they are prepended as high-impact gaps.
    """
    client = llm or LLM()
    session_id = session.get("_id")

    # Format already_asked topics from session asked_counts
    asked_counts = session.get("asked_counts", {})
    asked_topics = [topic for topic, cnt in asked_counts.items() if cnt > 0]
    already_asked_str = "\n".join(f"- {topic}" for topic in asked_topics) if asked_topics else "None so far"

    jurisdiction_str = case.hard_core.jurisdiction or "Unknown"
    legal_area_str = case.hard_core.legal_area_guess or "Unknown"
    summary_str = render_case_summary(case)

    template_text, _ = load_prompt("gap_analyzer")
    rendered_system = render(
        template_text,
        jurisdiction=jurisdiction_str,
        legal_area=legal_area_str,
        case_summary=summary_str,
        already_asked=already_asked_str,
    )

    gap_analysis = await client.json(
        prompt_name="gap_analyzer",
        system=rendered_system,
        messages=[],
        schema=GapAnalysis,
        tier="fast",
        session_id=session_id,
    )

    # Inject service-requested gaps if present
    pending = session.get("pending_service_gaps", [])
    if pending:
        injected: list[Gap] = []
        for item in pending:
            if isinstance(item, Gap):
                injected.append(item.model_copy(update={"impact": "high"}))
            elif isinstance(item, dict):
                d = dict(item)
                d["impact"] = "high"
                injected.append(Gap.model_validate(d))
            elif isinstance(item, str):
                injected.append(
                    Gap(
                        topic=item,
                        question_hint=f"Information needed: {item}",
                        why_it_matters="The legal review service needs this specific information to proceed.",
                        impact="high",
                        example_answer="",
                        can_skip=False,
                    )
                )

        gap_analysis.gaps = (injected + gap_analysis.gaps)[:3]
        gap_analysis.blocking_for_service = True

    # Ensure hard-core jurisdiction is prioritized if missing and not already asked twice
    has_jurisdiction = bool(case.hard_core.jurisdiction and case.hard_core.jurisdiction.strip())
    if not has_jurisdiction:
        if location_asked_count(asked_counts) < 2 and not any(is_location_topic(g.topic) for g in gap_analysis.gaps):
            gap_analysis.gaps.insert(0, location_gap())

    return gap_analysis


LOCATION_GAP_TOPIC = "Location / Jurisdiction"
GOAL_GAP_TOPIC = "Desired outcome"


def is_location_topic(topic: str) -> bool:
    t = topic.lower()
    return "location" in t or "jurisdiction" in t


def location_asked_count(asked_counts: dict[str, int]) -> int:
    """How often the location was asked, under any topic name the gap analyzer used."""
    return sum(n for topic, n in (asked_counts or {}).items() if is_location_topic(topic))


def location_gap() -> Gap:
    return Gap(
        topic=LOCATION_GAP_TOPIC,
        question_hint="Where is this happening — which city and state?",
        why_it_matters="Rules differ from state to state, so this can change the answer.",
        impact="high",
        example_answer="Pune, Maharashtra",
        can_skip=False,
    )


def goal_gap() -> Gap:
    return Gap(
        topic=GOAL_GAP_TOPIC,
        question_hint="What would a good outcome look like for you?",
        why_it_matters="It tells us what to focus the answer on.",
        impact="high",
        example_answer="Getting my full severance pay",
        can_skip=True,
    )
