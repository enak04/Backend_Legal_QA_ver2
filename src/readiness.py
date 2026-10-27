"""
src/readiness.py - Pure code evaluation of case readiness and gap tracking.

Determines whether the chatbot has gathered enough information to transition
from COLLECTING to CONFIRMING state, or if further questions are required.
"""

from src.config import Settings
from src.schemas import CaseFile, Gap, GapAnalysis


def is_ready(
    case: CaseFile,
    gap_analysis: GapAnalysis,
    wants_answer_now: bool,
    turn_count: int,
    settings: Settings,
    session: dict | None = None,
    asked_counts: dict[str, int] | None = None,
) -> tuple[bool, str]:
    """
    Evaluates whether the case file is ready for confirmation.

    Rules:
    1. jurisdiction and user_goal must be present, otherwise NOT ready even if
       the user is impatient.
    2. Ready if wants_answer_now and hard core details are present.
    3. Ready if turn_count >= settings.max_collecting_turns and hard core present.
    4. Ready if there are no high-impact gaps (ignoring can_skip ones that
       were already asked twice).

    Returns:
        tuple[bool, str]: (is_ready, reason_string)
    """
    # 1. Hard core check: jurisdiction and user_goal must be present
    has_jurisdiction = bool(case.hard_core.jurisdiction and case.hard_core.jurisdiction.strip())
    has_goal = bool(case.hard_core.user_goal and case.hard_core.user_goal.strip())
    if not (has_jurisdiction and has_goal):
        missing = []
        if not has_jurisdiction:
            missing.append("jurisdiction")
        if not has_goal:
            missing.append("user_goal")
        return False, f"Missing required core details: {', '.join(missing)}"

    # 2. Impatient user check: ready if wants_answer_now and hard core present
    if wants_answer_now:
        return True, "User requested answer immediately and hard core details are present"

    # 3. Maximum collecting turns reached
    if turn_count >= settings.max_collecting_turns:
        return True, "I think I have enough, shall I summarize?"

    # 4. Check for blocking high-impact gaps
    counts: dict[str, int]
    if session is not None:
        counts = session.setdefault("asked_counts", {})
    elif asked_counts is not None:
        counts = asked_counts
    else:
        counts = {}

    blocking_gaps: list[Gap] = []
    for gap in gap_analysis.gaps:
        times_asked = counts.get(gap.topic, 0)
        if times_asked >= 2:
            # Topic asked twice with no answer is treated as unknown: record assumption
            assumption_text = f"Unknown: {gap.topic}"
            if assumption_text not in case.assumptions:
                case.assumptions.append(assumption_text)
            # If can_skip, it is ignored when asked twice
            if gap.can_skip:
                continue

        if gap.impact == "high":
            blocking_gaps.append(gap)

    if not blocking_gaps:
        return True, "No high-impact gaps remaining"

    return False, f"High-impact gap remaining: {blocking_gaps[0].topic}"


def track_asked_gap(session: dict, topic: str) -> int:
    """
    Increments the asked count for a topic in session['asked_counts'].
    Returns the new count.
    """
    counts = session.setdefault("asked_counts", {})
    counts[topic] = counts.get(topic, 0) + 1
    return counts[topic]


def filter_gaps_to_ask(
    session: dict,
    case: CaseFile,
    gaps: list[Gap],
    max_questions: int = 1,
) -> list[Gap]:
    """
    Selects top gaps to ask.
    Filters out any topic that has already been asked twice (recording an assumption
    'Unknown: <topic>') so the bot stops asking it.
    Limits the result to max_questions.
    """
    counts = session.setdefault("asked_counts", {})
    to_ask: list[Gap] = []

    for gap in gaps:
        times_asked = counts.get(gap.topic, 0)
        if times_asked >= 2:
            assumption_text = f"Unknown: {gap.topic}"
            if assumption_text not in case.assumptions:
                case.assumptions.append(assumption_text)
            continue

        to_ask.append(gap)
        if len(to_ask) >= max_questions:
            break

    return to_ask
