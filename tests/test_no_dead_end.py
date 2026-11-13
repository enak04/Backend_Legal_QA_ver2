"""
tests/test_no_dead_end.py - The bot must never reach "not ready AND nothing to ask".

In that state the talker used to be called with an empty question list and improvised a
wrap-up ("You're all set for the legal team... you'll hear back soon"), so the person got
neither a recap nor an answer. It must instead ask for the missing core detail, or move to
the recap (which leads to the answer).
"""

import pytest

from src.db import create_session
from src.llm import FakeLLM
from src.pipeline import handle_turn
from src.schemas import AffectReading, CaseFile, CasePatch, Gap, GapAnalysis, HardCore, Intent, UnderstandOutput

NO_GAPS = "(no gaps identified)"


def _talker_systems(fake: FakeLLM) -> list[str]:
    return [c["system"] for c in fake.call_history if c["prompt_name"] == "talker"]


@pytest.mark.asyncio
async def test_location_asked_under_another_name_goes_to_recap():
    """Location was asked twice as 'Work location' (an LLM-chosen topic name), never answered."""
    session = await create_session()
    session["case_file"] = CaseFile(hard_core=HardCore(user_goal="Get severance from Google")).model_dump()
    session["asked_counts"] = {"Work location": 2}
    session["turn_count"] = 4

    fake = FakeLLM(responses={
        "extractor": UnderstandOutput(intent=Intent.provides_info, patch=CasePatch()),
        "affect": AffectReading(),
        "gap_analyzer": GapAnalysis(gaps=[Gap(topic="Work location", question_hint="Which city?", why_it_matters="x", impact="high")]),
        "confirm_summary": "Let me check I've got this right: Google let you go...",
        "talker": "You're all set for the legal team to look into this. Hang tight!",
    })
    reply, _ = await handle_turn(session, "general", llm=fake, stream=False)

    assert NO_GAPS not in "".join(_talker_systems(fake))
    assert session["status"] == "CONFIRMING"
    assert session["case_file"]["hard_core"]["jurisdiction"] == "General"
    assert "legal team" not in reply


@pytest.mark.asyncio
async def test_missing_goal_after_correction_asks_for_it():
    """After correcting the recap, no gaps remain but the goal is unknown: ask for it."""
    session = await create_session()
    session["status"] = "CONFIRMING"
    session["case_file"] = CaseFile(hard_core=HardCore(jurisdiction="Bengaluru, Karnataka")).model_dump()

    fake = FakeLLM(responses={
        "extractor": UnderstandOutput(intent=Intent.corrects, patch=CasePatch()),
        "affect": AffectReading(),
        "gap_analyzer": GapAnalysis(gaps=[]),
        "talker": "Got it. What would a good outcome look like for you?",
    })
    await handle_turn(session, "I work at google if that helps", llm=fake, stream=False)

    systems = _talker_systems(fake)
    assert systems, "talker should ask for the missing goal"
    assert NO_GAPS not in systems[0]
    assert "good outcome" in systems[0]


@pytest.mark.asyncio
async def test_talker_never_called_with_empty_gap_list_when_collecting():
    """Location missing and never asked, gap analyzer returns nothing: ask where, don't wrap up."""
    session = await create_session()
    session["case_file"] = CaseFile(hard_core=HardCore(user_goal="Get severance")).model_dump()
    session["asked_counts"] = {"Location / Jurisdiction": 2, "Termination reason": 2}
    session["turn_count"] = 3

    fake = FakeLLM(responses={
        "extractor": UnderstandOutput(intent=Intent.provides_info, patch=CasePatch()),
        "affect": AffectReading(),
        "gap_analyzer": GapAnalysis(gaps=[Gap(topic="Termination reason", question_hint="Why?", why_it_matters="x", impact="high")]),
        "confirm_summary": "Let me check I've got this right...",
        "talker": "Hang tight, you'll hear back soon.",
    })
    await handle_turn(session, "general", llm=fake, stream=False)
    assert NO_GAPS not in "".join(_talker_systems(fake))
    assert session["status"] == "CONFIRMING"
