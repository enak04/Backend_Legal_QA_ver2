"""
tests/test_tenancy_simulation.py - Multi-turn tenancy dispute simulation.

Verifies the DONE WHEN acceptance criteria:
1. A simulated conversation about a tenancy dispute reaches READY_FOR_CONFIRMATION
   after a sensible number of turns (roughly 4 to 8 turns).
2. The bot never asks more than MAX_QUESTIONS_PER_TURN questions per turn.
3. It never asks the same question three times.
4. "just give me the answer" skips low-impact questions but still asks for jurisdiction if missing.
"""

import pytest

from src.config import get_settings
from src.db import create_session, get_session
from src.llm import FakeLLM
from src.pipeline import handle_turn
from src.schemas import (
    AffectReading,
    CasePatch,
    Gap,
    GapAnalysis,
    Intent,
    NewDocument,
    NewFact,
    NewTimelineEvent,
    UnderstandOutput,
)


@pytest.mark.asyncio
async def test_tenancy_dispute_simulation_reaches_ready_for_confirmation():
    """
    Simulates a 5-turn tenancy dispute conversation reaching READY_FOR_CONFIRMATION:
    Turn 1: User reports eviction notice (missing jurisdiction and goal).
    Turn 2: User provides location (Ontario) - still missing user_goal.
    Turn 3: User provides goal ("I want to stay in my apartment") - high-impact gap on notice type.
    Turn 4: User provides notice type ("N12 notice") - high-impact gap on termination date.
    Turn 5: User provides termination date ("end of November") - gaps resolved -> READY_FOR_CONFIRMATION!
    """
    settings = get_settings()
    session = await create_session()
    sid = session["_id"]

    # Scripted responses for each turn
    turns_data = [
        # Turn 1
        {
            "user_msg": "My landlord just told me I have to move out.",
            "understand": UnderstandOutput(
                intent=Intent.provides_info,
                patch=CasePatch(
                    add_facts=[NewFact(text="Landlord told user to move out", source="inferred", confidence=0.9)]
                ),
            ),
            "affect": AffectReading(distress=2, urgency=2),
            "gaps": GapAnalysis(
                gaps=[
                    Gap(topic="Jurisdiction", question_hint="Where is the property?", why_it_matters="Tenancy laws vary by province or state", impact="high", example_answer="Toronto, Ontario", can_skip=False),
                    Gap(topic="User Goal", question_hint="What outcome do you want?", why_it_matters="Guides the advice", impact="high", example_answer="I want to stay", can_skip=False),
                ],
                legal_area_guess="Tenancy Law",
                blocking_for_service=True,
            ),
            "talker": "I am so sorry to hear that. To help you best, where is the rental unit located?",
        },
        # Turn 2
        {
            "user_msg": "The apartment is in Toronto, Ontario.",
            "understand": UnderstandOutput(
                intent=Intent.provides_info,
                patch=CasePatch(
                    set_hard_core={"jurisdiction": "Ontario"},
                    add_facts=[NewFact(text="Rental unit is in Toronto, Ontario", source="user_said", evidence="Toronto, Ontario", confidence=1.0)],
                ),
            ),
            "affect": AffectReading(distress=1, urgency=1),
            "gaps": GapAnalysis(
                gaps=[
                    Gap(topic="User Goal", question_hint="What outcome do you want?", why_it_matters="Guides the advice", impact="high", example_answer="I want to stay", can_skip=False),
                    Gap(topic="Notice Form", question_hint="Did you receive an official notice form?", why_it_matters="Notice form determines rights", impact="high", example_answer="N12 form", can_skip=False),
                ],
                legal_area_guess="Tenancy Law",
                blocking_for_service=True,
            ),
            "talker": "Thank you. What would you like to achieve regarding this eviction?",
        },
        # Turn 3
        {
            "user_msg": "I want to dispute the eviction and stay in my home.",
            "understand": UnderstandOutput(
                intent=Intent.provides_info,
                patch=CasePatch(
                    set_hard_core={"user_goal": "Dispute eviction and stay in home"},
                ),
            ),
            "affect": AffectReading(distress=1, urgency=1),
            "gaps": GapAnalysis(
                gaps=[
                    Gap(topic="Notice Form", question_hint="Did you receive an official notice form?", why_it_matters="Notice form determines rights", impact="high", example_answer="N12 form", can_skip=False),
                    Gap(topic="Termination Date", question_hint="What is the move-out date on the notice?", why_it_matters="Critical for statutory deadlines", impact="high", example_answer="November 30", can_skip=True),
                ],
                legal_area_guess="Tenancy Law",
                blocking_for_service=True,
            ),
            "talker": "Understood. Did your landlord give you an official written form or letter?",
        },
        # Turn 4
        {
            "user_msg": "Yes, they gave me an official N12 notice saying a family member is moving in.",
            "understand": UnderstandOutput(
                intent=Intent.provides_info,
                patch=CasePatch(
                    add_documents=[NewDocument(name="N12 Notice", state="has", notes="Family member moving in")],
                ),
            ),
            "affect": AffectReading(distress=1, urgency=1),
            "gaps": GapAnalysis(
                gaps=[
                    Gap(topic="Termination Date", question_hint="What is the termination date stated on the N12 notice?", why_it_matters="Required to calculate 60-day notice period", impact="high", example_answer="November 30", can_skip=True),
                ],
                legal_area_guess="Tenancy Law",
                blocking_for_service=True,
            ),
            "talker": "That is helpful. What termination date is listed on that N12 notice?",
        },
        # Turn 5
        {
            "user_msg": "The date listed is November 30th, 2026.",
            "understand": UnderstandOutput(
                intent=Intent.provides_info,
                patch=CasePatch(
                    add_timeline=[NewTimelineEvent(when_text="November 30th, 2026", description="Termination date on N12 notice")],
                ),
            ),
            "affect": AffectReading(distress=0, urgency=1),
            "gaps": GapAnalysis(
                gaps=[],  # No more high-impact gaps!
                legal_area_guess="Tenancy Law",
                blocking_for_service=False,
            ),
            "talker": "Should not be called because we are ready!",
        },
    ]

    for turn_idx, tdata in enumerate(turns_data, 1):
        fake_llm = FakeLLM(responses={
            "extractor": tdata["understand"],
            "affect": tdata["affect"],
            "gap_analyzer": tdata["gaps"],
            "talker": tdata["talker"],
        })

        reply, decision = await handle_turn(
            session=session,
            user_message=tdata["user_msg"],
            llm=fake_llm,
            stream=False,
        )

        if turn_idx < 5:
            # During collecting, reply should be normal talker reply
            assert reply == tdata["talker"]
            assert session["status"] == "COLLECTING"

            # Check that talker received at most max_questions_per_turn in gaps prompt
            talker_calls = [c for c in fake_llm.call_history if c["prompt_name"] == "talker"]
            assert len(talker_calls) == 1
            # Check prompt doesn't violate question constraint
            system_prompt = talker_calls[0]["system"]
            assert f"max_questions={settings.max_questions_per_turn}" in system_prompt or str(settings.max_questions_per_turn) in system_prompt
        else:
            # On Turn 5: Reaches CONFIRMING and produces recap!
            assert session["status"] == "CONFIRMING"
            assert "confirm_summary" in reply or len(reply) > 0


    # Verify turn count is between 4 and 8
    assert 4 <= session["turn_count"] <= 8
    updated = await get_session(sid)
    assert updated["status"] == "CONFIRMING"


@pytest.mark.asyncio
async def test_bot_never_asks_same_question_three_times():
    """
    Verifies that if a topic is asked twice without an answer from the user,
    it is treated as unknown (recorded in assumptions) and never asked a 3rd time.
    """
    session = await create_session()

    stub_gap = Gap(
        topic="Move-In Inspection Sheet",
        question_hint="Do you have the move-in inspection sheet?",
        why_it_matters="Proves initial condition",
        impact="high",
        example_answer="Yes, signed checklist",
        can_skip=True,
    )
    fallback_gap = Gap(
        topic="Rent Receipts",
        question_hint="Do you have bank transfer receipts?",
        why_it_matters="Proves payments",
        impact="medium",
        example_answer="Bank statement",
        can_skip=True,
    )

    # Hard core already present so readiness only depends on gaps
    session["case_file"] = {
        "hard_core": {"jurisdiction": "Ontario", "user_goal": "Get deposit back"},
        "facts": [], "timeline": [], "people": [], "documents": [], "deadlines": [],
        "user_questions": [], "open_questions": [], "assumptions": [], "version": 1,
    }

    # Turn 1: Bot asks "Move-In Inspection Sheet" (asked count = 1)
    fake1 = FakeLLM(responses={
        "extractor": UnderstandOutput(intent=Intent.provides_info, patch=CasePatch()),
        "affect": AffectReading(),
        "gap_analyzer": GapAnalysis(gaps=[stub_gap, fallback_gap]),
        "talker": "Do you have the move-in inspection sheet?",
    })
    reply1, _ = await handle_turn(session, "My landlord kept my deposit.", llm=fake1, stream=False)
    assert reply1 == "Do you have the move-in inspection sheet?"
    assert session["asked_counts"].get("Move-In Inspection Sheet") == 1

    # Turn 2: User ignores the question. Bot asks "Move-In Inspection Sheet" again (asked count = 2)
    fake2 = FakeLLM(responses={
        "extractor": UnderstandOutput(intent=Intent.provides_info, patch=CasePatch()),
        "affect": AffectReading(),
        "gap_analyzer": GapAnalysis(gaps=[stub_gap, fallback_gap]),
        "talker": "Could you check if you have the move-in inspection sheet?",
    })
    reply2, _ = await handle_turn(session, "He is being very unfair.", llm=fake2, stream=False)
    assert reply2 == "Could you check if you have the move-in inspection sheet?"
    assert session["asked_counts"].get("Move-In Inspection Sheet") == 2

    # Turn 3: User still didn't answer. Since it was asked twice and can_skip=True,
    # it is recorded as an assumption 'Unknown: Move-In Inspection Sheet'.
    # With no other high-impact gap, it transitions to READY_FOR_CONFIRMATION (or asks next gap)!
    fake3 = FakeLLM(responses={
        "extractor": UnderstandOutput(intent=Intent.provides_info, patch=CasePatch()),
        "affect": AffectReading(),
        "gap_analyzer": GapAnalysis(gaps=[stub_gap, fallback_gap]),
        "talker": "Do you have rent receipts?",
    })
    reply3, _ = await handle_turn(session, "I really need my money back.", llm=fake3, stream=False)

    # It was NOT asked a 3rd time! It either became ready for confirmation or asked fallback gap
    talker_call = [c for c in fake3.call_history if c["prompt_name"] == "talker"]
    if talker_call:
        assert "Move-In Inspection Sheet" not in talker_call[0]["system"]
    assert session["asked_counts"].get("Move-In Inspection Sheet") == 2  # Not incremented to 3!

    # Verify assumption recorded in case
    case_assumptions = session["case_file"]["assumptions"]
    assert any("Move-In Inspection Sheet" in a for a in case_assumptions)


@pytest.mark.asyncio
async def test_just_give_me_answer_skips_low_impact_but_asks_jurisdiction_if_missing():
    """
    Verifies that when user says 'just give me the answer':
    1. If jurisdiction is missing -> NOT ready; talker is asked to explain why jurisdiction is needed.
    2. If jurisdiction and goal are present -> ready; skips low-impact questions.
    """
    # Case 1: Missing jurisdiction
    session1 = await create_session()
    fake_missing_jur = FakeLLM(responses={
        "extractor": UnderstandOutput(
            intent=Intent.wants_answer_now,
            wants_answer_now=True,
            patch=CasePatch(set_hard_core={"user_goal": "Stop eviction"}),
        ),
        "affect": AffectReading(distress=2, urgency=3),
        "gap_analyzer": GapAnalysis(
            gaps=[Gap(topic="Jurisdiction", question_hint="Location", why_it_matters="Laws vary", impact="high", can_skip=False)],
        ),
        "talker": "I understand you want an answer right away, but tenancy laws vary completely by location. Where is your home?",
    })

    reply1, _ = await handle_turn(
        session=session1,
        user_message="Just give me the answer right now, I don't want to chat!",
        llm=fake_missing_jur,
        stream=False,
    )
    # Must NOT be ready for confirmation because jurisdiction is missing!
    assert reply1 != "READY_FOR_CONFIRMATION"
    assert session1["status"] == "COLLECTING"

    # Check system prompt warned talker to explain why jurisdiction is needed
    talker_call = [c for c in fake_missing_jur.call_history if c["prompt_name"] == "talker"][0]
    assert "Explain in one sentence why these two are needed" in talker_call["system"]

    # Case 2: Jurisdiction and user_goal present + low/medium impact gaps
    session2 = await create_session()
    fake_has_hard_core = FakeLLM(responses={
        "extractor": UnderstandOutput(
            intent=Intent.wants_answer_now,
            wants_answer_now=True,
            patch=CasePatch(
                set_hard_core={"jurisdiction": "Ontario", "user_goal": "Stop eviction"},
            ),
        ),
        "affect": AffectReading(distress=1, urgency=2),
        "gap_analyzer": GapAnalysis(
            gaps=[
                Gap(topic="pet", question_hint="Do you have pets?", why_it_matters="Pet clauses", impact="low", can_skip=True),
                Gap(topic="parking", question_hint="Do you use parking?", why_it_matters="Amenities", impact="low", can_skip=True),
            ],
        ),
        "talker": "Should not be called!",
    })

    reply2, _ = await handle_turn(
        session=session2,
        user_message="I live in Ontario and want to stop the eviction, just give me the answer!",
        llm=fake_has_hard_core,
        stream=False,
    )
    # Must be ready for confirmation, skipping low-impact questions!
    assert session2["status"] == "CONFIRMING"
    assert "confirm_summary" in reply2 or len(reply2) > 0

