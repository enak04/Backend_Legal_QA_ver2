"""
tests/test_pipeline.py - Pipeline integration tests with FakeLLM.

Verifies:
- Crisis triggers hard-coded reply without calling the talker.
- Category is stored in session.safety_flags.
- Next normal turn passes system note to talker ("A safety message was already shown. Check in gently and do not repeat it.").
- Affect reading is stored with messages.
"""

import pytest

from src.db import create_session, get_recent_messages, get_session
from src.llm import FakeLLM
from src.pipeline import SAFETY_ALREADY_SHOWN_NOTE, handle_turn
from src.safety import SAFETY_FOLLOW_UP_NOTES
from src.schemas import AffectReading, CasePatch, Intent, NewFact, Source, UnderstandOutput


@pytest.mark.asyncio
async def test_pipeline_crisis_triggers_safety_without_talker():
    """Verify that a crisis signal intercepts the turn, produces the hard-coded reply, and skips talker."""
    fake = FakeLLM(responses={
        "extractor": UnderstandOutput(
            intent=Intent.provides_info,
            patch=CasePatch(add_facts=[NewFact(text="User expressed severe despair", source="inferred", confidence=0.8)]),
        ),
        "affect": AffectReading(
            distress=3,
            crisis_self_harm=True,
            language="English",
        ),
        "talker": "This talker reply should NEVER be called!",
    })

    session = await create_session()
    sid = session["_id"]

    reply, decision = await handle_turn(
        session=session,
        user_message="I want to end my life, I cannot go on anymore",
        llm=fake,
        stream=False,
    )

    assert decision.triggered is True
    assert decision.category == "self_harm"
    assert "988" in reply or "741741" in reply

    # Verify talker was NEVER called
    talker_calls = [c for c in fake.call_history if c["prompt_name"] == "talker"]
    assert len(talker_calls) == 0

    # Verify session safety_flags updated
    updated_session = await get_session(sid)
    assert "self_harm" in updated_session["safety_flags"]

    # Verify message has affect stored
    msgs = await get_recent_messages(sid, n=1)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["affect"]["crisis_self_harm"] is True


@pytest.mark.asyncio
async def test_pipeline_next_turn_after_safety_injects_system_note():
    """
    Verify that on the turn immediately following a safety message,
    the talker receives system_notes warning not to repeat it.
    """
    fake = FakeLLM(responses={
        "extractor": UnderstandOutput(intent=Intent.provides_info, patch=CasePatch()),
        "affect": AffectReading(distress=1, crisis_self_harm=False),
        "talker": "I am here with you. What else happened with your apartment?",
    })

    # Create session where a self-harm safety message was just shown
    session = await create_session()
    session["safety_flags"] = ["self_harm"]
    session["safety_followup_pending"] = "self_harm"

    reply, decision = await handle_turn(
        session=session,
        user_message="I am feeling a little better now, my landlord sent another letter",
        llm=fake,
        stream=False,
    )

    assert decision.triggered is False

    # Check that talker was called
    talker_calls = [c for c in fake.call_history if c["prompt_name"] == "talker"]
    assert len(talker_calls) == 1

    # Immediately after: a category-specific, human follow-up note that forbids repeating the helpline
    talker_system_prompt = talker_calls[0]["system"]
    assert SAFETY_FOLLOW_UP_NOTES["self_harm"] in talker_system_prompt
    assert "Do not repeat the helpline details." in talker_system_prompt

    # The follow-up is consumed: the next turn only carries the light "don't repeat it" note
    updated = await get_session(session["_id"])
    assert updated.get("safety_followup_pending") is None
    assert "safety_followup_pending" not in session
    fake.call_history.clear()
    await handle_turn(session=session, user_message="He wants me out by Friday", llm=fake, stream=False)
    second_prompt = [c for c in fake.call_history if c["prompt_name"] == "talker"][0]["system"]
    assert SAFETY_ALREADY_SHOWN_NOTE in second_prompt
    assert SAFETY_FOLLOW_UP_NOTES["self_harm"] not in second_prompt


@pytest.mark.asyncio
async def test_pipeline_safety_message_sets_followup_pending():
    """A newly triggered safety message marks the category for a human follow-up on the next turn."""
    fake = FakeLLM(responses={
        "extractor": UnderstandOutput(intent=Intent.provides_info, patch=CasePatch()),
        "affect": AffectReading(distress=2, crisis_detention=True),
    })
    session = await create_session()
    reply, decision = await handle_turn(
        session=session,
        user_message="The police arrested my brother last night",
        llm=fake,
        stream=False,
    )
    assert decision.category == "arrest_detention"
    updated = await get_session(session["_id"])
    assert updated["safety_followup_pending"] == "arrest_detention"


@pytest.mark.asyncio
async def test_pipeline_stores_style_hint():
    """The affect-derived style hint is saved on the session so the presenter can use it."""
    fake = FakeLLM(responses={
        "extractor": UnderstandOutput(intent=Intent.provides_info, patch=CasePatch()),
        "affect": AffectReading(anger=3),
        "talker": "That would make anyone furious. Where is the flat?",
    })
    session = await create_session()
    await handle_turn(session=session, user_message="My landlord kept my deposit for no reason!", llm=fake, stream=False)
    updated = await get_session(session["_id"])
    assert "The person is angry." in updated["style_hint"]


@pytest.mark.asyncio
async def test_pipeline_safe_confirmation_breaks_safety_loop():
    """
    Verify that when user confirms they are safe after violence_abuse was flagged,
    even if affect model still sets crisis_violence_or_abuse=True, the pipeline does NOT
    repeat the helpline, calls the talker, and injects safe confirmation notes.
    """
    fake = FakeLLM(responses={
        "extractor": UnderstandOutput(
            intent=Intent.confirms,
            patch=CasePatch(add_facts=[NewFact(text="User confirmed they are in a safe place", source=Source.user_said, confidence=1.0)]),
        ),
        "affect": AffectReading(distress=1, crisis_violence_or_abuse=True),
        "talker": "I am glad to hear you are safe now. Let us discuss the divorce threat and your footage.",
    })

    session = await create_session()
    session["safety_flags"] = ["violence_abuse"]

    reply, decision = await handle_turn(
        session=session,
        user_message="yes I am safe",
        llm=fake,
        stream=False,
    )

    assert decision.triggered is False

    # Check that talker was called
    talker_calls = [c for c in fake.call_history if c["prompt_name"] == "talker"]
    assert len(talker_calls) == 1

    talker_system_prompt = talker_calls[0]["system"]
    assert SAFETY_ALREADY_SHOWN_NOTE in talker_system_prompt
    assert "The user explicitly confirmed they are currently safe." in talker_system_prompt

