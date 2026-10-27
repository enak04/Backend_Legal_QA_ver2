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
from src.pipeline import handle_turn
from src.schemas import AffectReading, CasePatch, Intent, NewFact, UnderstandOutput


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

    # Create session with existing safety flag
    session = await create_session()
    session["safety_flags"] = ["self_harm"]

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

    # Verify that the system prompt sent to talker contains the warning
    talker_system_prompt = talker_calls[0]["system"]
    assert "A safety message was already shown. Check in gently and do not repeat it." in talker_system_prompt
