"""
tests/test_affect.py - Tests for affect reading and style hint derivation.
"""

import pytest

from src.affect import affect_to_style_hint, read_affect
from src.llm import FakeLLM
from src.schemas import AffectReading


def test_affect_to_style_hint_thresholds():
    """Verify conversion of affect numeric readings to tone guidance."""
    # 1. Distress == 2
    a_distress_2 = AffectReading(distress=2)
    hint = affect_to_style_hint(a_distress_2)
    assert "The person is distressed. Acknowledge their feelings first" in hint
    assert "Do not ask for dates" not in hint

    # 2. Distress == 3
    a_distress_3 = AffectReading(distress=3)
    hint3 = affect_to_style_hint(a_distress_3)
    assert "The person is distressed" in hint3
    assert "Do not ask for dates, documents or amounts this turn." in hint3

    # 3. Confusion >= 2
    a_confusion_2 = AffectReading(confusion=2)
    hint_conf = affect_to_style_hint(a_confusion_2)
    assert "The person is confused. Explain why you are asking and give a simple example answer." in hint_conf

    # 4. Anger >= 2
    a_anger_2 = AffectReading(anger=2)
    hint_ang = affect_to_style_hint(a_anger_2)
    assert "The person is angry. Let them feel heard in one sentence" in hint_ang

    # 5. Shame >= 2
    a_shame_2 = AffectReading(shame=2)
    hint_shame = affect_to_style_hint(a_shame_2)
    assert "The person feels ashamed. Be non-judgmental and normalize asking for help." in hint_shame

    # 6. Neutral (all < 2)
    a_neutral = AffectReading(distress=1, anger=0, confusion=1, shame=0)
    assert affect_to_style_hint(a_neutral) == "Neutral and friendly."

    # 7. Style note appended
    a_with_note = AffectReading(distress=0, style_note="Keep responses very warm.")
    assert "Neutral and friendly. Keep responses very warm." == affect_to_style_hint(a_with_note)


@pytest.mark.asyncio
async def test_read_affect_with_fake_llm():
    """Verify read_affect execution using fast LLM tier."""
    expected = AffectReading(
        distress=3,
        anger=1,
        confusion=0,
        shame=0,
        urgency=3,
        crisis_self_harm=False,
        crisis_violence_or_abuse=True,
        language="English",
        style_note="Prioritize safety immediately.",
    )
    fake = FakeLLM(responses={"affect": expected})
    session = {"_id": "session_test", "status": "COLLECTING"}

    result = await read_affect(
        session=session,
        new_message="Help, my ex is breaking into my house",
        llm=fake,
    )

    assert result.crisis_violence_or_abuse is True
    assert result.distress == 3
    assert len(fake.call_history) == 1
    assert fake.call_history[0]["prompt_name"] == "affect"
    assert fake.call_history[0]["tier"] == "fast"
