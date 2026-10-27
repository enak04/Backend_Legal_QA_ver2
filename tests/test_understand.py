"""
tests/test_understand.py - Tests for understand / extractor function with FakeLLM.
"""

import pytest

from src.llm import FakeLLM
from src.schemas import CasePatch, Intent, NewFact, UnderstandOutput
from src.understand import understand


@pytest.mark.asyncio
async def test_understand_extractor_call():
    """Verify that understand loads extractor prompt and parses UnderstandOutput from FakeLLM."""
    expected_output = UnderstandOutput(
        intent=Intent.provides_info,
        wants_answer_now=False,
        language="English",
        patch=CasePatch(
            add_facts=[
                NewFact(text="Landlord raised rent by 20%", source="user_said", evidence="raised rent by 20%")
            ],
            set_hard_core={"jurisdiction": "California"},
        ),
        change_summary="Recorded rent increase and jurisdiction",
    )

    fake = FakeLLM(responses={"extractor": expected_output})
    session = {
        "_id": "session_abc",
        "status": "COLLECTING",
        "language": "English",
        "case_file": {},
    }

    result = await understand(
        session=session,
        new_message="My landlord raised rent by 20% in California.",
        llm=fake,
        recent_messages=[],
    )

    assert isinstance(result, UnderstandOutput)
    assert result.intent == Intent.provides_info
    assert result.patch.set_hard_core["jurisdiction"] == "California"
    assert len(result.patch.add_facts) == 1
    assert result.patch.add_facts[0].text == "Landlord raised rent by 20%"
    assert len(fake.call_history) == 1
    assert fake.call_history[0]["prompt_name"] == "extractor"
    assert fake.call_history[0]["tier"] == "fast"
