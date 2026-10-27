"""
tests/test_llm.py - Tests for LLM interface and FakeLLM behavior.
"""

from pydantic import BaseModel
import pytest

from src.llm import FakeLLM, LLMOutputError


class MockOutputSchema(BaseModel):
    intent: str
    confidence: float


@pytest.mark.asyncio
async def test_fake_llm_text():
    """Verify that FakeLLM returns canned text responses."""
    fake = FakeLLM(responses={"test_prompt": "Hello from mock"})
    result = await fake.text(
        prompt_name="test_prompt",
        system="system instruction",
        messages=[{"role": "user", "content": "hi"}],
        tier="smart",
        session_id="test_session",
    )
    assert result == "Hello from mock"
    assert len(fake.call_history) == 1
    assert fake.call_history[0]["prompt_name"] == "test_prompt"


@pytest.mark.asyncio
async def test_fake_llm_json_success():
    """Verify that FakeLLM returns validated Pydantic instances."""
    canned = {"intent": "ask_for_help", "confidence": 0.95}
    fake = FakeLLM(responses={"intent_prompt": canned})

    result = await fake.json(
        prompt_name="intent_prompt",
        system="system instruction",
        messages=[],
        schema=MockOutputSchema,
        tier="fast",
        session_id="test_session",
    )
    assert isinstance(result, MockOutputSchema)
    assert result.intent == "ask_for_help"
    assert result.confidence == 0.95


@pytest.mark.asyncio
async def test_fake_llm_json_validation_failure():
    """Verify that FakeLLM raises LLMOutputError when response does not fit schema."""
    invalid_canned = {"wrong_field": 123}
    fake = FakeLLM(responses={"intent_prompt": invalid_canned})

    with pytest.raises(LLMOutputError):
        await fake.json(
            prompt_name="intent_prompt",
            system="system instruction",
            messages=[],
            schema=MockOutputSchema,
            tier="fast",
            session_id="test_session",
        )


@pytest.mark.asyncio
async def test_fake_llm_stream():
    """Verify that FakeLLM yields stream chunks correctly."""
    fake = FakeLLM(responses={"stream_prompt": "Word1 Word2 Word3"})
    stream_gen = fake.stream(
        prompt_name="stream_prompt",
        system="system",
        messages=[],
        tier="smart",
        session_id="test_session",
    )

    chunks = []
    async for chunk in stream_gen:
        chunks.append(chunk)

    full_output = "".join(chunks)
    assert full_output == "Word1 Word2 Word3"
