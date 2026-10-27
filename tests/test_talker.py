"""
tests/test_talker.py - Tests for talker prompt formatting and execution.
"""

import pytest

from src.llm import FakeLLM
from src.talker import build_talker_reply


@pytest.mark.asyncio
async def test_build_talker_reply_formats_user_message_tags():
    """Verify that build_talker_reply wraps user messages in <user_message> tags and calls smart tier."""
    fake = FakeLLM(responses={"talker": "I understand and I'm listening."})
    session = {"_id": "session_123", "status": "COLLECTING", "language": "en"}
    messages = [
        {"role": "user", "text": "I got an eviction notice."},
        {"role": "assistant", "text": "When did you receive it?"},
        {"role": "user", "text": "Yesterday."},
    ]

    reply = await build_talker_reply(session=session, messages=messages, llm=fake, stream=False)
    assert reply == "I understand and I'm listening."

    # Check the call history in FakeLLM
    assert len(fake.call_history) == 1
    call = fake.call_history[0]
    assert call["prompt_name"] == "talker"
    assert call["tier"] == "smart"

    # Verify user message wrapping
    history_sent = call["messages"]
    assert "<user_message>\nI got an eviction notice.\n</user_message>" == history_sent[0]["content"]
    assert "When did you receive it?" == history_sent[1]["content"]
    assert "<user_message>\nYesterday.\n</user_message>" == history_sent[2]["content"]
