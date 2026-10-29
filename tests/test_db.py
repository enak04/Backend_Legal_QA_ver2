"""
tests/test_db.py - Tests for MongoDB operations and helper functions using mongomock.
"""

import asyncio
from datetime import datetime, timezone
import mongomock
import pytest

from src.db import (
    add_message,
    create_session,
    get_recent_messages,
    get_session,
    init_db,
    log_llm_call,
    log_service_call,
    save_case_version,
    set_db,
    update_session,
)


@pytest.fixture(autouse=True)
def mock_db():
    """Sets up an in-memory mongomock database for all tests in this module."""
    client = mongomock.MongoClient()
    db = client["test_legal_intake_bot"]
    init_db(db)
    set_db(db)
    return db


@pytest.mark.asyncio
async def test_session_lifecycle():
    """Verify session creation, retrieval, and updating."""
    now = datetime.now(timezone.utc)
    session = await create_session(
        user_id="user_123",
        language="en",
        consent=True,
        consent_ts=now,
    )
    session_id = session["_id"]
    assert session_id is not None
    assert session["status"] == "COLLECTING"
    assert session["consent"]["granted"] is True

    # Retrieve
    fetched = await get_session(session_id)
    assert fetched is not None
    assert fetched["_id"] == session_id
    assert fetched["user_id"] == "user_123"

    # Update
    updated = await update_session(session_id, {"status": "CONFIRMING", "turn_count": 3})
    assert updated is True

    refetched = await get_session(session_id)
    assert refetched["status"] == "CONFIRMING"
    assert refetched["turn_count"] == 3


@pytest.mark.asyncio
async def test_messages_chronological_order():
    """Verify adding messages and retrieving them in chronological order."""
    session = await create_session()
    sid = session["_id"]

    await add_message(sid, role="user", text="First message")
    await asyncio.sleep(0.01)
    await add_message(sid, role="assistant", text="Second message")
    await asyncio.sleep(0.01)
    await add_message(sid, role="user", text="Third message")

    recent = await get_recent_messages(sid, n=2)
    assert len(recent) == 2
    assert recent[0]["text"] == "Second message"
    assert recent[1]["text"] == "Third message"

    # Verify embedded messages in session document
    session_doc = await get_session(sid)
    assert session_doc is not None
    assert "messages" in session_doc
    assert len(session_doc["messages"]) == 3
    assert session_doc["messages"][0]["text"] == "First message"
    assert session_doc["messages"][1]["text"] == "Second message"
    assert session_doc["messages"][2]["text"] == "Third message"


@pytest.mark.asyncio
async def test_save_case_version():
    """Verify saving immutable case versions."""
    session = await create_session()
    sid = session["_id"]

    patch = {"fact": "incident occurred on Monday"}
    full_case = {"facts": ["incident occurred on Monday"]}

    v_doc = await save_case_version(session_id=sid, version=1, patch=patch, full_case=full_case)
    assert v_doc["version"] == 1
    assert v_doc["session_id"] == sid
    assert v_doc["patch"] == patch


@pytest.mark.asyncio
async def test_log_llm_and_service_calls():
    """Verify logging of LLM calls and external service calls."""
    session = await create_session()
    sid = session["_id"]

    llm_doc = await log_llm_call(
        session_id=sid,
        prompt_name="talker",
        prompt_version="abc12345",
        model="gpt-4o",
        input_chars=120,
        output="Hello",
        latency_ms=350,
        tokens_in=30,
        tokens_out=10,
        success=True,
    )
    assert llm_doc["prompt_name"] == "talker"
    assert llm_doc["success"] is True

    svc_doc = await log_service_call(
        session_id=sid,
        request={"query": "test query"},
        response={"answer": "mock answer"},
        latency_ms=500,
        status="ok",
    )
    assert svc_doc["status"] == "ok"
    assert svc_doc["latency_ms"] == 500
