"""
tests/test_app.py - Tests for FastAPI web endpoints, SSE streaming, auth, and concurrency.

Verifies:
1. Health check works without authentication.
2. X-API-Key is enforced on all protected endpoints.
3. Consent is strictly required (400 if false).
4. Session creation returns welcome, disclaimer, and privacy notice.
5. Streaming SSE events arrive in exact order: status -> token -> status -> done.
6. Safety messages trigger a safety event and the emergency response.
7. DELETE /sessions/{id} permanently removes everything across all collections (returns 204).
8. Concurrent messages to one session are serialized using the per-session lock.
9. GET /sessions/{id} returns message history and hides internal case notes.
"""

import asyncio
import json
import pytest
import httpx
from fastapi.testclient import TestClient

from app import app, set_app_llm
from src.config import get_settings
from src.db import get_db
from src.llm import FakeLLM
from src.schemas import AffectReading, CasePatch, GapAnalysis, Intent, NewFact, UnderstandOutput


VALID_API_KEY = "test_app_key"
AUTH_HEADERS = {"X-API-Key": VALID_API_KEY}


@pytest.fixture(autouse=True)
def setup_fake_llm():
    """Configures deterministic FakeLLM for web endpoint tests."""
    fake = FakeLLM(responses={
        "extractor": UnderstandOutput(
            intent=Intent.provides_info,
            language="English",
            patch=CasePatch(
                add_facts=[NewFact(text="Tenant received notice to vacate", source="inferred", confidence=0.8)]
            ),
            change_summary="Recorded notice details",
        ),
        "affect": AffectReading(
            distress=1,
            anger=0,
            confusion=0,
            shame=0,
            urgency=1,
            language="English",
            style_note="Supportive, calm.",
        ),
        "gap_analyzer": GapAnalysis(gaps=[], blocking_for_service=False),
        "talker": "I hear you, and I am here to help you outline what happened. When did you receive this notice?",
    })
    set_app_llm(fake)
    yield fake
    set_app_llm(None)


def parse_sse_events(raw_body: str) -> list[tuple[str, dict]]:
    """Helper to parse raw SSE text body into a list of (event_name, data_dict) tuples."""
    events = []
    parts = raw_body.strip().split("\n\n")
    for part in parts:
        if not part.strip():
            continue
        lines = part.strip().split("\n")
        event_name = "message"
        data_str = ""
        for line in lines:
            if line.startswith("event: "):
                event_name = line[len("event: "):].strip()
            elif line.startswith("data: "):
                data_str = line[len("data: "):].strip()

        try:
            data = json.loads(data_str)
        except Exception:
            data = {"raw": data_str}
        events.append((event_name, data))
    return events


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_health_check_unauthenticated():
    """Verify /health returns 200 without requiring X-API-Key."""
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "healthy"}


def test_api_key_enforcement():
    """Verify that requests missing X-API-Key or having an invalid key are rejected with 401."""
    client = TestClient(app)

    # Missing API Key
    res = client.post("/sessions", json={"consent": True})
    assert res.status_code == 401
    assert "Invalid or missing API key" in res.json()["detail"]

    # Invalid API Key
    res = client.post("/sessions", json={"consent": True}, headers={"X-API-Key": "wrong_key"})
    assert res.status_code == 401

    # Protected GET route
    res = client.get("/sessions/some-id", headers={"X-API-Key": "wrong_key"})
    assert res.status_code == 401

    # Protected DELETE route
    res = client.delete("/sessions/some-id", headers={"X-API-Key": "wrong_key"})
    assert res.status_code == 401


def test_consent_required():
    """Verify that POST /sessions rejects with 400 if consent is false."""
    client = TestClient(app)
    res = client.post("/sessions", json={"consent": False}, headers=AUTH_HEADERS)
    assert res.status_code == 400
    assert "Consent must be explicitly accepted" in res.json()["detail"]


def test_create_session_success():
    """Verify POST /sessions creates session and returns welcome, disclaimer, and privacy notice."""
    client = TestClient(app)
    res = client.post("/sessions", json={"consent": True, "language": "en"}, headers=AUTH_HEADERS)
    assert res.status_code == 201

    data = res.json()
    assert "session_id" in data
    assert len(data["session_id"]) > 10
    assert "welcome_text" in data
    assert "disclaimer" in data
    assert "privacy_notice" in data


@pytest.mark.asyncio
async def test_streaming_events_arrive_in_order():
    """Verify SSE streaming events arrive in exact order: status -> token -> status -> done."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Create session
        res = await client.post("/sessions", json={"consent": True}, headers=AUTH_HEADERS)
        assert res.status_code == 201
        sid = res.json()["session_id"]

        # 2. Send message and read SSE stream
        msg_res = await client.post(
            f"/sessions/{sid}/messages",
            json={"text": "Hello, I got an eviction notice today."},
            headers=AUTH_HEADERS,
        )
        assert msg_res.status_code == 200
        assert "text/event-stream" in msg_res.headers["content-type"]

        events = parse_sse_events(msg_res.text)
        assert len(events) >= 3

        # Event 1: status -> thinking
        assert events[0][0] == "status"
        assert events[0][1]["status"] == "thinking"

        # Middle events: token chunks
        token_events = [e for e in events if e[0] == "token"]
        assert len(token_events) > 0
        streamed_reply = "".join(e[1].get("token") or e[1].get("text") for e in token_events)
        assert len(streamed_reply) > 10

        # Event N-1: updated status
        status_events = [e for e in events if e[0] == "status"]
        assert len(status_events) >= 2
        assert status_events[-1][1]["status"] in ("collecting", "confirming", "answered")

        # Event N: done event with final reply, status, and message_id
        done_event = events[-1]
        assert done_event[0] == "done"
        assert "reply" in done_event[1]
        assert done_event[1]["reply"] == streamed_reply
        assert "status" in done_event[1]
        assert "message_id" in done_event[1]
        assert len(done_event[1]["message_id"]) > 5


@pytest.mark.asyncio
async def test_safety_event_in_stream():
    """Verify emergency phrases trigger a safety event followed by the hard-coded crisis response."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/sessions", json={"consent": True}, headers=AUTH_HEADERS)
        sid = res.json()["session_id"]

        # Emergency self-harm phrase
        msg_res = await client.post(
            f"/sessions/{sid}/messages",
            json={"text": "I want to end my life, I have no reason to live."},
            headers=AUTH_HEADERS,
        )
        assert msg_res.status_code == 200

        events = parse_sse_events(msg_res.text)
        event_names = [e[0] for e in events]

        # Verify safety event was emitted
        assert "safety" in event_names
        safety_evt = next(e for e in events if e[0] == "safety")
        assert safety_evt[1]["category"] == "self_harm"
        assert safety_evt[1]["triggered"] is True

        # Verify token content contains crisis helpline (988 or 741741)
        tokens = [e[1].get("token", "") for e in events if e[0] == "token"]
        full_tokens = "".join(tokens)
        assert "988" in full_tokens or "741741" in full_tokens


@pytest.mark.asyncio
async def test_delete_removes_everything():
    """Verify DELETE /sessions/{id} permanently removes records across all collections and returns 204."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Create session and send message
        res = await client.post("/sessions", json={"consent": True}, headers=AUTH_HEADERS)
        sid = res.json()["session_id"]

        await client.post(
            f"/sessions/{sid}/messages",
            json={"text": "I am being evicted unfairly."},
            headers=AUTH_HEADERS,
        )

        # 2. Check DB records exist
        db = get_db()
        assert db.sessions.count_documents({"_id": sid}) == 1
        assert db.messages.count_documents({"session_id": sid}) >= 2
        assert db.case_versions.count_documents({"session_id": sid}) >= 1
        assert db.llm_logs.count_documents({"session_id": sid}) >= 1

        # 3. Call DELETE
        del_res = await client.delete(f"/sessions/{sid}", headers=AUTH_HEADERS)
        assert del_res.status_code == 204

        # 4. Verify all records across all collections are deleted
        assert db.sessions.count_documents({"_id": sid}) == 0
        assert db.messages.count_documents({"session_id": sid}) == 0
        assert db.case_versions.count_documents({"session_id": sid}) == 0
        assert db.llm_logs.count_documents({"session_id": sid}) == 0
        assert db.service_calls.count_documents({"session_id": sid}) == 0

        # 5. Subsequent GET returns 404
        get_res = await client.get(f"/sessions/{sid}", headers=AUTH_HEADERS)
        assert get_res.status_code == 404


@pytest.mark.asyncio
async def test_concurrent_messages_serialized():
    """Verify two concurrent messages to one session are serialized without error or corruption."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/sessions", json={"consent": True}, headers=AUTH_HEADERS)
        sid = res.json()["session_id"]

        # Send two messages simultaneously
        task1 = client.post(
            f"/sessions/{sid}/messages",
            json={"text": "Message 1: Landlord gave me a 3-day notice."},
            headers=AUTH_HEADERS,
        )
        task2 = client.post(
            f"/sessions/{sid}/messages",
            json={"text": "Message 2: I also paid my rent on time."},
            headers=AUTH_HEADERS,
        )

        resp1, resp2 = await asyncio.gather(task1, task2)

        assert resp1.status_code == 200
        assert resp2.status_code == 200

        # Verify both messages and both assistant replies exist in DB
        db = get_db()
        user_msgs = list(db.messages.find({"session_id": sid, "role": "user"}))
        asst_msgs = list(db.messages.find({"session_id": sid, "role": "assistant"}))

        assert len(user_msgs) == 2
        assert len(asst_msgs) == 2


@pytest.mark.asyncio
async def test_get_session_details_and_debug_flag():
    """Verify GET /sessions/{id} returns message history and hides case_file when DEBUG=false."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/sessions", json={"consent": True, "language": "en"}, headers=AUTH_HEADERS)
        sid = res.json()["session_id"]

        await client.post(
            f"/sessions/{sid}/messages",
            json={"text": "I got an eviction notice today."},
            headers=AUTH_HEADERS,
        )

        get_res = await client.get(f"/sessions/{sid}", headers=AUTH_HEADERS)
        assert get_res.status_code == 200
        data = get_res.json()

        assert data["session_id"] == sid
        assert data["status"] in ("COLLECTING", "CONFIRMING", "ANSWERED")
        assert data["language"] == "en"
        assert "messages" in data
        assert len(data["messages"]) >= 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["role"] == "assistant"

        # Case notes hidden when debug=false
        settings = get_settings()
        if not settings.debug:
            assert "case_file" not in data
