"""
tests/test_compat_routes.py - Tests for Backend_Legal_QA API route compatibility.
"""

import json
import pytest
from fastapi.testclient import TestClient

from app import app, set_app_llm
from src.llm import FakeLLM
from src.schemas import AffectReading, CasePatch, GapAnalysis, Intent, NewFact, UnderstandOutput


@pytest.fixture(autouse=True)
def setup_fake_llm():
    """Configures deterministic FakeLLM for compatibility route tests."""
    fake = FakeLLM(responses={
        "extractor": UnderstandOutput(
            intent=Intent.provides_info,
            language="English",
            patch=CasePatch(
                add_facts=[NewFact(text="Tenant received eviction notice", source="inferred", confidence=0.8)]
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
        "talker": "I hear you. Could you share what city or state your rented property is in?",
    })
    set_app_llm(fake)
    yield fake
    set_app_llm(None)


def test_modes_endpoint():
    """Verify GET /api/modes and GET /modes return list of 3 modes."""
    client = TestClient(app)
    for path in ("/api/modes", "/modes"):
        res = client.get(path)
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)
        assert len(data) == 3
        ids = [m["id"] for m in data]
        assert "actionable" in ids
        assert "informative" in ids
        assert "readable" in ids


def test_api_health():
    """Verify GET /api/health returns status ok."""
    client = TestClient(app)
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_create_and_get_conversation_flow():
    """Verify complete CRUD flow: create, get, message, stream, list, delete."""
    client = TestClient(app)

    # 1. Create conversation
    create_res = client.post("/api/conversations", json={"mode": "actionable", "language": "en"})
    assert create_res.status_code == 200
    conv_data = create_res.json()
    assert "conversation_id" in conv_data
    assert conv_data["mode"] == "actionable"
    assert conv_data["stage"] == "initial"
    cid = conv_data["conversation_id"]

    # 2. Send message (POST /api/conversations/{id}/messages)
    msg_res = client.post(
        f"/api/conversations/{cid}/messages",
        json={"message": "My landlord wants to evict me without notice."},
    )
    assert msg_res.status_code == 200
    msg_data = msg_res.json()
    assert msg_data["conversation_id"] == cid
    assert msg_data["type"] == "follow_up"
    assert "message" in msg_data
    assert "answer" in msg_data
    assert msg_data["message"] == msg_data["answer"]
    assert "case_state" in msg_data

    # 3. Stream message (POST /api/conversations/{id}/messages/stream)
    stream_res = client.post(
        f"/api/conversations/{cid}/messages/stream",
        json={"content": "I am in Bangalore, Karnataka."},
    )
    assert stream_res.status_code == 200
    assert "text/event-stream" in stream_res.headers.get("content-type", "")
    lines = stream_res.text.strip().split("\n\n")
    events = [json.loads(line.replace("data: ", "")) for line in lines if line.startswith("data: ")]
    event_types = [e["type"] for e in events]
    assert "status" in event_types
    assert "token" in event_types
    assert "complete" in event_types

    # 4. Get full conversation (GET /api/conversations/{id})
    get_res = client.get(f"/api/conversations/{cid}")
    assert get_res.status_code == 200
    state_data = get_res.json()
    assert state_data["conversation_id"] == cid
    assert len(state_data["messages"]) >= 2
    # Verify content field is present on messages
    assert "content" in state_data["messages"][0]

    # 5. List conversations (GET /api/conversations)
    list_res = client.get("/api/conversations")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert any(c["conversation_id"] == cid for c in list_data)

    # 6. Upload file (POST /api/conversations/{id}/files)
    file_payload = {"file": ("lease.txt", b"Standard lease agreement terms", "text/plain")}
    file_res = client.post(f"/api/conversations/{cid}/files", files=file_payload)
    assert file_res.status_code == 200
    f_data = file_res.json()
    assert f_data["filename"] == "lease.txt"
    assert "file_id" in f_data

    # 7. Delete conversation (DELETE /api/conversations/{id})
    del_res = client.delete(f"/api/conversations/{cid}")
    assert del_res.status_code == 204

    # 8. Verify 404 after deletion
    del_verify = client.get(f"/api/conversations/{cid}")
    assert del_verify.status_code == 404
