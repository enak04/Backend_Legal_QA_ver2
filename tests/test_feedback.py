"""
tests/test_feedback.py - Optional user feedback on bot messages and conversations.

Verifies:
- every assistant message records what produced it (meta.kind, prompt version)
- the SSE done event exposes message_id + message_kind
- POST /sessions/{id}/feedback: create, overwrite (one entry), validation, 404/401/422
- conversation-level feedback
- comments are PII-redacted; feedback is deleted with the session
- /api/conversations routes: turn response has message_id; feedback endpoint works
- feedback_report.summarize aggregation
"""

import json
import sys
from pathlib import Path

import httpx
import pytest

from app import app, set_app_llm
from src.db import create_session, get_db, get_recent_messages
from src.feedback import FEEDBACK_REASONS
from src.llm import FakeLLM
from src.pipeline import handle_turn
from src.schemas import (
    AffectReading,
    CaseFile,
    CasePatch,
    FaithfulnessReport,
    Gap,
    GapAnalysis,
    HardCore,
    Intent,
    ServiceQuery,
    UnderstandOutput,
)
from src.service_client import ServiceClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
from feedback_report import summarize  # noqa: E402

AUTH = {"X-API-Key": "test_app_key"}

LOCATION_KNOWN = UnderstandOutput(
    intent=Intent.provides_info,
    patch=CasePatch(set_hard_core={"jurisdiction": "Pune", "user_goal": "Get my deposit back"}),
)


@pytest.fixture(autouse=True)
def fake_llm():
    fake = FakeLLM(responses={
        "extractor": LOCATION_KNOWN,
        "affect": AffectReading(),
        "gap_analyzer": GapAnalysis(gaps=[Gap(topic="Deposit amount", question_hint="How much was the deposit?", why_it_matters="x", impact="high")]),
        "talker": "Got it, Pune. How much was the deposit?",
    })
    set_app_llm(fake)
    yield fake
    set_app_llm(None)


def _done_event(raw: str) -> dict:
    for part in raw.strip().split("\n\n"):
        if part.startswith("event: done"):
            return json.loads(part.split("data: ", 1)[1])
    raise AssertionError("no done event")


async def _session_with_reply(client: httpx.AsyncClient) -> tuple[str, dict]:
    sid = (await client.post("/sessions", json={"consent": True}, headers=AUTH)).json()["session_id"]
    res = await client.post(f"/sessions/{sid}/messages", json={"text": "My landlord in Pune kept my deposit"}, headers=AUTH)
    return sid, _done_event(res.text)


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


# ---------------------------------------------------------------------------
# Message meta
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_done_event_has_message_kind():
    async with _client() as client:
        _, done = await _session_with_reply(client)
    assert done["message_id"]
    assert done["message_kind"] == "question"


@pytest.mark.asyncio
async def test_assistant_messages_record_kind_and_prompt_version():
    session = await create_session()
    srv = ServiceClient(use_mock=True)

    # question
    fake_q = FakeLLM(responses={
        "extractor": LOCATION_KNOWN,
        "affect": AffectReading(),
        "gap_analyzer": GapAnalysis(gaps=[Gap(topic="Deposit amount", question_hint="How much?", why_it_matters="x", impact="high")]),
        "talker": "How much was the deposit?",
    })
    await handle_turn(session, "My landlord in Pune kept my deposit", llm=fake_q, service_client=srv)
    msg = (await get_recent_messages(session["_id"], n=1))[0]
    assert msg["meta"]["kind"] == "question"
    assert msg["meta"]["prompt_name"] == "talker"
    assert len(msg["meta"]["prompt_version"]) == 8

    # recap
    fake_r = FakeLLM(responses={
        "extractor": UnderstandOutput(intent=Intent.provides_info),
        "affect": AffectReading(),
        "gap_analyzer": GapAnalysis(gaps=[]),
        "confirm_summary": "Let me check I've got this right...",
    })
    await handle_turn(session, "It was 50,000", llm=fake_r, service_client=srv)
    assert (await get_recent_messages(session["_id"], n=1))[0]["meta"]["kind"] == "recap"

    # answer
    fake_a = FakeLLM(responses={
        "extractor": UnderstandOutput(intent=Intent.confirms),
        "affect": AffectReading(),
        "rewriter": ServiceQuery(query_text="Deposit", documents=[{"name": "Lease", "state": "has"}]),
        "presenter": "Short version: you can ask for it back.",
        "faithfulness": FaithfulnessReport(ok=True),
    })
    await handle_turn(session, "yes", llm=fake_a, service_client=srv)
    msg = (await get_recent_messages(session["_id"], n=1))[0]
    assert msg["meta"]["kind"] == "answer"
    assert msg["meta"]["presenter_path"] == "primary"
    assert msg["meta"]["status"] == "ANSWERED"


@pytest.mark.asyncio
async def test_safety_message_kind():
    session = await create_session()
    fake = FakeLLM(responses={
        "extractor": UnderstandOutput(intent=Intent.provides_info),
        "affect": AffectReading(crisis_self_harm=True),
    })
    await handle_turn(session, "I want to end my life", llm=fake)
    assert (await get_recent_messages(session["_id"], n=1))[0]["meta"]["kind"] == "safety"


# ---------------------------------------------------------------------------
# /sessions feedback endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rate_message_and_overwrite():
    async with _client() as client:
        sid, done = await _session_with_reply(client)
        mid = done["message_id"]

        res = await client.post(f"/sessions/{sid}/feedback", json={"message_id": mid, "rating": "down"}, headers=AUTH)
        assert res.status_code == 201
        assert res.json()["rating"] == "down"

        res = await client.post(
            f"/sessions/{sid}/feedback",
            json={"message_id": mid, "rating": "down", "reasons": ["Too many questions"], "comment": "Mail me at a.b@example.com"},
            headers=AUTH,
        )
        assert res.status_code == 201

        docs = list(get_db().feedback.find({"session_id": sid}))
        assert len(docs) == 1
        doc = docs[0]
        assert doc["reasons"] == ["Too many questions"]
        assert "a.b@example.com" not in doc["comment"]
        assert "[REDACTED_EMAIL]" in doc["comment"]
        assert doc["snapshot"]["kind"] == "question"
        assert doc["snapshot"]["prompt_name"] == "talker"
        assert "Pune" in doc["snapshot"]["text"]

        # Rating shows up when the conversation is reloaded
        history = (await client.get(f"/sessions/{sid}", headers=AUTH)).json()["messages"]
        rated = next(m for m in history if m["id"] == mid)
        assert rated["feedback"] == "down"
        assert rated["kind"] == "question"


@pytest.mark.asyncio
async def test_feedback_validation_errors():
    async with _client() as client:
        sid, done = await _session_with_reply(client)
        mid = done["message_id"]
        user_msg = next(m for m in (await client.get(f"/sessions/{sid}", headers=AUTH)).json()["messages"] if m["role"] == "user")

        cases = [
            ({"message_id": user_msg["id"], "rating": "up"}, 422),          # user's own message
            ({"message_id": "does-not-exist", "rating": "up"}, 422),        # unknown message
            ({"rating": "up"}, 422),                                        # message_id missing
            ({"message_id": mid, "rating": "meh"}, 422),                    # bad rating
            ({"message_id": mid, "rating": "down", "reasons": ["Not a real reason"]}, 422),
            ({"message_id": mid, "rating": "down", "comment": "x" * 1001}, 422),
        ]
        for body, expected in cases:
            res = await client.post(f"/sessions/{sid}/feedback", json=body, headers=AUTH)
            assert res.status_code == expected, (body, res.text)

        assert (await client.post("/sessions/nope/feedback", json={"target": "conversation", "rating": "up"}, headers=AUTH)).status_code == 404
        assert (await client.post(f"/sessions/{sid}/feedback", json={"message_id": mid, "rating": "up"})).status_code == 401


@pytest.mark.asyncio
async def test_conversation_feedback_and_delete():
    async with _client() as client:
        sid, _ = await _session_with_reply(client)
        res = await client.post(
            f"/sessions/{sid}/feedback",
            json={"target": "conversation", "rating": "up", "reasons": ["Felt understood"], "comment": "Nice"},
            headers=AUTH,
        )
        assert res.status_code == 201
        doc = get_db().feedback.find_one({"session_id": sid, "target": "conversation"})
        assert doc["message_id"] is None
        assert doc["snapshot"]["message_count"] >= 2

        opts = (await client.get(f"/sessions/{sid}/feedback/options", headers=AUTH)).json()
        assert opts == FEEDBACK_REASONS

        assert (await client.delete(f"/sessions/{sid}", headers=AUTH)).status_code == 204
        assert get_db().feedback.count_documents({"session_id": sid}) == 0


# ---------------------------------------------------------------------------
# /api/conversations (external frontend)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compat_routes_expose_message_id_and_accept_feedback():
    async with _client() as client:
        cid = (await client.post("/api/conversations", json={"mode": "actionable"})).json()["conversation_id"]
        turn = (await client.post(f"/api/conversations/{cid}/messages", json={"message": "My landlord in Pune kept my deposit"})).json()
        assert turn["message_id"]
        assert turn["message_kind"] == "question"

        res = await client.post(
            f"/api/conversations/{cid}/feedback",
            json={"message_id": turn["message_id"], "rating": "up", "reasons": ["Clear"]},
        )
        assert res.status_code == 201

        convo = (await client.get(f"/api/conversations/{cid}")).json()
        rated = next(m for m in convo["messages"] if m["id"] == turn["message_id"])
        assert rated["feedback"] == "up"
        assert (await client.get("/api/feedback/options")).json() == FEEDBACK_REASONS


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def test_feedback_report_summarize():
    docs = [
        {"target": "message", "rating": "down", "reasons": ["Too many questions"], "comment": "too long",
         "snapshot": {"kind": "question", "prompt_name": "talker", "prompt_version": "aaaa1111"}, "updated_at": "2026-11-13"},
        {"target": "message", "rating": "up", "reasons": [], "comment": "",
         "snapshot": {"kind": "question", "prompt_name": "talker", "prompt_version": "bbbb2222"}},
        {"target": "message", "rating": "up", "reasons": ["Clear"], "comment": "",
         "snapshot": {"kind": "answer", "prompt_name": "presenter", "prompt_version": "cccc3333"}},
        {"target": "conversation", "rating": "down", "reasons": ["Took too long"], "comment": "", "snapshot": {}},
    ]
    s = summarize(docs)
    assert s["by_kind"]["question"] == {"down": 1, "up": 1}
    assert s["by_kind"]["answer"] == {"up": 1}
    assert s["by_version"]["talker@aaaa1111"] == {"down": 1}
    assert s["conversation"] == {"down": 1}
    assert s["reasons"]["question (down)"] == [("Too many questions", 1)]
    assert s["comments"][0]["comment"] == "too long"
    assert s["total"] == 4
