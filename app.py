"""
app.py - FastAPI Web Application for legal-intake-bot.

Provides:
- POST /sessions: create session with mandatory consent
- POST /sessions/{id}/messages: stream assistant reply via SSE with per-session serialization
- GET /sessions/{id}: retrieve session state and message history
- DELETE /sessions/{id}: permanently delete session and all associated records across collections
- GET /health: unauthenticated health check
- GET /: serves the clean, calm vanilla web interface
- Header X-API-Key authentication matching APP_API_KEY
- CORS origins configured via ALLOWED_ORIGINS
"""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.config import get_settings
from src.db import create_session, delete_session, get_all_messages, get_recent_messages, get_session
from src.llm import BaseLLM, FakeLLM, LLM
from src.messages import DISCLAIMER, PRIVACY_NOTICE, WELCOME
from src.pipeline import handle_turn
from src.schemas import AffectReading, CasePatch, GapAnalysis, Intent, NewFact, UnderstandOutput


app = FastAPI(
    title="legal-intake-bot API",
    description="Conversational intake chatbot web service for legal assistance.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS Configuration
# ---------------------------------------------------------------------------
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Per-Session Concurrency Locks
# ---------------------------------------------------------------------------
_session_locks: dict[str, asyncio.Lock] = {}
_locks_guard = asyncio.Lock()
_app_llm: BaseLLM | None = None


def get_llm_client() -> BaseLLM:
    """Returns the application LLM client, using FakeLLM if running in mock/fake mode."""
    global _app_llm
    if _app_llm is not None:
        return _app_llm
    current_settings = get_settings()
    if (
        current_settings.provider == "fake"
        or not current_settings.llm_api_key
        or current_settings.llm_api_key.startswith("mock_")
    ):
        return FakeLLM(responses={
            "extractor": UnderstandOutput(
                intent=Intent.provides_info,
                language="English",
                patch=CasePatch(
                    add_facts=[NewFact(text="User provided case details", source="inferred", confidence=0.8)]
                ),
                change_summary="Recorded user details",
            ),
            "affect": AffectReading(
                distress=1,
                anger=0,
                confusion=0,
                shame=0,
                urgency=1,
                language="English",
                style_note="Keep tone supportive and calm.",
            ),
            "gap_analyzer": GapAnalysis(gaps=[], blocking_for_service=False),
            "talker": "I hear you, and I am here to help you organize what happened. Could you share a few more details about the timeline or key people involved?",
        })
    return LLM()


def set_app_llm(custom_llm: BaseLLM | None) -> None:
    """Injects a custom LLM client for tests."""
    global _app_llm
    _app_llm = custom_llm


async def get_session_lock(session_id: str) -> asyncio.Lock:
    """Retrieves or creates a dedicated asyncio.Lock for a given session."""
    async with _locks_guard:
        if session_id not in _session_locks:
            _session_locks[session_id] = asyncio.Lock()
        return _session_locks[session_id]


# ---------------------------------------------------------------------------
# Authentication Dependency
# ---------------------------------------------------------------------------
async def verify_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> str:
    """Validates that the incoming request contains a valid X-API-Key header."""
    current_settings = get_settings()
    if not x_api_key or x_api_key != current_settings.app_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key in X-API-Key header.",
        )
    return x_api_key


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------
class CreateSessionRequest(BaseModel):
    consent: bool
    user_id: str | None = None
    language: str = "en"


class MessageRequest(BaseModel):
    text: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check() -> dict[str, str]:
    """Public health check endpoint."""
    return {"status": "healthy"}


@app.get("/", response_class=FileResponse)
async def serve_web_ui():
    """Serves the minimal, mobile-friendly chat page."""
    web_file = Path(__file__).resolve().parent / "web" / "index.html"
    if not web_file.is_file():
        raise HTTPException(status_code=404, detail="Web interface not found.")
    return FileResponse(web_file, media_type="text/html")


@app.post("/sessions", dependencies=[Depends(verify_api_key)], status_code=status.HTTP_201_CREATED)
async def create_session_endpoint(req: CreateSessionRequest) -> dict[str, Any]:
    """
    Creates a new intake session.
    Rejects with 400 Bad Request if consent is not explicitly true.
    """
    if not req.consent:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Consent must be explicitly accepted (consent: true) to start a session.",
        )

    now = datetime.now(timezone.utc)
    session = await create_session(
        user_id=req.user_id,
        language=req.language or "en",
        consent=True,
        consent_ts=now,
    )

    return {
        "session_id": session["_id"],
        "welcome_text": WELCOME,
        "disclaimer": DISCLAIMER,
        "privacy_notice": PRIVACY_NOTICE,
    }


@app.post("/sessions/{session_id}/messages", dependencies=[Depends(verify_api_key)])
async def post_message_stream(session_id: str, req: MessageRequest):
    """
    Handles a user message and streams the assistant response via Server-Sent Events (SSE).
    Uses a per-session lock to serialize concurrent messages.
    """
    session = await get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    async def sse_event_stream():
        lock = await get_session_lock(session_id)
        async with lock:
            # Refresh session document inside the lock
            current_session = await get_session(session_id)
            if not current_session:
                yield f"event: error\ndata: {json.dumps({'error': 'Session not found.'})}\n\n"
                return

            # 1. Initial status event: thinking
            yield f"event: status\ndata: {json.dumps({'status': 'thinking'})}\n\n"

            try:
                # 2. Execute pipeline turn with streaming enabled
                reply_or_stream, safety_decision = await handle_turn(
                    session=current_session,
                    user_message=req.text.strip(),
                    llm=get_llm_client(),
                    stream=True,
                )

                # 3. If safety crisis was triggered, send safety event
                if safety_decision.triggered:
                    yield f"event: safety\ndata: {json.dumps({'category': safety_decision.category, 'triggered': True})}\n\n"

                # 4. Stream tokens
                full_chunks: list[str] = []
                if hasattr(reply_or_stream, "__aiter__"):
                    async for chunk in reply_or_stream:
                        if chunk:
                            full_chunks.append(chunk)
                            yield f"event: token\ndata: {json.dumps({'token': chunk, 'text': chunk})}\n\n"
                else:
                    # Single token event for hard-coded or non-streamed steps
                    text_str = str(reply_or_stream)
                    full_chunks.append(text_str)
                    yield f"event: token\ndata: {json.dumps({'token': text_str, 'text': text_str})}\n\n"

                final_reply = "".join(full_chunks)

                # 5. Fetch updated session state and emit final status
                updated_session = await get_session(session_id) or current_session
                final_status = updated_session.get("status", "COLLECTING")
                yield f"event: status\ndata: {json.dumps({'status': final_status.lower()})}\n\n"

                # 6. Retrieve latest message ID from database
                recent = await get_recent_messages(session_id=session_id, n=1)
                msg_id = str(recent[0]["_id"]) if recent else ""

                # 7. Final done event
                yield f"event: done\ndata: {json.dumps({'reply': final_reply, 'text': final_reply, 'status': final_status, 'message_id': msg_id})}\n\n"

            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(sse_event_stream(), media_type="text/event-stream")


@app.get("/sessions/{session_id}", dependencies=[Depends(verify_api_key)])
async def get_session_endpoint(session_id: str) -> dict[str, Any]:
    """
    Returns session status, language, safety flags, and complete message history.
    Internal case notes are omitted unless DEBUG=true.
    """
    session = await get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    current_settings = get_settings()
    raw_messages = await get_all_messages(session_id)
    formatted_messages = [
        {
            "id": str(m["_id"]),
            "role": m.get("role", ""),
            "text": m.get("text", ""),
            "timestamp": m["ts"].isoformat() if isinstance(m.get("ts"), datetime) else str(m.get("ts", "")),
        }
        for m in raw_messages
    ]

    response_data: dict[str, Any] = {
        "session_id": session_id,
        "status": session.get("status", "COLLECTING"),
        "language": session.get("language", "en"),
        "safety_flag": bool(session.get("safety_flags")),
        "safety_flags": session.get("safety_flags", []),
        "messages": formatted_messages,
    }

    if current_settings.debug:
        response_data["case_file"] = session.get("case_file", {})
        response_data["case_version"] = session.get("case_version", 0)

    return response_data


@app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(verify_api_key)])
async def delete_session_endpoint(session_id: str):
    """
    Permanently deletes a session and all related documents across collections
    (messages, case_versions, llm_logs, service_calls).
    """
    session = await get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    await delete_session(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Backend_Legal_QA Compatibility Routes (/api/modes, /api/conversations, etc.)
# ---------------------------------------------------------------------------
from src.compat_routes import router as compat_router

app.include_router(compat_router)


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port)


