"""
src/compat_routes.py - Compatibility API routes matching Backend_Legal_QA interface.

Provides full REST and SSE compatibility for frontends expecting:
- GET  /api/modes (and /modes)
- GET  /api/health
- POST /api/conversations (and /conversations)
- POST /api/conversations/{id}/messages (and /conversations/{id}/messages)
- POST /api/conversations/{id}/messages/stream (and /conversations/{id}/messages/stream)
- GET  /api/conversations/{id} (and /conversations/{id})
- DELETE /api/conversations/{id} (and /conversations/{id})
- GET  /api/conversations
- POST /api/conversations/{id}/files
- POST /api/audio/transcribe
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
from typing import Any
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from src.config import get_settings
from src.db import (
    create_session,
    delete_session,
    get_all_messages,
    get_db,
    get_session,
    update_session,
)
from src.pipeline import handle_turn

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Schemas ──────────────────────────────────────────────────────────

class CreateConversationRequest(BaseModel):
    mode: str = Field(default="actionable", description="One of: actionable, informative, readable")
    language: str | None = Field(default="en", description="Optional language code (default 'en')")


class SendMessageRequest(BaseModel):
    message: str | None = Field(default=None, description="The user's message text")
    content: str | None = Field(default=None, description="Alternative field for message text")
    language: str | None = Field(default=None, description="Optional language code")

    @property
    def text(self) -> str:
        return (self.message or self.content or "").strip()


# ── Helpers ──────────────────────────────────────────────────────────

def format_compact_case_state(session: dict) -> dict[str, Any]:
    """Formats case_file into the compact CaseState dictionary expected by frontend."""
    raw_case = session.get("case_file") or {}
    hard_core = raw_case.get("hard_core", {}) if isinstance(raw_case, dict) else {}
    financial = raw_case.get("financial", {}) if isinstance(raw_case, dict) else {}
    urgency_data = raw_case.get("urgency", {}) if isinstance(raw_case, dict) else {}
    facts = raw_case.get("facts", []) if isinstance(raw_case, dict) else []

    issues = []
    dispute = raw_case.get("dispute_type") or hard_core.get("user_goal")
    if dispute:
        issues.append(dispute)
    for f in facts:
        if isinstance(f, dict) and f.get("status") == "active":
            txt = f.get("text", "")
            if 0 < len(txt) <= 80:
                issues.append(txt)

    amt = financial.get("amount_raw")
    if not amt and financial.get("amount_clean") is not None:
        amt = f"₹{financial['amount_clean']:,.0f}"

    evidence_count = len([
        f for f in facts
        if isinstance(f, dict) and (
            "evidence" in str(f.get("source", "")).lower()
            or "document" in str(f.get("text", "")).lower()
            or "cctv" in str(f.get("text", "")).lower()
            or "record" in str(f.get("text", "")).lower()
        )
    ])

    return {
        "domain": raw_case.get("dispute_type") or "General Legal",
        "language": hard_core.get("language") or session.get("language", "en"),
        "issues": issues,
        "jurisdiction": hard_core.get("jurisdiction"),
        "urgency": urgency_data.get("level", "normal") if isinstance(urgency_data, dict) else "normal",
        "amount": amt,
        "evidence_count": evidence_count,
        "user_goal": hard_core.get("user_goal"),
    }


def build_conversation_turn_response(
    session: dict,
    reply_text: str,
) -> dict[str, Any]:
    """Transforms a turn output into the JSON shape expected by Backend_Legal_QA frontends."""
    conversation_id = session["_id"]
    mode = session.get("mode", "actionable")
    lang = session.get("language", "en")
    curr_status = session.get("status", "COLLECTING")
    compact_state = format_compact_case_state(session)
    now_iso = datetime.now(timezone.utc).isoformat()

    if curr_status == "ANSWERED":
        # Final answer turn
        qa_raw = session.get("last_service_answer") or {}
        reasoning_chain = qa_raw.get("reasoning_chain") or []
        retrieved_cases = qa_raw.get("retrieved_cases") or qa_raw.get("sources") or []
        sources = [
            {"question": c.get("question", ""), "answer": c.get("answer", "")}
            for c in retrieved_cases if isinstance(c, dict)
        ]
        authorities = qa_raw.get("authorities") or []

        return {
            "type": "final_answer",
            "conversation_id": conversation_id,
            "mode": mode,
            "language": lang,
            "stage": "answered",
            "answer": reply_text,
            "original_answer": reply_text,
            "message": reply_text,
            "original_message": reply_text,
            "case_state": compact_state,
            "compact_case_state": compact_state,
            "legal_assessment": None,
            "reasoning_chain": reasoning_chain,
            "sources": sources,
            "authorities": authorities,
            "timestamp": now_iso,
        }
    else:
        # Follow-up intake turn (COLLECTING or CONFIRMING)
        return {
            "type": "follow_up",
            "conversation_id": conversation_id,
            "mode": mode,
            "language": lang,
            "stage": "gathering_info",
            "message": reply_text,
            "original_message": reply_text,
            "answer": reply_text,
            "case_state": compact_state,
            "compact_case_state": compact_state,
            "timestamp": now_iso,
        }


# ── Modes & Health Endpoints ─────────────────────────────────────────

@router.get("/modes")
@router.get("/api/modes")
async def list_modes() -> list[dict[str, str]]:
    """Return the available legal-assistance modes with descriptions."""
    return [
        {
            "id": "actionable",
            "name": "Actionable",
            "description": (
                "Guides you toward an appropriate legal remedy or action. "
                "Collects relevant facts through a conversational intake process "
                "before providing actionable legal guidance."
            ),
        },
        {
            "id": "informative",
            "name": "Informative",
            "description": (
                "Helps you understand a legal issue, law, section, concept, or legal question. "
                "Provides direct, informative answers without unnecessary follow-up questions."
            ),
        },
        {
            "id": "readable",
            "name": "Readable",
            "description": (
                "Makes legal information easier for a normal user to understand. "
                "Provides simplified, accessible explanations of legal concepts and procedures."
            ),
        },
    ]


@router.get("/api/health")
async def api_health() -> dict[str, str]:
    """Health check for api prefix."""
    return {"status": "ok"}


# ── Conversation CRUD Endpoints ──────────────────────────────────────

@router.get("/conversations")
@router.get("/api/conversations")
async def list_conversations() -> list[dict[str, Any]]:
    """Lists conversations for history display."""
    db = get_db()
    cursor = db.sessions.find().sort("updated_at", -1).limit(50)
    sessions = cursor if isinstance(cursor, list) else list(cursor)
    results = []
    for s in sessions:
        created = s.get("created_at")
        updated = s.get("updated_at")
        results.append({
            "conversation_id": s["_id"],
            "mode": s.get("mode", "actionable"),
            "stage": "answered" if s.get("status") == "ANSWERED" else "gathering_info",
            "language": s.get("language", "en"),
            "created_at": created.isoformat() if isinstance(created, datetime) else str(created or ""),
            "updated_at": updated.isoformat() if isinstance(updated, datetime) else str(updated or ""),
        })
    return results


@router.post("/conversations")
@router.post("/api/conversations")
async def create_conversation(body: CreateConversationRequest) -> dict[str, Any]:
    """Starts a new conversation matching Backend_Legal_QA response shape."""
    valid_mode = body.mode if body.mode in ("actionable", "informative", "readable") else "actionable"
    lang = body.language or "en"
    now = datetime.now(timezone.utc)

    session = await create_session(
        language=lang,
        consent=True,
        consent_ts=now,
    )
    # Save selected mode
    await update_session(session_id=session["_id"], updates={"mode": valid_mode})

    return {
        "conversation_id": session["_id"],
        "mode": valid_mode,
        "language": lang,
        "stage": "initial",
        "created_at": now.isoformat(),
    }


@router.get("/conversations/{conversation_id}")
@router.get("/api/conversations/{conversation_id}")
async def get_conversation_endpoint(conversation_id: str) -> dict[str, Any]:
    """Retrieve full conversation state."""
    session = await get_session(conversation_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conversation_id}")

    raw_messages = await get_all_messages(conversation_id)
    formatted_messages = [
        {
            "role": m.get("role", ""),
            "content": m.get("text", ""),
            "text": m.get("text", ""),
            "original_content": m.get("text", ""),
            "timestamp": m["ts"].isoformat() if isinstance(m.get("ts"), datetime) else str(m.get("ts", "")),
        }
        for m in raw_messages
    ]

    compact_state = format_compact_case_state(session)
    curr_status = session.get("status", "COLLECTING")

    qa_results = []
    if session.get("last_service_answer"):
        qa_ans = session["last_service_answer"]
        qa_query = session.get("last_service_query") or {}
        qa_results.append({
            "question": qa_query.get("query_text", ""),
            "answer": qa_ans.get("answer", ""),
            "original_answer": qa_ans.get("answer", ""),
            "reasoning_chain": qa_ans.get("reasoning_chain", []),
            "sources": qa_ans.get("retrieved_cases", qa_ans.get("sources", [])),
        })

    created = session.get("created_at")
    updated = session.get("updated_at")

    return {
        "conversation_id": conversation_id,
        "mode": session.get("mode", "actionable"),
        "stage": "answered" if curr_status == "ANSWERED" else "gathering_info",
        "language": session.get("language", "en"),
        "case_state": compact_state,
        "compact_case_state": compact_state,
        "legal_assessment": None,
        "facts": session.get("case_file", {}),
        "messages": formatted_messages,
        "legal_qa_results": qa_results,
        "files": session.get("files", []),
        "created_at": created.isoformat() if isinstance(created, datetime) else str(created or ""),
        "updated_at": updated.isoformat() if isinstance(updated, datetime) else str(updated or ""),
    }


@router.delete("/conversations/{conversation_id}")
@router.delete("/api/conversations/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: str):
    """Deletes a conversation."""
    session = await get_session(conversation_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    await delete_session(conversation_id)
    return Response(status_code=204)


# ── Message Turn Endpoints ───────────────────────────────────────────

@router.post("/conversations/{conversation_id}/messages")
@router.post("/api/conversations/{conversation_id}/messages")
async def send_message_endpoint(
    conversation_id: str,
    body: SendMessageRequest,
    request: Request,
) -> dict[str, Any]:
    """Processes a user message and returns follow_up or final_answer."""
    session = await get_session(conversation_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conversation_id}")

    user_text = body.text
    if not user_text:
        raise HTTPException(status_code=422, detail="Field 'message' or 'content' is required and cannot be empty.")

    # Get app's LLM client and session lock
    from app import get_llm_client, get_session_lock

    lock = await get_session_lock(conversation_id)
    async with lock:
        current_session = await get_session(conversation_id) or session
        if body.language and body.language != current_session.get("language"):
            await update_session(conversation_id, {"language": body.language})
            current_session["language"] = body.language

        reply_or_stream, safety_decision = await handle_turn(
            session=current_session,
            user_message=user_text,
            llm=get_llm_client(),
            stream=False,
        )

        reply_text = str(reply_or_stream)
        updated_session = await get_session(conversation_id) or current_session
        return build_conversation_turn_response(updated_session, reply_text)


@router.post("/conversations/{conversation_id}/messages/stream")
@router.post("/api/conversations/{conversation_id}/messages/stream")
async def send_message_stream_endpoint(
    conversation_id: str,
    body: SendMessageRequest,
    request: Request,
):
    """
    Stream message response using Server-Sent Events (SSE).
    Emits status, token, and complete events matching Backend_Legal_QA frontend expectations.
    """
    session = await get_session(conversation_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conversation_id}")

    user_text = body.text
    if not user_text:
        raise HTTPException(status_code=422, detail="Field 'message' or 'content' is required and cannot be empty.")

    from app import get_llm_client, get_session_lock

    async def event_generator():
        try:
            # 1. Yield initial status immediately (<200ms)
            yield f"data: {json.dumps({'type': 'status', 'status': 'analyzing'})}\n\n"

            lock = await get_session_lock(conversation_id)
            async with lock:
                current_session = await get_session(conversation_id) or session
                if body.language and body.language != current_session.get("language"):
                    await update_session(conversation_id, {"language": body.language})
                    current_session["language"] = body.language

                reply_or_stream, safety_decision = await handle_turn(
                    session=current_session,
                    user_message=user_text,
                    llm=get_llm_client(),
                    stream=True,
                )

                full_chunks: list[str] = []
                if hasattr(reply_or_stream, "__aiter__"):
                    async for chunk in reply_or_stream:
                        if chunk:
                            full_chunks.append(chunk)
                            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                            await asyncio.sleep(0.005)
                else:
                    text_str = str(reply_or_stream)
                    full_chunks.append(text_str)
                    yield f"data: {json.dumps({'type': 'token', 'content': text_str})}\n\n"

                final_reply = "".join(full_chunks)
                updated_session = await get_session(conversation_id) or current_session
                result = build_conversation_turn_response(updated_session, final_reply)

                # Emit final complete payload
                yield f"data: {json.dumps({'type': 'complete', 'data': result})}\n\n"

        except HTTPException as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc.detail)})}\n\n"
        except Exception as exc:
            logger.exception("Error during streaming SSE turn")
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Files & Voice Endpoints ──────────────────────────────────────────

@router.post("/conversations/{conversation_id}/files")
@router.post("/api/conversations/{conversation_id}/files")
async def upload_file_endpoint(
    conversation_id: str,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Upload document attachment linked to conversation."""
    session = await get_session(conversation_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    content = await file.read()
    file_id = str(uuid.uuid4())[:12]
    now_iso = datetime.now(timezone.utc).isoformat()

    from src.document_parser import extract_text_from_file

    extracted = extract_text_from_file(
        content=content,
        filename=file.filename or "attachment",
        content_type=file.content_type or "",
    )

    file_record = {
        "file_id": file_id,
        "filename": file.filename or "attachment",
        "content_type": file.content_type or "application/octet-stream",
        "size_bytes": len(content),
        "extracted_text": extracted,
        "uploaded_at": now_iso,
    }

    files = session.get("files") or []
    files.append(file_record)
    await update_session(conversation_id, {"files": files})

    return file_record


@router.post("/audio/transcribe")
@router.post("/api/audio/transcribe")
async def transcribe_audio_endpoint(
    file: UploadFile = File(..., description="Audio file (.wav, .mp3, .m4a, .webm, .ogg)"),
    language: str | None = Form(None),
    provider: str = Form("whisper"),
) -> dict[str, Any]:
    """Speech transcription endpoint."""
    settings = get_settings()
    if settings.llm_api_key and not settings.llm_api_key.startswith("mock_") and settings.provider == "openai":
        try:
            import httpx
            content = await file.read()
            if content:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        "https://api.openai.com/v1/audio/transcriptions",
                        headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                        files={"file": (file.filename or "audio.wav", content, file.content_type or "audio/wav")},
                        data={"model": "whisper-1", **({"language": language} if language else {})},
                    )
                    if resp.status_code == 200:
                        text_val = resp.json().get("text", "")
                        return {"text": text_val, "language": language or "en", "provider": "whisper"}
        except Exception as e:
            logger.warning("Whisper transcription failed: %s", e)

    return {"text": "", "language": language or "en", "provider": "fallback"}
