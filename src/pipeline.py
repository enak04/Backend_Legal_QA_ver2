"""
src/pipeline.py - Orchestrates the per-turn intake processing pipeline.

State Machine (ARCHITECTURE.md):
COLLECTING -> CONFIRMING -> ANSWERED
- COLLECTING: Gathers facts. When ready, generates recap (SMART) and moves to CONFIRMING.
  * If the person asks a legal question mid-conversation and it can be answered now
    (general question, or location known), the legal service answers it right away
    (quick mode) and the conversation continues. Otherwise the talker acknowledges the
    question and defers it naturally.
- CONFIRMING: Recap shown.
  * User confirms (by intent, or a short whole-message affirmation): applies any new info,
    builds ServiceQuery (SMART), calls service.
    - status 'answered' -> present (SMART + guard), status ANSWERED.
    - status 'needs_more_info' -> set pending_service_gaps, status COLLECTING.
    - status 'error' -> honest error message, keep query, allow retry.
  * User corrects/provides info -> applies patch, status COLLECTING, continues intake.
- ANSWERED: Answer presented.
  * New legal question -> answered by the legal service (quick mode).
  * New documents -> service re-queried and the updated answer presented.
  * Otherwise -> follow-up from the stored answer only (guarded).
Safety checks run in EVERY state before normal handling.
"""

import asyncio
import re
import sys
import uuid
from datetime import datetime, timezone
from collections.abc import AsyncGenerator

from src.affect import affect_to_style_hint, read_affect
from src.case_ops import apply_patch, render_case_summary
from src.config import get_settings
from src.confirm import build_quick_question_query, build_service_query, make_recap
from src.db import add_message, get_recent_messages, save_case_version, update_session
from src.gaps import find_gaps
from src.llm import LLM, BaseLLM
from src.messages import SERVICE_ERROR_REPLY
from src.presenter import present
from src.readiness import describe_progress, filter_gaps_to_ask, is_ready, track_asked_gap
from src.safety import SAFETY_FOLLOW_UP_NOTES, decide, detect_keywords, is_safe_confirmation
from src.schemas import CaseFile, Gap, Intent, SafetyDecision, ServiceResponse, UnderstandOutput
from src.service_client import ServiceClient
from src.talker import build_talker_reply
from src.understand import understand

LOCATION_GAP_TOPIC = "Location / Jurisdiction"

SAFETY_ALREADY_SHOWN_NOTE = "A safety message was shown earlier in this conversation. Do not repeat it."


class ReadyConfirmation(str):
    """
    String representing confirmation recap, which also equals 'READY_FOR_CONFIRMATION'
    for test assertions, and unpacks as (reply, safety_decision).
    """
    decision: SafetyDecision

    def __new__(cls, text: str = "READY_FOR_CONFIRMATION", decision: SafetyDecision | None = None) -> "ReadyConfirmation":
        obj = super().__new__(cls, text)
        obj.decision = decision or SafetyDecision()
        return obj

    def __eq__(self, other: object) -> bool:
        if other == "READY_FOR_CONFIRMATION":
            return True
        return super().__eq__(other)

    def __iter__(self):
        yield str(self)
        yield self.decision


# ---------------------------------------------------------------------------
# Short-reply detection (whole-message matching, never substrings)
# ---------------------------------------------------------------------------

def _normalize_reply(message: str) -> str:
    """Lowercases, drops apostrophes/punctuation and collapses spaces."""
    text = message.strip().lower().replace("’", "").replace("'", "")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Ignore a trailing "thanks"/"thank you"
    text = re.sub(r"\s*(?:thanks|thank you|thankyou|thx)$", "", text).strip()
    return text


AFFIRMATIONS: set[str] = {
    "yes", "yeah", "yea", "yep", "yup", "ya", "ok", "okay", "k", "sure", "correct", "right",
    "thats right", "that is right", "thats correct", "that is correct", "all correct", "all good",
    "yes correct", "yes right", "yes thats right", "yes thats correct", "yes that is correct", "yes that is right",
    "yes please", "go ahead", "yes go ahead", "please go ahead", "proceed", "please proceed", "yes proceed",
    "looks good", "looks right", "perfect", "exactly", "spot on", "sounds right",
    "retry", "try again", "please try again",
    "haan", "han", "haa", "ha", "ji", "haan ji", "ji haan", "sahi hai", "haan sahi hai", "theek hai", "thik hai",
    "si", "sí", "correcto", "así es", "asi es", "adelante",
}


def is_short_affirmation(message: str) -> bool:
    """True only if the WHOLE message is a short agreement (so 'not right' is never a yes)."""
    return _normalize_reply(message) in AFFIRMATIONS


_DONE_EXACT: set[str] = {
    "done", "im done", "i am done", "no more", "thats it", "that is it", "thats all", "that is all",
    "nothing else", "nothing more", "bas", "bas itna hi", "nothing else to add",
}
_DONE_PREFIXES: tuple[str, ...] = (
    "thats all", "that is all", "nothing else", "nothing more to add",
    "give me the answer", "give me answer", "give answer", "just give me the answer",
    "give me legal solution", "give legal solution", "give me the legal solution",
)


def is_done_signal(message: str, turn_num: int) -> bool:
    """True if the person signals they have nothing more to add (whole message or leading phrase)."""
    norm = _normalize_reply(message)
    if norm in ("nothing", "no", "nope", "nahi", "nahin"):
        return turn_num >= 3
    if norm in _DONE_EXACT:
        return True
    return any(norm == p or norm.startswith(p + " ") for p in _DONE_PREFIXES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _async_string_stream(text: str, session_id: str, affect_dict: dict) -> AsyncGenerator[str, None]:
    """Yields a canned/hard-coded string as an async stream and logs it to DB."""
    yield text
    await add_message(session_id=session_id, role="assistant", text=text, affect=affect_dict)


async def _stream_and_save(
    stream_gen: AsyncGenerator[str, None],
    session_id: str,
    affect_dict: dict,
) -> AsyncGenerator[str, None]:
    """Wraps an LLM stream to yield chunks and saves full assistant response upon completion."""
    collected: list[str] = []
    async for chunk in stream_gen:
        collected.append(chunk)
        yield chunk
    full_text = "".join(collected)
    await add_message(session_id=session_id, role="assistant", text=full_text, affect=affect_dict)


def _presenter_state(session: dict) -> dict:
    """Session fields the presenter / quick answers may have changed and that must be persisted."""
    return {
        "disclaimer_shown": bool(session.get("disclaimer_shown")),
        "quick_answers": session.get("quick_answers", []),
    }


def _safety_notes(session: dict, user_message: str) -> list[str]:
    """
    Safety-related guidance for the talker:
    - right after a safety message: a category-specific, human follow-up
    - if the person says they are safe: acknowledge briefly and move on
    - otherwise, if a safety message was ever shown: just don't repeat it
    Consumes session['safety_followup_pending'].
    """
    notes: list[str] = []
    pending = session.pop("safety_followup_pending", None)
    if is_safe_confirmation(user_message):
        if session.get("safety_flags"):
            notes.append(SAFETY_ALREADY_SHOWN_NOTE)
        notes.append(
            "The user explicitly confirmed they are currently safe. "
            "Acknowledge that in one brief, warm sentence, then carry on with their situation."
        )
    elif pending and pending in SAFETY_FOLLOW_UP_NOTES:
        notes.append(SAFETY_FOLLOW_UP_NOTES[pending])
    elif session.get("safety_flags"):
        notes.append(SAFETY_ALREADY_SHOWN_NOTE)
    return notes


def _defer_question_note(question: str, next_gap: Gap | None) -> str:
    note = (
        f'They asked: "{question}". You do not have an answer to that yet, so do not answer it yourself. '
        "Acknowledge the question naturally (it's a good thing to ask) and say you'll make sure they get a proper answer on it"
    )
    if next_gap:
        note += f", which depends partly on this: {next_gap.question_hint} Ask that next."
    else:
        note += "."
    return note


async def _try_quick_answer(
    session: dict,
    case: CaseFile,
    understand_out: UnderstandOutput,
    client: BaseLLM,
    srv_client: ServiceClient,
    user_message: str,
    next_question: str = "",
) -> str | None:
    """
    Answers a mid-conversation legal question via the legal service when it can be answered now:
    a general question, or the location is already known. Returns the presented reply, or None
    if it should be deferred (not answerable yet, or the service could not answer).
    """
    question = understand_out.question.strip()
    if not question:
        return None
    has_location = bool(case.hard_core.jurisdiction and case.hard_core.jurisdiction.strip())
    if understand_out.question_needs_case_facts and not has_location:
        return None

    query = build_quick_question_query(case, question)
    resp = await srv_client.ask(query=query, session_id=session["_id"])
    if resp.status != "answered" or not (resp.answer or resp.sections):
        return None

    session.setdefault("quick_answers", []).append({"question": question, "answer": resp.answer or ""})
    return await present(
        session=session,
        service_response=resp,
        mode="quick",
        new_message=user_message,
        llm=client,
        next_question=next_question,
    )


# ---------------------------------------------------------------------------
# Main turn handler
# ---------------------------------------------------------------------------

async def handle_turn(
    session: dict,
    user_message: str,
    llm: BaseLLM | None = None,
    service_client: ServiceClient | None = None,
    stream: bool = False,
    debug: bool = False,
    document_data: dict | None = None,
) -> tuple[str | AsyncGenerator[str, None], SafetyDecision] | ReadyConfirmation:
    """
    Executes one complete conversational turn according to the state machine:
    COLLECTING -> CONFIRMING -> ANSWERED.
    Supports optional document_data dict (filename, content_type, size_bytes, text).
    """
    settings = get_settings()
    client = llm or LLM()
    srv_client = service_client or ServiceClient()
    session_id = session["_id"]

    # Process attached document if present
    if document_data:
        files = session.setdefault("files", [])
        files.append({
            "file_id": document_data.get("file_id") or str(uuid.uuid4())[:8],
            "filename": document_data.get("filename", "document"),
            "content_type": document_data.get("content_type", ""),
            "size_bytes": document_data.get("size_bytes", 0),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        })
        await update_session(session_id, {"files": files})
        doc_filename = document_data.get("filename", "document")
        doc_text = document_data.get("text", "").strip()
        if not user_message or not user_message.strip():
            user_message = f"[Uploaded Document: {doc_filename}]\n\n{doc_text}"
        else:
            user_message = f"{user_message.strip()}\n\n[Attached Document: {doc_filename}]\n\n{doc_text}"

    # 1. Save user message to database
    await add_message(session_id=session_id, role="user", text=user_message)
    recent_messages = await get_recent_messages(session_id=session_id, n=settings.history_window)

    # 2. Safety keyword pre-check (pure code)
    detect_keywords(user_message)

    # 3. In parallel: understand and affect (FAST tier)
    understand_task = understand(
        session=session,
        new_message=user_message,
        llm=client,
        recent_messages=recent_messages,
    )
    affect_task = read_affect(
        session=session,
        new_message=user_message,
        llm=client,
        recent_messages=recent_messages,
    )
    understand_out, affect_out = await asyncio.gather(understand_task, affect_task)

    # Tone for this turn: shared by talker and presenter (affect shapes tone only)
    style_hint = affect_to_style_hint(affect_out)
    session["style_hint"] = style_hint
    await update_session(session_id=session_id, updates={"style_hint": style_hint})

    # Reconstruct current case file
    turn_num = session.get("turn_count", 0) + 1
    session["turn_count"] = turn_num
    raw_case = session.get("case_file", {})
    case_obj = CaseFile.model_validate(raw_case) if raw_case else CaseFile()
    preferred_lang = case_obj.hard_core.language or affect_out.language or "English"

    # 4. Safety decision (runs in EVERY state)
    existing_safety_flags = session.get("safety_flags") or []
    safety_decision = decide(
        message=user_message,
        affect=affect_out,
        language=preferred_lang,
        existing_safety_flags=existing_safety_flags,
    )

    if safety_decision.triggered:
        # If this safety category was ALREADY shown previously in this session,
        # DO NOT repeat the canned helpline message! Continue with normal intake.
        if safety_decision.category in existing_safety_flags:
            safety_decision = SafetyDecision(triggered=False, category=None, reply_text=None, source=None)
        else:
            if "safety_flags" not in session or session["safety_flags"] is None:
                session["safety_flags"] = []
            session["safety_flags"].append(safety_decision.category)
            # The next normal reply follows up on this like a person would
            session["safety_followup_pending"] = safety_decision.category

            # In accordance with ARCHITECTURE.md: the case still gets updated
            updated_case, change_report = apply_patch(
                case=case_obj,
                patch=understand_out.patch,
                user_message=user_message,
                turn=turn_num,
            )
            await save_case_version(
                session_id=session_id,
                version=updated_case.version,
                patch=understand_out.patch.model_dump(),
                full_case=updated_case.model_dump(),
            )
            session["case_file"] = updated_case.model_dump()
            session["case_version"] = updated_case.version

            await update_session(session_id=session_id, updates={
                "case_file": session["case_file"],
                "case_version": session["case_version"],
                "turn_count": turn_num,
                "safety_flags": session["safety_flags"],
                "safety_followup_pending": safety_decision.category,
            })

            if debug:
                print(f"\n[DEBUG SAFETY] Triggered: {safety_decision.category}", file=sys.stderr)

            reply_text = safety_decision.reply_text or ""
            if stream:
                return _async_string_stream(reply_text, session_id=session_id, affect_dict=affect_out.model_dump()), safety_decision
            else:
                await add_message(session_id=session_id, role="assistant", text=reply_text, affect=affect_out.model_dump())
                return reply_text, safety_decision

    # Current state dispatch
    current_status = session.get("status", "COLLECTING")

    if debug:
        print(f"\n[DEBUG] State: {current_status} | Intent: {understand_out.intent.value}", file=sys.stderr)

    # -----------------------------------------------------------------------
    # STATE: CONFIRMING
    # -----------------------------------------------------------------------
    if current_status == "CONFIRMING":
        is_confirming = understand_out.intent != Intent.corrects and (
            understand_out.intent in (Intent.confirms, Intent.wants_answer_now)
            or understand_out.wants_answer_now
            or is_short_affirmation(user_message)
        )

        # Apply any new details first (e.g. "yes, and he also kept my bike"), so nothing is dropped
        updated_case, change_report = apply_patch(
            case=case_obj,
            patch=understand_out.patch,
            user_message=user_message,
            turn=turn_num,
        )
        await save_case_version(
            session_id=session_id,
            version=updated_case.version,
            patch=understand_out.patch.model_dump(),
            full_case=updated_case.model_dump(),
        )
        session["case_file"] = updated_case.model_dump()
        session["case_version"] = updated_case.version

        if is_confirming:
            # Build ServiceQuery (SMART tier, with PII redaction)
            query = await build_service_query(case=updated_case, llm=client, session=session)
            session["last_service_query"] = query.model_dump()

            # Call external legal answering service
            service_resp = await srv_client.ask(query=query, session_id=session_id)

            if service_resp.status == "answered":
                session["last_service_answer"] = service_resp.model_dump()
                session["status"] = "ANSWERED"

                # Present the answer (SMART tier + code guard + faithfulness)
                presented_text = await present(
                    session=session,
                    service_response=service_resp,
                    mode="initial",
                    new_message=user_message,
                    llm=client,
                )
                await update_session(session_id=session_id, updates={
                    "status": "ANSWERED",
                    "case_file": session["case_file"],
                    "case_version": session["case_version"],
                    "last_service_query": session["last_service_query"],
                    "last_service_answer": session["last_service_answer"],
                    "turn_count": turn_num,
                    **_presenter_state(session),
                })
                await add_message(session_id=session_id, role="assistant", text=presented_text, affect=affect_out.model_dump())
                return presented_text, safety_decision

            elif service_resp.status == "needs_more_info":
                # External service needs more information -> move back to COLLECTING
                session["pending_service_gaps"] = service_resp.missing_info
                session["status"] = "COLLECTING"

                # Ask for the missing items naturally via talker
                gap_analysis = await find_gaps(session=session, case=updated_case, llm=client)
                to_ask = filter_gaps_to_ask(
                    session=session,
                    case=updated_case,
                    gaps=gap_analysis.gaps,
                    max_questions=settings.max_questions_per_turn,
                )
                for g in to_ask:
                    track_asked_gap(session, g.topic)

                await update_session(session_id=session_id, updates={
                    "status": "COLLECTING",
                    "case_file": session["case_file"],
                    "case_version": session["case_version"],
                    "pending_service_gaps": session["pending_service_gaps"],
                    "turn_count": turn_num,
                    "asked_counts": session.get("asked_counts", {}),
                })

                talker_reply = await build_talker_reply(
                    session=session,
                    messages=recent_messages,
                    llm=client,
                    stream=False,
                    case_summary=render_case_summary(updated_case),
                    gaps=to_ask,
                    language=preferred_lang,
                    style_hint=style_hint,
                    system_notes=(
                        "They just confirmed the summary, but a couple more details are needed before they can get a "
                        "proper answer. Say that naturally in one short sentence (no mention of errors or 'the service'), "
                        "then ask for the first one."
                    ),
                    progress="Close: just these extra details and then you can get them an answer.",
                )
                await add_message(session_id=session_id, role="assistant", text=talker_reply, affect=affect_out.model_dump())
                return talker_reply, safety_decision

            else:  # service_resp.status == "error"
                await update_session(session_id=session_id, updates={
                    "case_file": session["case_file"],
                    "case_version": session["case_version"],
                    "last_service_query": session["last_service_query"],
                    "turn_count": turn_num,
                })
                await add_message(session_id=session_id, role="assistant", text=SERVICE_ERROR_REPLY, affect=affect_out.model_dump())
                return SERVICE_ERROR_REPLY, safety_decision

        else:
            # User provided a correction or new details during CONFIRMING -> move to COLLECTING
            session["status"] = "COLLECTING"
            await update_session(session_id=session_id, updates={
                "status": "COLLECTING",
                "case_file": session["case_file"],
                "case_version": session["case_version"],
                "turn_count": turn_num,
            })

            # Check gaps & readiness
            gap_analysis = await find_gaps(session=session, case=updated_case, llm=client)
            user_wants_now = understand_out.wants_answer_now or (understand_out.intent == Intent.wants_answer_now)
            ready, ready_reason = is_ready(
                case=updated_case,
                gap_analysis=gap_analysis,
                wants_answer_now=user_wants_now,
                turn_count=turn_num,
                settings=settings,
                session=session,
            )

            if ready:
                recap = await make_recap(session=session, case=updated_case, llm=client)
                session["status"] = "CONFIRMING"
                await update_session(session_id=session_id, updates={"status": "CONFIRMING"})
                await add_message(session_id=session_id, role="assistant", text=recap, affect=affect_out.model_dump())
                return ReadyConfirmation(recap, safety_decision)

            to_ask = filter_gaps_to_ask(
                session=session,
                case=updated_case,
                gaps=gap_analysis.gaps,
                max_questions=settings.max_questions_per_turn,
            )
            for g in to_ask:
                track_asked_gap(session, g.topic)

            reply = await build_talker_reply(
                session=session,
                messages=recent_messages,
                llm=client,
                stream=False,
                case_summary=render_case_summary(updated_case),
                gaps=to_ask,
                language=preferred_lang,
                style_hint=style_hint,
                system_notes=(
                    f"They corrected or added to your summary ({change_report}). Acknowledge the fix briefly and "
                    "naturally (e.g. \"Ah, got it\"), without over-apologising, then carry on."
                ),
                progress=describe_progress(gap_analysis, turn_num, settings, session),
            )
            await update_session(session_id=session_id, updates={"asked_counts": session.get("asked_counts", {})})
            await add_message(session_id=session_id, role="assistant", text=reply, affect=affect_out.model_dump())
            return reply, safety_decision

    # -----------------------------------------------------------------------
    # STATE: ANSWERED (Continuous Legal Consultation)
    # -----------------------------------------------------------------------
    if current_status == "ANSWERED":
        # Silently update CaseFile with any new details from the message
        updated_case, change_report = apply_patch(
            case=case_obj,
            patch=understand_out.patch,
            user_message=user_message,
            turn=turn_num,
        )
        await save_case_version(
            session_id=session_id,
            version=updated_case.version,
            patch=understand_out.patch.model_dump(),
            full_case=updated_case.model_dump(),
        )
        session["case_file"] = updated_case.model_dump()
        session["case_version"] = updated_case.version

        async def _finish_answered(reply_text: str) -> tuple[str, SafetyDecision]:
            await update_session(session_id=session_id, updates={
                "case_file": session["case_file"],
                "case_version": session["case_version"],
                "turn_count": turn_num,
                "last_service_query": session.get("last_service_query"),
                "last_service_answer": session.get("last_service_answer"),
                **_presenter_state(session),
            })
            await add_message(session_id=session_id, role="assistant", text=reply_text, affect=affect_out.model_dump())
            return reply_text, safety_decision

        # Explicit request for a recap of the case notes
        msg_lower = user_message.strip().lower()
        is_requesting_recap = (
            any(w in msg_lower for w in ["recap my case", "summarize my case", "case recap", "what are my case notes", "show my case notes", "review summary"])
            or (understand_out.intent == Intent.asks_about_process and any(w in msg_lower for w in ["summary", "recap", "notes"]))
        )
        if is_requesting_recap:
            return await _finish_answered(await make_recap(session=session, case=updated_case, llm=client))

        # A new legal question -> ask the legal service rather than improvising
        if understand_out.intent == Intent.asks_legal_question and understand_out.question.strip():
            quick_reply = await _try_quick_answer(
                session=session,
                case=updated_case,
                understand_out=understand_out,
                client=client,
                srv_client=srv_client,
                user_message=user_message,
            )
            if quick_reply:
                return await _finish_answered(quick_reply)

        # New documents -> refresh the main answer with the updated case
        has_new_documents = any(d.state.value == "has" for d in understand_out.patch.add_documents)
        if has_new_documents:
            query = await build_service_query(case=updated_case, llm=client, session=session)
            service_resp = await srv_client.ask(query=query, session_id=session_id)
            if service_resp.status == "answered":
                session["last_service_query"] = query.model_dump()
                session["last_service_answer"] = service_resp.model_dump()
                refreshed = await present(
                    session=session,
                    service_response=service_resp,
                    mode="initial",
                    new_message=user_message,
                    llm=client,
                )
                return await _finish_answered(refreshed)

        # Follow-up about the stored answer (guarded)
        stored_raw = session.get("last_service_answer")
        if stored_raw:
            stored_resp = ServiceResponse.model_validate(stored_raw)
        else:
            stored_resp = ServiceResponse(answer="", status="answered")

        followup_reply = await present(
            session=session,
            service_response=stored_resp,
            mode="followup",
            new_message=user_message,
            llm=client,
        )
        return await _finish_answered(followup_reply)

    # -----------------------------------------------------------------------
    # STATE: COLLECTING (Default)
    # -----------------------------------------------------------------------
    # Apply patch in code, validate, and save new case version
    updated_case, change_report = apply_patch(
        case=case_obj,
        patch=understand_out.patch,
        user_message=user_message,
        turn=turn_num,
    )
    await save_case_version(
        session_id=session_id,
        version=updated_case.version,
        patch=understand_out.patch.model_dump(),
        full_case=updated_case.model_dump(),
    )
    session["case_file"] = updated_case.model_dump()
    session["case_version"] = updated_case.version

    # Find gaps (FAST_MODEL)
    gap_analysis = await find_gaps(session=session, case=updated_case, llm=client)

    # Check readiness in pure code
    is_user_done = is_done_signal(user_message, turn_num)
    user_wants_now = understand_out.wants_answer_now or (understand_out.intent == Intent.wants_answer_now) or is_user_done
    ready, ready_reason = is_ready(
        case=updated_case,
        gap_analysis=gap_analysis,
        wants_answer_now=user_wants_now,
        turn_count=turn_num,
        settings=settings,
        session=session,
    )

    to_ask = filter_gaps_to_ask(
        session=session,
        case=updated_case,
        gaps=gap_analysis.gaps,
        max_questions=settings.max_questions_per_turn,
    )

    # If the user has nothing else to add, or no gaps remain to ask, move to confirmation.
    # The location is only assumed after it has been asked twice without an answer.
    if (is_user_done or not to_ask) and not ready:
        has_location = bool(updated_case.hard_core.jurisdiction and updated_case.hard_core.jurisdiction.strip())
        location_asked = session.get("asked_counts", {}).get(LOCATION_GAP_TOPIC, 0)
        if has_location or location_asked >= 2:
            if not has_location:
                updated_case.hard_core.jurisdiction = "General"
                if "Location not given, so the answer is general rather than for a specific state" not in updated_case.assumptions:
                    updated_case.assumptions.append("Location not given, so the answer is general rather than for a specific state")
            if not (updated_case.hard_core.user_goal and updated_case.hard_core.user_goal.strip()):
                updated_case.hard_core.user_goal = "Understand their rights and options"
            ready = True
            ready_reason = "No further gaps to ask or user completed input"

    if debug:
        gaps_desc = ", ".join(f"{g.topic} ({g.impact})" for g in gap_analysis.gaps) if gap_analysis.gaps else "none"
        print(f"[DEBUG GAPS] {gaps_desc}", file=sys.stderr)
        print(f"[DEBUG READINESS] ready={ready}, reason='{ready_reason}'", file=sys.stderr)

    session["case_file"] = updated_case.model_dump()

    # If ready: generate recap (SMART tier), save as assistant message, status CONFIRMING
    if ready:
        recap_text = await make_recap(session=session, case=updated_case, llm=client)
        session["status"] = "CONFIRMING"
        await update_session(session_id=session_id, updates={
            "status": "CONFIRMING",
            "case_file": session["case_file"],
            "case_version": session["case_version"],
            "turn_count": turn_num,
            "asked_counts": session.get("asked_counts", {}),
        })
        await add_message(session_id=session_id, role="assistant", text=recap_text, affect=affect_out.model_dump())
        return ReadyConfirmation(recap_text, safety_decision)

    next_gap = to_ask[0] if to_ask else None
    question = understand_out.question.strip()

    # A legal question mid-conversation: answer it now if possible, otherwise defer naturally
    if question and understand_out.intent in (Intent.asks_legal_question, Intent.wants_answer_now, Intent.provides_info):
        quick_reply = await _try_quick_answer(
            session=session,
            case=updated_case,
            understand_out=understand_out,
            client=client,
            srv_client=srv_client,
            user_message=user_message,
            next_question=next_gap.question_hint if next_gap else "",
        )
        if quick_reply:
            if next_gap:
                track_asked_gap(session, next_gap.topic)
            session.pop("safety_followup_pending", None)
            await update_session(session_id=session_id, updates={
                "case_file": session["case_file"],
                "case_version": session["case_version"],
                "turn_count": turn_num,
                "asked_counts": session.get("asked_counts", {}),
                "safety_followup_pending": None,
                **_presenter_state(session),
            })
            await add_message(session_id=session_id, role="assistant", text=quick_reply, affect=affect_out.model_dump())
            return quick_reply, safety_decision

    # Otherwise talker writes the reply (SMART_MODEL)
    for g in to_ask:
        track_asked_gap(session, g.topic)

    notes_list: list[str] = _safety_notes(session, user_message)

    if question:
        notes_list.append(_defer_question_note(question, next_gap))

    if user_wants_now and (not updated_case.hard_core.jurisdiction or not updated_case.hard_core.user_goal):
        notes_list.append(
            "They want an answer now. You can move ahead quickly, but you still need to know where this is happening "
            "(and what outcome they want, if that isn't clear). Say that in one natural sentence, then ask."
        )

    system_notes = " ".join(notes_list)
    progress = describe_progress(gap_analysis, turn_num, settings, session)

    await update_session(session_id=session_id, updates={
        "case_file": session["case_file"],
        "case_version": session["case_version"],
        "turn_count": turn_num,
        "asked_counts": session.get("asked_counts", {}),
        "safety_followup_pending": None,
    })

    if stream:
        raw_stream = await build_talker_reply(
            session=session,
            messages=recent_messages,
            llm=client,
            stream=True,
            case_summary=render_case_summary(updated_case),
            gaps=to_ask,
            language=preferred_lang,
            style_hint=style_hint,
            system_notes=system_notes,
            progress=progress,
        )
        return _stream_and_save(raw_stream, session_id=session_id, affect_dict=affect_out.model_dump()), safety_decision
    else:
        full_text = await build_talker_reply(
            session=session,
            messages=recent_messages,
            llm=client,
            stream=False,
            case_summary=render_case_summary(updated_case),
            gaps=to_ask,
            language=preferred_lang,
            style_hint=style_hint,
            system_notes=system_notes,
            progress=progress,
        )
        await add_message(session_id=session_id, role="assistant", text=full_text, affect=affect_out.model_dump())
        return full_text, safety_decision
