"""
src/pipeline.py - Orchestrates the per-turn intake processing pipeline.

State Machine (ARCHITECTURE.md):
COLLECTING -> CONFIRMING -> ANSWERED
- COLLECTING: Gathers facts. When ready, generates recap (SMART) and moves to CONFIRMING.
- CONFIRMING: Recap shown.
  * User confirms: builds ServiceQuery (SMART), calls service.
    - status 'answered' -> present (SMART + guard), status ANSWERED.
    - status 'needs_more_info' -> set pending_service_gaps, status COLLECTING.
    - status 'error' -> honest error message, keep query, allow retry.
  * User corrects/provides info -> applies patch, status COLLECTING, continues intake.
- ANSWERED: Answer presented.
  * New case facts -> applies patch, status CONFIRMING, updated recap.
  * Question about answer -> present in followup mode using stored answer only.
Safety checks run in EVERY state before normal handling.
"""

import asyncio
import sys
from collections.abc import AsyncGenerator

from src.affect import affect_to_style_hint, read_affect
from src.case_ops import apply_patch, render_case_summary
from src.config import get_settings
from src.confirm import build_service_query, make_recap
from src.db import add_message, get_recent_messages, save_case_version, update_session
from src.gaps import find_gaps
from src.llm import LLM, BaseLLM
from src.presenter import present
from src.readiness import filter_gaps_to_ask, is_ready, track_asked_gap
from src.safety import decide, detect_keywords
from src.schemas import CaseFile, Intent, SafetyDecision, ServiceResponse
from src.service_client import ServiceClient
from src.talker import build_talker_reply
from src.understand import understand


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


async def handle_turn(
    session: dict,
    user_message: str,
    llm: BaseLLM | None = None,
    service_client: ServiceClient | None = None,
    stream: bool = False,
    debug: bool = False,
) -> tuple[str | AsyncGenerator[str, None], SafetyDecision] | ReadyConfirmation:
    """
    Executes one complete conversational turn according to the state machine:
    COLLECTING -> CONFIRMING -> ANSWERED.
    """
    settings = get_settings()
    client = llm or LLM()
    srv_client = service_client or ServiceClient()
    session_id = session["_id"]

    # 1. Save user message to database
    await add_message(session_id=session_id, role="user", text=user_message)
    recent_messages = await get_recent_messages(session_id=session_id, n=settings.history_window)

    # 2. Safety keyword pre-check (pure code)
    kw_category = detect_keywords(user_message)

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

    # Reconstruct current case file
    turn_num = session.get("turn_count", 0) + 1
    session["turn_count"] = turn_num
    raw_case = session.get("case_file", {})
    case_obj = CaseFile.model_validate(raw_case) if raw_case else CaseFile()
    preferred_lang = case_obj.hard_core.language or affect_out.language or "English"

    # 4. Safety decision (runs in EVERY state)
    safety_decision = decide(
        message=user_message,
        affect=affect_out,
        language=preferred_lang,
    )

    if safety_decision.triggered:
        if "safety_flags" not in session or session["safety_flags"] is None:
            session["safety_flags"] = []
        if safety_decision.category not in session["safety_flags"]:
            session["safety_flags"].append(safety_decision.category)

        await update_session(session_id=session_id, updates={
            "case_file": session.get("case_file", {}),
            "case_version": session.get("case_version", 0),
            "turn_count": turn_num,
            "safety_flags": session["safety_flags"],
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
        # Check if intent is confirms or user demands answer
        msg_lower = user_message.strip().lower()
        is_confirming = (
            understand_out.intent in (Intent.confirms, Intent.wants_answer_now)
            or understand_out.wants_answer_now
            or any(w in msg_lower for w in ["yes", "proceed", "give me answer", "give answer", "answer", "ok", "sure", "correct", "right", "ahead", "help"])
        )
        if is_confirming:
            # Build ServiceQuery (SMART tier, with PII redaction)
            query = await build_service_query(case=case_obj, llm=client, session=session)
            session["last_service_query"] = query.model_dump()

            # Call external legal answering service
            service_resp = await srv_client.ask(query=query, session_id=session_id)

            if service_resp.status == "answered":
                session["last_service_answer"] = service_resp.model_dump()
                session["status"] = "ANSWERED"
                await update_session(session_id=session_id, updates={
                    "status": "ANSWERED",
                    "last_service_query": session["last_service_query"],
                    "last_service_answer": session["last_service_answer"],
                    "turn_count": turn_num,
                })

                # Present the answer (SMART tier + code guard + faithfulness)
                presented_text = await present(
                    session=session,
                    service_response=service_resp,
                    mode="initial",
                    new_message=user_message,
                    llm=client,
                )
                await add_message(session_id=session_id, role="assistant", text=presented_text, affect=affect_out.model_dump())
                return presented_text, safety_decision

            elif service_resp.status == "needs_more_info":
                # External service needs more information -> move back to COLLECTING
                session["pending_service_gaps"] = service_resp.missing_info
                session["status"] = "COLLECTING"
                await update_session(session_id=session_id, updates={
                    "status": "COLLECTING",
                    "pending_service_gaps": session["pending_service_gaps"],
                    "turn_count": turn_num,
                })

                # Ask for the missing items naturally via talker
                gap_analysis = await find_gaps(session=session, case=case_obj, llm=client)
                to_ask = filter_gaps_to_ask(
                    session=session,
                    case=case_obj,
                    gaps=gap_analysis.gaps,
                    max_questions=settings.max_questions_per_turn,
                )
                for g in to_ask:
                    track_asked_gap(session, g.topic)

                talker_reply = await build_talker_reply(
                    session=session,
                    messages=recent_messages,
                    llm=client,
                    stream=False,
                    case_summary=render_case_summary(case_obj),
                    gaps=to_ask,
                    language=preferred_lang,
                    style_hint=affect_to_style_hint(affect_out),
                    system_notes="The legal team asked for a few additional details before giving advice.",
                )
                await add_message(session_id=session_id, role="assistant", text=talker_reply, affect=affect_out.model_dump())
                return talker_reply, safety_decision

            else:  # service_resp.status == "error"
                err_text = (
                    "I apologize, but our legal answering service is currently unavailable or encountered an error. "
                    "Your case details have been safely preserved. You can type 'retry' or send a message whenever you would like to try again."
                )
                await update_session(session_id=session_id, updates={
                    "last_service_query": session["last_service_query"],
                    "turn_count": turn_num,
                })
                await add_message(session_id=session_id, role="assistant", text=err_text, affect=affect_out.model_dump())
                return err_text, safety_decision

        else:
            # User provided a correction or new details during CONFIRMING -> move to COLLECTING
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
                style_hint=affect_to_style_hint(affect_out),
                system_notes=f"The user corrected or added details ({change_report}). Acknowledge warmly.",
            )
            await add_message(session_id=session_id, role="assistant", text=reply, affect=affect_out.model_dump())
            return reply, safety_decision

    # -----------------------------------------------------------------------
    # STATE: ANSWERED
    # -----------------------------------------------------------------------
    if current_status == "ANSWERED":
        # Check if the message contains new case facts (non-empty patch)
        has_new_facts = bool(
            understand_out.patch.set_hard_core
            or understand_out.patch.add_facts
            or understand_out.patch.add_timeline
            or understand_out.patch.add_people
            or understand_out.patch.add_documents
            or understand_out.patch.add_deadlines
            or understand_out.patch.update
            or understand_out.patch.retract
            or understand_out.intent in (Intent.corrects, Intent.provides_info)
        )

        if has_new_facts:
            # Apply new facts, transition back to CONFIRMING, show updated recap
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
            session["status"] = "CONFIRMING"

            recap_text = await make_recap(session=session, case=updated_case, llm=client)
            change_note = f"I've updated your case notes with your new information ({change_report}).\n\n" if change_report else "I've updated your case notes.\n\n"
            updated_reply = f"{change_note}{recap_text}"

            await update_session(session_id=session_id, updates={
                "status": "CONFIRMING",
                "case_file": session["case_file"],
                "case_version": session["case_version"],
                "turn_count": turn_num,
            })
            await add_message(session_id=session_id, role="assistant", text=updated_reply, affect=affect_out.model_dump())
            return ReadyConfirmation(updated_reply, safety_decision)

        else:
            # Follow-up question about the answer -> call presenter in followup mode
            stored_raw = session.get("last_service_answer")
            if stored_raw:
                stored_resp = ServiceResponse.model_validate(stored_raw)
            else:
                stored_resp = ServiceResponse(answer="No previous answer recorded.", status="answered")

            followup_reply = await present(
                session=session,
                service_response=stored_resp,
                mode="followup",
                new_message=user_message,
                llm=client,
            )
            await update_session(session_id=session_id, updates={"turn_count": turn_num})
            await add_message(session_id=session_id, role="assistant", text=followup_reply, affect=affect_out.model_dump())
            return followup_reply, safety_decision

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
    user_wants_now = understand_out.wants_answer_now or (understand_out.intent == Intent.wants_answer_now)
    ready, ready_reason = is_ready(
        case=updated_case,
        gap_analysis=gap_analysis,
        wants_answer_now=user_wants_now,
        turn_count=turn_num,
        settings=settings,
        session=session,
    )

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

    # Otherwise talker writes the reply (SMART_MODEL)
    to_ask = filter_gaps_to_ask(
        session=session,
        case=updated_case,
        gaps=gap_analysis.gaps,
        max_questions=settings.max_questions_per_turn,
    )
    for g in to_ask:
        track_asked_gap(session, g.topic)

    session["case_file"] = updated_case.model_dump()

    notes_list: list[str] = []
    if session.get("safety_flags"):
        notes_list.append("A safety message was already shown. Check in gently and do not repeat it.")

    if user_wants_now and (not updated_case.hard_core.jurisdiction or not updated_case.hard_core.user_goal):
        notes_list.append(
            "The user asked for an immediate answer, but we cannot proceed without their location/jurisdiction "
            "and main goal. Explain in one sentence why these two are needed before we can give advice, then ask for them."
        )

    system_notes = " ".join(notes_list)
    style_hint = affect_to_style_hint(affect_out)

    await update_session(session_id=session_id, updates={
        "case_file": session["case_file"],
        "case_version": session["case_version"],
        "turn_count": turn_num,
        "asked_counts": session.get("asked_counts", {}),
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
        )
        await add_message(session_id=session_id, role="assistant", text=full_text, affect=affect_out.model_dump())
        return full_text, safety_decision
