"""
src/pipeline.py - Orchestrates the per-turn intake processing pipeline.

Per-turn pipeline steps (ARCHITECTURE.md / COLLECTING state):
1. Save the user message to database.
2. Safety pre-check (keywords in code).
3. In parallel (asyncio.gather): understand (extract case patch + intent) and affect (emotion + crisis flags).
4. Apply the patch in code, validate, and save a new case version.
5. Safety decision: keyword hit OR affect crisis flags -> hard-coded message (case still gets updated, talker NOT called).
6. On next normal turn, if safety was previously shown, pass a system_notes line to talker.
7. Otherwise call talker with style_hint derived from affect_to_style_hint.
8. Save the assistant message with affect data.
"""

import asyncio
from collections.abc import AsyncGenerator
import sys
from typing import Any

from src.affect import affect_to_style_hint, read_affect
from src.case_ops import apply_patch, render_case_summary
from src.config import get_settings
from src.db import add_message, get_recent_messages, save_case_version, update_session
from src.gaps import find_gaps
from src.llm import BaseLLM, LLM
from src.readiness import filter_gaps_to_ask, is_ready, track_asked_gap
from src.safety import decide, detect_keywords
from src.schemas import AffectReading, CaseFile, Intent, SafetyDecision
from src.talker import build_talker_reply
from src.understand import understand


class ReadyConfirmation(str):
    """
    String equal to 'READY_FOR_CONFIRMATION' that also supports unpacking
    as a 2-tuple: (reply, safety_decision).
    """
    decision: SafetyDecision

    def __new__(cls, text: str = "READY_FOR_CONFIRMATION", decision: SafetyDecision | None = None) -> "ReadyConfirmation":
        obj = super().__new__(cls, text)
        obj.decision = decision or SafetyDecision()
        return obj

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
    stream: bool = False,
    debug: bool = False,
) -> tuple[str | AsyncGenerator[str, None], SafetyDecision]:
    """
    Executes one complete conversational turn according to the COLLECTING pipeline:
    1. Save user message to database
    2. Safety pre-check on keywords (pure code)
    3. In parallel (asyncio.gather): understand and affect
    4. Apply patch in code, validate, and save new case version
    5. Safety decision: keyword hit OR affect crisis flags
    6. Find gaps (FAST_MODEL)
    7. Check readiness (pure code)
       If ready: set session status to CONFIRMING and return 'READY_FOR_CONFIRMATION'
    8. Otherwise talker writes the reply (SMART_MODEL)
    """
    settings = get_settings()
    client = llm or LLM()
    session_id = session["_id"]

    # 1. Save user message to database
    await add_message(session_id=session_id, role="user", text=user_message)

    # Retrieve recent messages for context
    recent_messages = await get_recent_messages(session_id=session_id, n=settings.history_window)

    # 2. Safety pre-check on keywords (pure code)
    kw_category = detect_keywords(user_message)

    # 3. In parallel (asyncio.gather): understand and affect
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

    # 4. Apply patch in code, validate, and save new case version
    turn_num = session.get("turn_count", 0) + 1
    raw_case = session.get("case_file", {})
    case_obj = CaseFile.model_validate(raw_case) if raw_case else CaseFile()
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

    # Update session state with case information
    session["case_file"] = updated_case.model_dump()
    session["case_version"] = updated_case.version
    session["turn_count"] = turn_num

    # 5. Safety decision: keyword hit OR affect crisis flags
    preferred_lang = updated_case.hard_core.language or affect_out.language or "English"
    safety_decision = decide(
        message=user_message,
        affect=affect_out,
        language=preferred_lang,
    )

    if safety_decision.triggered:
        # Record category in session safety flags (safety escalation is a flag, not a state)
        if "safety_flags" not in session or session["safety_flags"] is None:
            session["safety_flags"] = []
        if safety_decision.category not in session["safety_flags"]:
            session["safety_flags"].append(safety_decision.category)

        await update_session(session_id=session_id, updates={
            "case_file": session["case_file"],
            "case_version": session["case_version"],
            "turn_count": turn_num,
            "safety_flags": session["safety_flags"],
        })

        if debug:
            print(f"\n[DEBUG] Intent: {understand_out.intent.value}", file=sys.stderr)
            print(
                f"[DEBUG] Affect: distress={affect_out.distress}, urgency={affect_out.urgency}, "
                f"self_harm={affect_out.crisis_self_harm}, violence={affect_out.crisis_violence_or_abuse}, "
                f"detention={affect_out.crisis_detention}",
                file=sys.stderr,
            )
            print(f"[DEBUG] Safety: triggered={safety_decision.triggered}, category={safety_decision.category}", file=sys.stderr)

        reply_text = safety_decision.reply_text or ""
        # The talker is NOT called
        if stream:
            return _async_string_stream(reply_text, session_id=session_id, affect_dict=affect_out.model_dump()), safety_decision
        else:
            await add_message(session_id=session_id, role="assistant", text=reply_text, affect=affect_out.model_dump())
            return reply_text, safety_decision

    # 6. Find gaps using FAST_MODEL
    gap_analysis = await find_gaps(session=session, case=updated_case, llm=client)

    # 7. Check readiness in pure code
    user_wants_now = understand_out.wants_answer_now or (understand_out.intent == Intent.wants_answer_now)
    ready, ready_reason = is_ready(
        case=updated_case,
        gap_analysis=gap_analysis,
        wants_answer_now=user_wants_now,
        turn_count=turn_num,
        settings=settings,
        session=session,
    )

    # Print debug diagnostics
    if debug:
        print(f"\n[DEBUG] Intent: {understand_out.intent.value}", file=sys.stderr)
        print(
            f"[DEBUG] Affect: distress={affect_out.distress}, urgency={affect_out.urgency}, "
            f"anger={affect_out.anger}, confusion={affect_out.confusion}, "
            f"self_harm={affect_out.crisis_self_harm}, violence={affect_out.crisis_violence_or_abuse}, "
            f"detention={affect_out.crisis_detention}",
            file=sys.stderr,
        )
        gaps_desc = ", ".join(f"{g.topic} ({g.impact})" for g in gap_analysis.gaps) if gap_analysis.gaps else "none"
        print(f"[DEBUG] Gaps: {gaps_desc}", file=sys.stderr)
        print(f"[DEBUG] Readiness: ready={ready}, reason='{ready_reason}'", file=sys.stderr)

    # Keep session['case_file'] synchronized in case is_ready added assumptions
    session["case_file"] = updated_case.model_dump()

    # If ready, transition state to CONFIRMING and return READY_FOR_CONFIRMATION
    if ready:
        session["status"] = "CONFIRMING"
        await update_session(session_id=session_id, updates={
            "status": "CONFIRMING",
            "case_file": session["case_file"],
            "case_version": session["case_version"],
            "turn_count": turn_num,
            "asked_counts": session.get("asked_counts", {}),
        })
        return ReadyConfirmation("READY_FOR_CONFIRMATION", safety_decision)

    # 8. Otherwise the talker writes the reply
    # Select top gaps to ask and track their asked counts
    to_ask = filter_gaps_to_ask(
        session=session,
        case=updated_case,
        gaps=gap_analysis.gaps,
        max_questions=settings.max_questions_per_turn,
    )
    for g in to_ask:
        track_asked_gap(session, g.topic)

    # Sync case_file again in case filter_gaps_to_ask added assumptions
    session["case_file"] = updated_case.model_dump()

    notes_list: list[str] = []
    if session.get("safety_flags"):
        notes_list.append("A safety message was already shown. Check in gently and do not repeat it.")

    # If the user wanted an answer right now but we can't proceed without hard core, explain why
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

