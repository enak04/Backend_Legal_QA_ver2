"""
src/feedback.py - Optional user feedback on bot messages and whole conversations.

Feedback is stored for review only (see evals/feedback_report.py). It NEVER changes the
case file, the prompts, or the query sent to the legal service.

Each message rating stores a snapshot of the rated message and its `meta` (kind, prompt
name/version, presenter path), so dislikes can be traced to the prompt version that produced them.
"""

from src.db import get_all_messages, get_session, upsert_feedback
from src.redact import redact_text
from src.schemas import FeedbackRequest

# Suggested reason chips per message kind. "up" applies to thumbs-up on anything.
_ANSWER_REASONS = [
    "Didn't answer my question",
    "Hard to understand",
    "Seems wrong",
    "Not helpful",
    "Tone felt off",
]
FEEDBACK_REASONS: dict[str, list[str]] = {
    "question": [
        "Not relevant to my situation",
        "You already asked this",
        "Confusing",
        "Too many questions",
        "Tone felt off",
    ],
    "answer": _ANSWER_REASONS,
    "quick_answer": _ANSWER_REASONS,
    "followup": _ANSWER_REASONS,
    "recap": ["Got my story wrong", "Missed something important", "Tone felt off"],
    "conversation": ["Took too long", "Didn't get a clear answer", "Felt robotic", "Felt understood", "Got what I needed"],
    "up": ["Clear", "Helpful", "Felt understood"],
}

ALL_REASONS: set[str] = {r for reasons in FEEDBACK_REASONS.values() for r in reasons}

SNAPSHOT_TEXT_LIMIT = 2000


class FeedbackError(ValueError):
    """Invalid feedback (unknown message, user message, unknown reason)."""


class SessionNotFound(LookupError):
    """The session does not exist."""


async def record_feedback(session_id: str, req: FeedbackRequest) -> dict:
    """
    Validates and stores one feedback entry (create or overwrite).
    Raises SessionNotFound or FeedbackError.
    """
    session = await get_session(session_id)
    if not session:
        raise SessionNotFound(session_id)

    unknown = [r for r in req.reasons if r not in ALL_REASONS]
    if unknown:
        raise FeedbackError(f"Unknown feedback reason(s): {unknown}")

    if req.target == "message":
        messages = await get_all_messages(session_id)
        msg = next((m for m in messages if str(m.get("_id")) == req.message_id), None)
        if msg is None:
            raise FeedbackError("Message not found in this conversation.")
        if msg.get("role") != "assistant":
            raise FeedbackError("Only the assistant's messages can be rated.")
        snapshot = {
            "text": (msg.get("text") or "")[:SNAPSHOT_TEXT_LIMIT],
            **(msg.get("meta") or {"kind": "unknown"}),
        }
    else:
        snapshot = {
            "status": session.get("status", "COLLECTING"),
            "turn_count": session.get("turn_count", 0),
            "message_count": len(session.get("messages") or []),
        }

    doc = await upsert_feedback(
        session_id=session_id,
        target=req.target,
        message_id=req.message_id,
        rating=req.rating,
        reasons=list(dict.fromkeys(req.reasons)),
        comment=redact_text(req.comment),
        snapshot=snapshot,
    )
    return {
        "feedback_id": doc["_id"],
        "target": req.target,
        "message_id": req.message_id,
        "rating": req.rating,
    }


def feedback_by_message(feedback_docs: list[dict]) -> dict[str, str]:
    """Maps message_id -> rating, for showing existing ratings when a conversation is reloaded."""
    return {d["message_id"]: d["rating"] for d in feedback_docs if d.get("target") == "message" and d.get("message_id")}
