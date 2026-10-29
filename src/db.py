"""
src/db.py - MongoDB database operations with asynchronous wrappers.

Uses PyMongo for synchronous database calls and wraps them with asyncio.to_thread
for non-blocking asynchronous usage. Configures collections, compound indexes,
and TTL indexes as specified in ARCHITECTURE.md.
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any

import pymongo
from pymongo.database import Database

from src.config import get_settings

# Local storage path for offline dev persistence when MongoDB server is not running
LOCAL_STORE_PATH = Path(__file__).resolve().parent.parent / ".mongo_dev_data.json"

# Global database client and instance
_mongo_client: pymongo.MongoClient | None = None
_database: Database | None = None


def _json_serial(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def _convert_dates(doc: dict) -> dict:
    for k, v in doc.items():
        if isinstance(v, str) and (k in ("ts", "created_at", "updated_at", "timestamp") or k.endswith("_at") or k.endswith("_ts")):
            try:
                doc[k] = datetime.fromisoformat(v)
            except Exception:
                pass
        elif isinstance(v, dict):
            _convert_dates(v)
    return doc


def _persist_mock_if_needed(target_db: Database) -> None:
    if getattr(target_db, "_is_mock_persistence", False):
        try:
            collections = ["sessions", "messages", "case_versions", "llm_logs", "service_calls"]
            data = {}
            for col_name in collections:
                data[col_name] = list(target_db[col_name].find({}))
            LOCAL_STORE_PATH.write_text(json.dumps(data, default=_json_serial, indent=2), encoding="utf-8")
        except Exception:
            pass


def _create_persistent_mock_db(db_name: str) -> Database:
    try:
        import mongomock  # type: ignore[import-untyped,import-not-found]
    except ImportError as err:
        raise ImportError(
            "mongomock is required for offline dev fallback. Run 'pip install mongomock'."
        ) from err

    client: Any = mongomock.MongoClient()
    db = client[db_name]
    setattr(db, "_is_mock_persistence", True)

    if LOCAL_STORE_PATH.is_file():
        try:
            raw_data = json.loads(LOCAL_STORE_PATH.read_text(encoding="utf-8"))
            for col_name, docs in raw_data.items():
                if docs:
                    for doc in docs:
                        _convert_dates(doc)
                    db[col_name].insert_many(docs)
        except Exception:
            pass
    return db


def get_db(client: Any | None = None, db_name: str | None = None) -> Database:
    """
    Returns the PyMongo Database instance.
    If live MongoDB is unreachable, gracefully falls back to an offline persistent mock database.
    """
    global _mongo_client, _database

    if client is not None:
        name = db_name or "legal_intake_bot"
        return client[name]

    if _database is None:
        settings = get_settings()
        try:
            candidate: Any = pymongo.MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=1000)
            candidate.admin.command("ping")
            _mongo_client = candidate
            _database = _mongo_client[settings.db_name]
        except Exception:
            # Fall back to persistent mock database
            _database = _create_persistent_mock_db(settings.db_name)

        init_db(_database, ttl_days=settings.session_ttl_days)

    return _database


def set_db(db: Database) -> None:
    """Explicitly sets the active database instance (e.g. for testing)."""
    global _database
    _database = db


def init_db(db: Database, ttl_days: int = 30) -> None:
    """
    Creates compound indexes and TTL indexes as defined in ARCHITECTURE.md:
    - messages: (session_id, ts) and TTL on ts
    - case_versions: (session_id, version)
    - llm_logs: (session_id, ts) and TTL on ts
    - service_calls: TTL on ts
    """
    ttl_seconds = ttl_days * 86400

    # messages
    db.messages.create_index([("session_id", pymongo.ASCENDING), ("ts", pymongo.ASCENDING)])
    db.messages.create_index("ts", expireAfterSeconds=ttl_seconds)

    # case_versions
    db.case_versions.create_index([("session_id", pymongo.ASCENDING), ("version", pymongo.ASCENDING)])

    # llm_logs
    db.llm_logs.create_index([("session_id", pymongo.ASCENDING), ("ts", pymongo.ASCENDING)])
    db.llm_logs.create_index("ts", expireAfterSeconds=ttl_seconds)

    # service_calls
    db.service_calls.create_index("ts", expireAfterSeconds=ttl_seconds)

    # Migrate any existing sessions without embedded messages
    _migrate_embed_messages_sync(db)


def _migrate_embed_messages_sync(db: Database) -> None:
    """Consolidates flat messages into embedded session documents if not already present."""
    try:
        sessions = list(db.sessions.find({
            "$or": [
                {"messages": {"$exists": False}},
                {"messages": {"$size": 0}},
            ]
        }))
        for s in sessions:
            sid = s.get("_id")
            if not sid:
                continue
            msgs = list(db.messages.find({"session_id": sid}).sort("ts", pymongo.ASCENDING))
            if msgs:
                db.sessions.update_one({"_id": sid}, {"$set": {"messages": msgs}})
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Synchronous helper implementations
# ---------------------------------------------------------------------------

def _create_session_sync(
    user_id: str | None = None,
    language: str = "en",
    consent: bool = False,
    consent_ts: datetime | None = None,
    session_id: str | None = None,
    db: Database | None = None,
) -> dict:
    target_db = db or get_db()
    now = datetime.now(UTC)
    sid = session_id or str(uuid.uuid4())

    session_doc: dict[str, Any] = {
        "_id": sid,
        "user_id": user_id,
        "status": "COLLECTING",
        "case_file": {},
        "case_version": 0,
        "language": language,
        "messages": [],
        "consent": {
            "granted": consent,
            "timestamp": consent_ts or (now if consent else None),
        },
        "turn_count": 0,
        "asked_counts": {},
        "safety_flags": [],
        "last_service_query": None,
        "last_service_answer": None,
        "created_at": now,
        "updated_at": now,
    }
    target_db.sessions.insert_one(session_doc)
    _persist_mock_if_needed(target_db)
    return session_doc


def _get_session_sync(session_id: str, db: Database | None = None) -> dict | None:
    target_db = db or get_db()
    return target_db.sessions.find_one({"_id": session_id})


def _update_session_sync(session_id: str, updates: dict, db: Database | None = None) -> bool:
    target_db = db or get_db()
    updates = dict(updates)
    updates["updated_at"] = datetime.now(UTC)
    result = target_db.sessions.update_one({"_id": session_id}, {"$set": updates})
    _persist_mock_if_needed(target_db)
    return result.matched_count > 0


def _add_message_sync(
    session_id: str,
    role: str,
    text: str,
    affect: dict | None = None,
    db: Database | None = None,
) -> dict:
    target_db = db or get_db()
    now = datetime.now(UTC)
    msg_doc = {
        "_id": str(uuid.uuid4()),
        "session_id": session_id,
        "role": role,
        "text": text,
        "affect": affect or {},
        "ts": now,
    }
    # Embed message directly into the session document
    target_db.sessions.update_one(
        {"_id": session_id},
        {
            "$push": {"messages": msg_doc},
            "$set": {"updated_at": now},
        },
    )
    # Also keep messages collection synchronized for querying
    target_db.messages.insert_one(msg_doc)
    _persist_mock_if_needed(target_db)
    return msg_doc


def _get_recent_messages_sync(session_id: str, n: int, db: Database | None = None) -> list[dict]:
    target_db = db or get_db()
    session = target_db.sessions.find_one({"_id": session_id})
    if session and "messages" in session and session["messages"]:
        return list(session["messages"][-n:])
    cursor = target_db.messages.find({"session_id": session_id}).sort("ts", pymongo.DESCENDING).limit(n)
    docs = list(cursor)
    # Reverse to return chronological order (oldest -> newest)
    docs.reverse()
    return docs


def _get_all_messages_sync(session_id: str, db: Database | None = None) -> list[dict]:
    target_db = db or get_db()
    session = target_db.sessions.find_one({"_id": session_id})
    if session and "messages" in session and session["messages"]:
        return list(session["messages"])
    cursor = target_db.messages.find({"session_id": session_id}).sort("ts", pymongo.ASCENDING)
    return list(cursor)


def _delete_session_sync(session_id: str, db: Database | None = None) -> bool:
    target_db = db or get_db()
    target_db.sessions.delete_one({"_id": session_id})
    target_db.messages.delete_many({"session_id": session_id})
    target_db.case_versions.delete_many({"session_id": session_id})
    target_db.llm_logs.delete_many({"session_id": session_id})
    target_db.service_calls.delete_many({"session_id": session_id})
    _persist_mock_if_needed(target_db)
    return True


def _save_case_version_sync(
    session_id: str,
    version: int,
    patch: dict,
    full_case: dict,
    db: Database | None = None,
) -> dict:
    target_db = db or get_db()
    now = datetime.now(UTC)
    version_doc = {
        "_id": str(uuid.uuid4()),
        "session_id": session_id,
        "version": version,
        "patch": patch,
        "full_case": full_case,
        "ts": now,
    }
    target_db.case_versions.insert_one(version_doc)
    _persist_mock_if_needed(target_db)
    return version_doc


def _log_llm_call_sync(
    session_id: str | None,
    prompt_name: str,
    prompt_version: str,
    model: str,
    input_chars: int,
    output: str,
    latency_ms: int,
    tokens_in: int,
    tokens_out: int,
    success: bool,
    error: str | None = None,
    db: Database | None = None,
) -> dict:
    target_db = db or get_db()
    now = datetime.now(UTC)
    log_doc = {
        "_id": str(uuid.uuid4()),
        "session_id": session_id,
        "prompt_name": prompt_name,
        "prompt_version": prompt_version,
        "model": model,
        "input_chars": input_chars,
        "output": output,
        "latency_ms": latency_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "success": success,
        "error": error,
        "ts": now,
    }
    target_db.llm_logs.insert_one(log_doc)
    _persist_mock_if_needed(target_db)
    return log_doc


def _log_service_call_sync(
    session_id: str,
    request: dict,
    response: dict,
    latency_ms: int,
    status: str,
    db: Database | None = None,
) -> dict:
    target_db = db or get_db()
    now = datetime.now(UTC)
    call_doc = {
        "_id": str(uuid.uuid4()),
        "session_id": session_id,
        "request": request,
        "response": response,
        "latency_ms": latency_ms,
        "status": status,
        "ts": now,
    }
    target_db.service_calls.insert_one(call_doc)
    _persist_mock_if_needed(target_db)
    return call_doc


# ---------------------------------------------------------------------------
# Async wrappers via asyncio.to_thread
# ---------------------------------------------------------------------------

async def create_session(
    user_id: str | None = None,
    language: str = "en",
    consent: bool = False,
    consent_ts: datetime | None = None,
    session_id: str | None = None,
    db: Database | None = None,
) -> dict:
    """Async wrapper to create a new intake session in MongoDB."""
    return await asyncio.to_thread(
        _create_session_sync,
        user_id=user_id,
        language=language,
        consent=consent,
        consent_ts=consent_ts,
        session_id=session_id,
        db=db,
    )


async def get_session(session_id: str, db: Database | None = None) -> dict | None:
    """Async wrapper to retrieve a session by ID from MongoDB."""
    return await asyncio.to_thread(_get_session_sync, session_id=session_id, db=db)


async def update_session(session_id: str, updates: dict, db: Database | None = None) -> bool:
    """Async wrapper to update fields in an existing session."""
    return await asyncio.to_thread(_update_session_sync, session_id=session_id, updates=updates, db=db)


async def add_message(
    session_id: str,
    role: str,
    text: str,
    affect: dict | None = None,
    db: Database | None = None,
) -> dict:
    """Async wrapper to insert a message into the messages collection."""
    return await asyncio.to_thread(
        _add_message_sync,
        session_id=session_id,
        role=role,
        text=text,
        affect=affect,
        db=db,
    )


async def get_recent_messages(session_id: str, n: int, db: Database | None = None) -> list[dict]:
    """Async wrapper to get the n most recent messages for a session (in chronological order)."""
    return await asyncio.to_thread(_get_recent_messages_sync, session_id=session_id, n=n, db=db)


async def save_case_version(
    session_id: str,
    version: int,
    patch: dict,
    full_case: dict,
    db: Database | None = None,
) -> dict:
    """Async wrapper to record a validated case version and patch."""
    return await asyncio.to_thread(
        _save_case_version_sync,
        session_id=session_id,
        version=version,
        patch=patch,
        full_case=full_case,
        db=db,
    )


async def log_llm_call(
    session_id: str | None,
    prompt_name: str,
    prompt_version: str,
    model: str,
    input_chars: int,
    output: str,
    latency_ms: int,
    tokens_in: int,
    tokens_out: int,
    success: bool,
    error: str | None = None,
    db: Database | None = None,
) -> dict:
    """Async wrapper to log details of an LLM call to llm_logs."""
    return await asyncio.to_thread(
        _log_llm_call_sync,
        session_id=session_id,
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        model=model,
        input_chars=input_chars,
        output=output,
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        success=success,
        error=error,
        db=db,
    )


async def log_service_call(
    session_id: str,
    request: dict,
    response: dict,
    latency_ms: int,
    status: str,
    db: Database | None = None,
) -> dict:
    """Async wrapper to log details of an external legal service call."""
    return await asyncio.to_thread(
        _log_service_call_sync,
        session_id=session_id,
        request=request,
        response=response,
        latency_ms=latency_ms,
        status=status,
        db=db,
    )


async def get_all_messages(session_id: str, db: Database | None = None) -> list[dict]:
    """Async wrapper to get all messages for a session in chronological order."""
    return await asyncio.to_thread(_get_all_messages_sync, session_id=session_id, db=db)


async def delete_session(session_id: str, db: Database | None = None) -> bool:
    """Async wrapper to permanently delete a session and all related records across all collections."""
    return await asyncio.to_thread(_delete_session_sync, session_id=session_id, db=db)
