"""
src/redact.py - PII and sensitive data redaction helpers before logging or external calls.

Strips anything that looks like:
- A phone number
- An email address
- A long digit sequence (>= 7 consecutive digits like SSN, account/ID numbers)
"""

import re
from typing import Any

from src.schemas import ServiceQuery, ServiceQueryFact

# Regex patterns for PII detection
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}\b|\b\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b")
LONG_DIGIT_REGEX = re.compile(r"\b\d{7,}\b")



def redact_text(text: str) -> str:
    """
    Replaces emails, phone numbers, and long digit sequences in text with [REDACTED].
    """
    if not text:
        return text

    # Redact emails first
    cleaned = EMAIL_REGEX.sub("[REDACTED_EMAIL]", text)
    # Redact phone numbers
    cleaned = PHONE_REGEX.sub("[REDACTED_PHONE]", cleaned)
    # Redact long digit sequences
    cleaned = LONG_DIGIT_REGEX.sub("[REDACTED_ID]", cleaned)

    return cleaned


def redact_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively redacts string values within dictionaries and lists."""
    result: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, str):
            result[k] = redact_text(v)
        elif isinstance(v, dict):
            result[k] = redact_dict(v)
        elif isinstance(v, list):
            result[k] = [
                redact_text(item) if isinstance(item, str)
                else redact_dict(item) if isinstance(item, dict)
                else item
                for item in v
            ]
        else:
            result[k] = v
    return result


def redact_service_query(query: ServiceQuery) -> ServiceQuery:
    """
    Returns a new ServiceQuery with PII stripped from query_text, facts,
    timeline, parties, documents, deadlines, and assumptions.
    """
    clean_query_text = redact_text(query.query_text)
    clean_facts = [
        ServiceQueryFact(text=redact_text(f.text), certainty=f.certainty)
        for f in query.facts
    ]
    clean_timeline = [redact_dict(t) for t in query.timeline]
    clean_parties = [redact_dict(p) for p in query.parties]
    clean_documents = [redact_dict(d) for d in query.documents]
    clean_deadlines = [redact_dict(dl) for dl in query.deadlines]
    clean_assumptions = [redact_text(a) for a in query.assumptions]
    clean_questions = [redact_text(q) for q in query.questions_to_answer]

    return query.model_copy(update={
        "query_text": clean_query_text,
        "facts": clean_facts,
        "timeline": clean_timeline,
        "parties": clean_parties,
        "documents": clean_documents,
        "deadlines": clean_deadlines,
        "assumptions": clean_assumptions,
        "questions_to_answer": clean_questions,
    })
