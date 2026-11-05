"""
tests/test_documents.py - Tests for document upload, extraction, and intake pipeline integration.
"""

import io
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app import app, set_app_llm
from src.document_parser import extract_text_from_file
from src.db import create_session, get_session
from src.llm import FakeLLM
from src.pipeline import handle_turn
from src.schemas import AffectReading, CasePatch, GapAnalysis, Intent, NewFact, UnderstandOutput


VALID_API_KEY = "test_app_key"
AUTH_HEADERS = {"X-API-Key": VALID_API_KEY}


def generate_dummy_pdf(text: str) -> bytes:
    """Helper to generate an in-memory PDF with text using pypdf."""
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    # Write empty or minimal PDF stream
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def setup_fake_llm():
    """Configures deterministic FakeLLM for document handling tests."""
    fake = FakeLLM(responses={
        "extractor": UnderstandOutput(
            intent=Intent.provides_info,
            language="English",
            patch=CasePatch(
                add_facts=[NewFact(text="Employment contract mentions 30-day notice period", source="inferred", confidence=0.9)]
            ),
            change_summary="Extracted contract notice clause",
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
        "talker": "I have reviewed your contract document. Did your employer give you 30 days notice?",
    })
    set_app_llm(fake)
    yield fake
    set_app_llm(None)


def test_extract_text_plain_text():
    """Verify plain text extraction."""
    raw = b"Clause 4.1: Either party may terminate with 30 days notice."
    extracted = extract_text_from_file(raw, "contract.txt", "text/plain")
    assert "Clause 4.1" in extracted
    assert "30 days notice" in extracted


def test_extract_text_truncation():
    """Verify truncation for documents exceeding max length."""
    long_text = ("Notice clause details. " * 500).encode("utf-8")
    extracted = extract_text_from_file(long_text, "large_doc.txt", "text/plain", max_length=500)
    assert len(extracted) <= 600
    assert "Document truncated at 500 characters" in extracted


def test_extract_text_pdf_fallback():
    """Verify PDF extraction handling."""
    dummy_pdf_bytes = generate_dummy_pdf("Sample PDF text")
    extracted = extract_text_from_file(dummy_pdf_bytes, "agreement.pdf", "application/pdf")
    assert isinstance(extracted, str)


@pytest.mark.asyncio
async def test_handle_turn_with_document_and_text(setup_fake_llm):
    """Verify handle_turn registers file and passes document text to extractor when text is also provided."""
    session = await create_session()
    sid = session["_id"]

    doc_data = {
        "filename": "termination_letter.txt",
        "content_type": "text/plain",
        "size_bytes": 128,
        "text": "Your employment is terminated effective Nov 1st due to restructuring.",
    }

    reply, decision = await handle_turn(
        session=session,
        user_message="Here is my letter",
        llm=setup_fake_llm,
        stream=False,
        document_data=doc_data,
    )

    assert reply is not None
    assert "contract document" in reply

    # Verify session registered file
    updated = await get_session(sid)
    assert "files" in updated
    assert len(updated["files"]) == 1
    assert updated["files"][0]["filename"] == "termination_letter.txt"

    # Verify extractor received document in prompt
    extractor_calls = [c for c in setup_fake_llm.call_history if c["prompt_name"] == "extractor"]
    assert len(extractor_calls) >= 1
    call_prompt = str(extractor_calls[0]["system"])
    assert "[Attached Document: termination_letter.txt]" in call_prompt
    assert "restructuring" in call_prompt


@pytest.mark.asyncio
async def test_handle_turn_with_document_alone(setup_fake_llm):
    """Verify handle_turn smoothly processes document upload when user provides NO text message."""
    session = await create_session()
    sid = session["_id"]

    doc_data = {
        "filename": "severance_agreement.txt",
        "content_type": "text/plain",
        "size_bytes": 256,
        "text": "Severance terms: 2 months compensation in lieu of notice.",
    }

    reply, decision = await handle_turn(
        session=session,
        user_message="",
        llm=setup_fake_llm,
        stream=False,
        document_data=doc_data,
    )

    assert reply is not None

    # Extractor should receive synthesized uploaded document tag
    extractor_calls = [c for c in setup_fake_llm.call_history if c["prompt_name"] == "extractor"]
    assert len(extractor_calls) >= 1
    call_prompt = str(extractor_calls[0]["system"])
    assert "[Uploaded Document: severance_agreement.txt]" in call_prompt
    assert "Severance terms" in call_prompt


def test_multipart_message_upload_endpoint(setup_fake_llm):
    """Verify POST /sessions/{id}/messages with multipart form data file upload."""
    client = TestClient(app)

    # 1. Create session
    res = client.post("/sessions", headers=AUTH_HEADERS, json={"consent": True})
    assert res.status_code == 201
    session_id = res.json()["session_id"]

    # 2. Upload file via multipart
    file_bytes = b"Contract: Clause 12 Non-compete within 5 miles."
    files = {"file": ("contract.txt", file_bytes, "text/plain")}
    data = {"text": "Please check this agreement"}

    response = client.post(
        f"/sessions/{session_id}/messages",
        headers=AUTH_HEADERS,
        data=data,
        files=files,
    )
    assert response.status_code == 200
    body = response.text
    assert "event: token" in body or "event: done" in body


def test_multipart_message_upload_alone(setup_fake_llm):
    """Verify POST /sessions/{id}/messages with file alone and no text succeeds."""
    client = TestClient(app)

    res = client.post("/sessions", headers=AUTH_HEADERS, json={"consent": True})
    assert res.status_code == 201
    session_id = res.json()["session_id"]

    file_bytes = b"Evidence of wrongful termination without prior performance review."
    files = {"file": ("evidence.txt", file_bytes, "text/plain")}

    response = client.post(
        f"/sessions/{session_id}/messages",
        headers=AUTH_HEADERS,
        files=files,
    )
    assert response.status_code == 200
    body = response.text
    assert "event: done" in body


def test_multipart_message_empty_fails(setup_fake_llm):
    """Verify POST /sessions/{id}/messages without file or text returns 422."""
    client = TestClient(app)

    res = client.post("/sessions", headers=AUTH_HEADERS, json={"consent": True})
    assert res.status_code == 201
    session_id = res.json()["session_id"]

    response = client.post(
        f"/sessions/{session_id}/messages",
        headers=AUTH_HEADERS,
        data={"text": "   "},
    )
    assert response.status_code == 422


def test_extract_text_image_fallback_in_mock():
    """Verify image parsing returns metadata tag when in mock/test mode without live key."""
    # 1x1 transparent PNG bytes
    dummy_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    extracted = extract_text_from_file(dummy_png, "notice.png", "image/png")
    assert "notice.png" in extracted


def test_extract_text_image_vision_mocked(monkeypatch):
    """Verify image parsing calls OpenAI Vision API when configured and extracts text."""
    from unittest.mock import MagicMock
    import httpx

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Sender: Google People Operations\nTermination effective November 5, 2026."
                }
            }
        ]
    }

    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: mock_resp)

    # Temporarily set valid-looking key in settings
    from src.config import Settings
    import src.config
    current_settings = src.config.get_settings()
    test_settings = Settings(
        provider="openai",
        llm_api_key="sk-real-looking-key-123456",
        fast_model="gpt-4o-mini",
        smart_model="gpt-4o",
        mongodb_uri=current_settings.mongodb_uri,
        db_name=current_settings.db_name,
        use_mock_service=current_settings.use_mock_service,
        service_url=current_settings.service_url,
        service_api_key=current_settings.service_api_key,
        max_questions_per_turn=current_settings.max_questions_per_turn,
        history_window=current_settings.history_window,
        max_collecting_turns=current_settings.max_collecting_turns,
        session_ttl_days=current_settings.session_ttl_days,
        debug=current_settings.debug,
    )
    monkeypatch.setattr(src.config, "get_settings", lambda: test_settings)

    dummy_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    extracted = extract_text_from_file(dummy_png, "termination_email.png", "image/png")
    assert "Google People Operations" in extracted
    assert "November 5, 2026" in extracted

