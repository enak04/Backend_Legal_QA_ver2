"""
src/document_parser.py - Text extraction and parsing for uploaded user documents.

Supports:
- PDF (.pdf) via pypdf
- Plain text / Markdown / CSV / JSON (.txt, .md, .csv, .json)
- Word documents (.docx) if python-docx is available, or fallback
- Image attachments (.png, .jpg, .jpeg, .webp) via OpenAI gpt-4o-mini Vision with OCR
- Truncation to safe token limit (max 8,000 chars) with truncation notice
"""

import base64
import io
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_EXTRACTED_CHARS = 8000


def _extract_text_from_image(content: bytes, filename: str, content_type: str = "") -> str:
    """
    Extracts text from images (photos of documents, monitor screens, letters)
    using OpenAI Vision (gpt-4o-mini). Falls back to metadata tag if no API key is set.
    """
    from src.config import get_settings
    settings = get_settings()

    # If in test/fake mode or no valid key, fallback to metadata tag
    if (
        not settings.llm_api_key
        or settings.llm_api_key.startswith("mock_")
        or settings.provider == "fake"
    ):
        size_kb = round(len(content) / 1024, 1)
        return f"[Attached Image: {filename} ({size_kb} KB, document proof/photo)]"

    mime = content_type if content_type and content_type.startswith("image/") else "image/jpeg"
    ext = Path(filename).suffix.lower()
    if ext == ".png":
        mime = "image/png"
    elif ext == ".webp":
        mime = "image/webp"

    img_bytes = content
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(content))
        max_dim = 2048
        w, h = img.size
        if max(w, h) > max_dim or len(content) > 2 * 1024 * 1024:
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                new_size = (int(w * scale), int(h * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(buf, format="JPEG", quality=85)
            img_bytes = buf.getvalue()
            mime = "image/jpeg"
    except Exception as exc:
        logger.debug("Pillow image optimization skipped: %s", exc)

    import httpx

    b64_data = base64.b64encode(img_bytes).decode("utf-8")
    vision_model = settings.fast_model or "gpt-4o-mini"

    prompt_text = (
        "You are an expert legal document and text extraction assistant. "
        "Examine this image carefully (which may be a photo of a computer screen, an email, printed letter, contract, notice, or document). "
        "Extract ALL visible text, headers, senders, recipients, subject lines, dates, clauses, and key factual details with complete accuracy. "
        "Format the extracted text clearly and preserve all names, organizations, dates, amounts, and specific terminology."
    )

    payload = {
        "model": vision_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_data}", "detail": "high"}},
                ],
            }
        ],
        "max_tokens": 1500,
        "temperature": 0.0,
    }

    try:
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=35.0,
        )
        if resp.status_code == 200:
            result_json = resp.json()
            extracted_text = result_json["choices"][0]["message"]["content"].strip()
            if extracted_text:
                return extracted_text
        else:
            logger.warning("OpenAI vision extraction returned HTTP %d: %s", resp.status_code, resp.text)
    except Exception as exc:
        logger.warning("OpenAI vision extraction error for %s: %s", filename, exc)

    size_kb = round(len(content) / 1024, 1)
    return f"[Attached Image: {filename} ({size_kb} KB, document proof/photo)]"


def extract_text_from_file(
    content: bytes, filename: str, content_type: str = "", max_length: int = MAX_EXTRACTED_CHARS
) -> str:
    """
    Extracts plain text representation from binary file content.
    Returns cleaned text string suitable for injecting into the conversational intake pipeline.
    """
    ext = Path(filename).suffix.lower()
    text = ""

    # 1. PDF extraction
    if ext == ".pdf" or "pdf" in content_type.lower():
        try:
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(content))
            pages_text = []
            for i, page in enumerate(reader.pages):
                page_content = page.extract_text() or ""
                if page_content.strip():
                    pages_text.append(f"--- Page {i + 1} ---\n{page_content.strip()}")
            text = "\n\n".join(pages_text)
        except Exception as exc:
            logger.warning(f"Failed to parse PDF {filename}: {exc}")
            text = f"[Attached PDF: {filename} (could not extract text: {exc})]"

    # 2. Plain text / CSV / JSON / Markdown
    elif ext in (".txt", ".md", ".csv", ".json", ".log") or "text" in content_type.lower():
        try:
            text = content.decode("utf-8", errors="replace").strip()
        except Exception as exc:
            logger.warning(f"Failed to decode text file {filename}: {exc}")
            text = f"[Attached text document: {filename}]"

    # 3. Word Document (.docx)
    elif ext == ".docx":
        try:
            import docx  # type: ignore[import-untyped,import-not-found]

            doc = docx.Document(io.BytesIO(content))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n\n".join(paragraphs)
        except Exception:
            text = f"[Attached Word document: {filename} (preview unavailable)]"

    # 4. Images (.png, .jpg, .jpeg, .webp) via Vision OCR
    elif ext in (".png", ".jpg", ".jpeg", ".webp") or "image" in content_type.lower():
        text = _extract_text_from_image(content=content, filename=filename, content_type=content_type)

    # 5. Fallback
    else:
        text = f"[Attached file: {filename} ({content_type or 'unknown format'})]"

    text = text.strip()
    if not text:
        text = f"[Attached file: {filename} (empty content)]"

    # Truncate if exceeds safe character limit
    if len(text) > max_length:
        text = (
            text[:max_length]
            + f"\n\n[... Document truncated at {max_length} characters ...]"
        )

    return text
