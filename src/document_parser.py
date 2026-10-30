"""
src/document_parser.py - Text extraction and parsing for uploaded user documents.

Supports:
- PDF (.pdf) via pypdf
- Plain text / Markdown / CSV / JSON (.txt, .md, .csv, .json)
- Word documents (.docx) if python-docx is available, or fallback
- Image attachments (.png, .jpg, .jpeg) with metadata and visual summary
- Truncation to safe token limit (max 8,000 chars) with truncation notice
"""

import io
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_EXTRACTED_CHARS = 8000


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

    # 4. Images (.png, .jpg, .jpeg, .webp)
    elif ext in (".png", ".jpg", ".jpeg", ".webp") or "image" in content_type.lower():
        size_kb = round(len(content) / 1024, 1)
        text = f"[Attached Image: {filename} ({size_kb} KB, document proof/photo)]"

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
