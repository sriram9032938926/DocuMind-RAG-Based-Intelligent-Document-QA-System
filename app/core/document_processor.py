"""
app/core/document_processor.py
Document parsing, validation, and text chunking.
Uses pypdf (already in requirements.txt) — no fitz/PyMuPDF needed.
"""

import logging
from pathlib import Path
from typing import List, Tuple

from app.core.config import settings

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".doc", ".md"}
_CHARS_PER_WORD = 5   # used by chunker and tests


def validate_file(filename: str, file_size: int) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"File type '{ext}' is not supported. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_bytes:
        raise ValueError(
            f"File size {file_size / 1024 / 1024:.1f} MB exceeds "
            f"the {settings.MAX_FILE_SIZE_MB} MB limit."
        )


def _extract_pdf(file_path: str) -> Tuple[str, int]:
    """Extract text using pypdf (pure-Python, no system libs needed)."""
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    total_pages = len(reader.pages)
    pages_to_read = reader.pages[: settings.MAX_PAGES_PER_DOC]
    text = "\n".join(p.extract_text() or "" for p in pages_to_read)
    return text, total_pages


def _extract_docx(file_path: str) -> Tuple[str, int]:
    from docx import Document as DocxDoc
    doc = DocxDoc(file_path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    pages = max(1, len(text) // 500)
    return text, pages


def _extract_txt(file_path: str) -> Tuple[str, int]:
    text = Path(file_path).read_text(encoding="utf-8", errors="replace")
    pages = max(1, len(text) // 500)
    return text, pages


def extract_text_from_file(file_path: str, filename: str) -> Tuple[str, int]:
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        text, pages = _extract_pdf(file_path)
    elif ext in (".docx", ".doc"):
        text, pages = _extract_docx(file_path)
    elif ext in (".txt", ".md"):
        text, pages = _extract_txt(file_path)
    else:
        raise ValueError(f"File type '{ext}' is not supported.")

    if not text.strip():
        raise ValueError(
            "No text could be extracted. "
            "The file may be image-only, password-protected, or corrupted."
        )

    if pages > settings.MAX_PAGES_PER_DOC:
        raise ValueError(
            f"Document has {pages} pages; maximum allowed is {settings.MAX_PAGES_PER_DOC}."
        )

    logger.info(f"Extracted {len(text)} chars from {filename} ({pages} pages)")
    return text, pages


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Split text into overlapping chunks for embedding."""
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap    = overlap    or settings.CHUNK_OVERLAP

    if not text.strip():
        return []

    chunks = []
    start  = 0
    length = len(text)

    while start < length:
        end   = min(start + chunk_size, length)
        chunk = text[start:end].strip()

       
        if end < length:
            for sep in (".\n", ". ", "!\n", "! ", "?\n", "? "):
                pos = chunk.rfind(sep)
                if pos > chunk_size // 2:
                    chunk = chunk[: pos + 1].strip()
                    end   = start + pos + 1
                    break

        if chunk:
            chunks.append(chunk)

        if end >= length:
            break
        start = end - overlap

    logger.info(f"Chunked into {len(chunks)} chunks")
    return chunks
