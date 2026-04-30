"""Metadata extraction helpers for ingested documents."""

import re

from app.ingestion.loader import LoadedDocument


def extract_document_metadata(doc: LoadedDocument) -> dict:
    """Enrich document-level metadata."""
    return {
        **doc.metadata,
        "filename": doc.filename,
        "doc_type": doc.doc_type,
        "page_count": len(doc.pages),
        "char_count": sum(len(p.content) for p in doc.pages),
    }


def detect_section_title(text: str) -> str | None:
    """Heuristically detect a section heading from the start of a chunk."""
    lines = text.strip().splitlines()
    if not lines:
        return None

    first_line = lines[0].strip()

    # All-caps line under 80 chars → likely a heading
    if first_line.isupper() and len(first_line) < 80:
        return first_line

    # Numbered section like "1.2 Introduction" or "Section 3:"
    if re.match(r"^(\d+\.)+\s+\w|^Section\s+\d+", first_line, re.IGNORECASE):
        return first_line

    # Short line ending with ":" or no sentence-ending punctuation
    if len(first_line) < 60 and not first_line.endswith((".", "?", "!")):
        return first_line

    return None
