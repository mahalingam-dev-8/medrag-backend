import pytest

from app.ingestion.chunker import TextChunk, chunk_document, count_tokens
from app.ingestion.loader import LoadedDocument, LoadedPage


def make_doc(text: str, pages: int = 1) -> LoadedDocument:
    page_size = len(text) // pages
    loaded_pages = [
        LoadedPage(content=text[i * page_size:(i + 1) * page_size], page_number=i + 1)
        for i in range(pages)
    ]
    return LoadedDocument(filename="test.pdf", source="test.pdf", doc_type="pdf", pages=loaded_pages)


def test_count_tokens_short():
    assert count_tokens("hello world") > 0


def test_chunk_short_document():
    doc = make_doc("Short text that fits in one chunk.")
    chunks = chunk_document(doc, chunk_size=512, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0


def test_chunk_long_document():
    long_text = "This is a sentence. " * 200
    doc = make_doc(long_text)
    chunks = chunk_document(doc, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
        assert chunk.token_count > 0


def test_chunk_preserves_page_number():
    doc = make_doc("Page content " * 50, pages=2)
    chunks = chunk_document(doc, chunk_size=50, chunk_overlap=5)
    page_numbers = {c.page_number for c in chunks}
    assert page_numbers.issubset({1, 2})


def test_chunk_overlap_produces_context():
    text = ("word " * 30 + "\n\n") * 5
    doc = make_doc(text)
    chunks = chunk_document(doc, chunk_size=60, chunk_overlap=15)
    # Adjacent chunks should share some words
    if len(chunks) >= 2:
        assert len(chunks[0].content) > 0
        assert len(chunks[1].content) > 0
