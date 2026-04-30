import pytest

from app.core.prompts import build_rag_prompt


def test_build_rag_prompt_includes_sources():
    chunks = [
        {"content": "Aspirin reduces fever.", "source": "drug_guide.pdf", "page_number": 12},
        {"content": "NSAIDs inhibit COX-2.", "source": "pharmacology.pdf", "page_number": 5},
    ]
    prompt = build_rag_prompt(chunks, "What does aspirin do?")
    assert "Aspirin reduces fever." in prompt
    assert "drug_guide.pdf" in prompt
    assert "page 12" in prompt
    assert "What does aspirin do?" in prompt


def test_build_rag_prompt_handles_missing_page():
    chunks = [{"content": "Some text.", "source": "doc.pdf"}]
    prompt = build_rag_prompt(chunks, "query")
    assert "page" not in prompt
    assert "doc.pdf" in prompt
