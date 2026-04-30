import pytest

from app.core.embeddings import EmbeddingService


def test_embedding_service_produces_correct_dimension():
    service = EmbeddingService(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
    )
    embedding = service.embed_text("test medical query")
    assert len(embedding) == 384


def test_embedding_batch_returns_same_count():
    service = EmbeddingService(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
    )
    texts = ["first chunk", "second chunk", "third chunk"]
    embeddings = service.embed_batch(texts)
    assert len(embeddings) == 3
    for emb in embeddings:
        assert len(emb) == 384


def test_embed_empty_batch():
    service = EmbeddingService(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
    )
    result = service.embed_batch([])
    assert result == []
