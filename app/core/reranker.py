import asyncio
from functools import lru_cache

from app.config import get_settings
from app.db.repositories.chunk_repo import SimilarChunk
from app.utils.exceptions import RerankingError
from app.utils.logger import get_logger

logger = get_logger(__name__)

# sentence-transformers model names map to the ONNX builds fastembed ships, so existing
# RERANKER_MODEL values keep working without installing PyTorch.
_ONNX_MODELS = {
    "cross-encoder/ms-marco-TinyBERT-L-2-v2": "Xenova/ms-marco-TinyBERT-L-2-v2",
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "Xenova/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2": "Xenova/ms-marco-MiniLM-L-12-v2",
}

# Working memory grows with batch size x text length. Batches of 8 full chunks pushed the
# process past 512 MB; 2 chunks of at most 1500 chars stays ~30 MB above idle. The start of
# a chunk is enough to judge relevance (the model cuts at 512 tokens anyway).
_BATCH_SIZE = 2
_MAX_CHARS = 1500


def load_cross_encoder(model_name: str):
    """Loads an ONNX cross-encoder, registering Hugging Face repos fastembed doesn't list.

    Runs on one thread: the free instance has 0.1 CPU, and ONNX Runtime's default of one
    thread per host core only makes those threads fight over that slice.
    """
    from fastembed.common.model_description import ModelSource
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    if model_name not in {m["model"] for m in TextCrossEncoder.list_supported_models()}:
        TextCrossEncoder.add_custom_model(model=model_name, sources=ModelSource(hf=model_name))
    return TextCrossEncoder(model_name=model_name, threads=1)


class Reranker:
    """Re-ranks retrieved chunks using a cross-encoder model.

    Cross-encoder reads query + chunk together (unlike bi-encoder which reads
    them separately) — gives more accurate relevance scores at the cost of speed.
    That's why we run it only on the top-20 candidates, not the whole table.

    Runs as ONNX through fastembed (no PyTorch), so it fits the same 512 MB instance as
    the embeddings. Chunks scoring below ``min_score`` are dropped rather than passed to the
    LLM: ms-marco scores are logits, and clearly unrelated chunks land far below zero.
    """

    def __init__(self, model_name: str, min_score: float, batch_size: int = _BATCH_SIZE) -> None:
        self.model_name = _ONNX_MODELS.get(model_name, model_name)
        self.min_score = min_score
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        # Loaded once. A load failure raises instead of silently skipping re-ranking,
        # which is how this step previously went missing unnoticed.
        if self._model is None:
            logger.info("loading_reranker_model", model=self.model_name)
            self._model = load_cross_encoder(self.model_name)
        return self._model

    async def rerank(
        self,
        query: str,
        results: list[SimilarChunk],
        top_k: int | None = None,
    ) -> list[SimilarChunk]:
        if not results:
            return results
        try:
            logger.info("reranking_chunks", candidates=len(results), top_k=top_k)
            loop = asyncio.get_running_loop()
            documents = [r.chunk.content[:_MAX_CHARS] for r in results]
            scores = await loop.run_in_executor(
                None, lambda: list(self.model.rerank(query, documents, batch_size=self.batch_size))
            )
            scored = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
            reranked = [chunk for chunk, score in scored if score >= self.min_score]
            logger.info(
                "reranking_complete",
                kept=len(reranked),
                dropped=len(scored) - len(reranked),
                top_score=round(float(scored[0][1]), 2),
            )
            return reranked[:top_k] if top_k else reranked
        except Exception as exc:
            raise RerankingError(f"Reranking failed: {exc}") from exc


@lru_cache
def get_reranker() -> Reranker:
    settings = get_settings()
    return Reranker(model_name=settings.reranker_model, min_score=settings.reranker_min_score)
