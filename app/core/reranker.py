import asyncio
from functools import lru_cache

from app.config import get_settings
from app.db.repositories.chunk_repo import SimilarChunk
from app.utils.exceptions import RerankingError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    """Re-ranks retrieved chunks using a cross-encoder model.

    Cross-encoder reads query + chunk together (unlike bi-encoder which reads
    them separately) — gives more accurate relevance scores at the cost of speed.
    That's why we run it only on the top-20 candidates, not the whole table.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info("loading_reranker_model", model=self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def _score(self, query: str, results: list[SimilarChunk]) -> list[tuple[SimilarChunk, float]]:
        pairs = [(query, r.chunk.content) for r in results]
        scores = self.model.predict(pairs)
        return list(zip(results, scores.tolist()))

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
            scored = await loop.run_in_executor(None, lambda: self._score(query, results))
            scored.sort(key=lambda x: x[1], reverse=True)
            reranked = [chunk for chunk, _ in scored]
            logger.info("reranking_complete", returned=min(top_k or len(reranked), len(reranked)))
            return reranked[:top_k] if top_k else reranked
        except Exception as exc:
            raise RerankingError(f"Reranking failed: {exc}") from exc


@lru_cache
def get_reranker() -> Reranker:
    settings = get_settings()
    return Reranker(model_name=settings.reranker_model)
