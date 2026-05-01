import asyncio
import gc
from functools import lru_cache

from fastembed import TextEmbedding

from app.config import get_settings
from app.utils.exceptions import EmbeddingError
from app.utils.logger import get_logger

logger = get_logger(__name__)

_BATCH_SIZE = 8  # small batches to keep peak memory low on free tier


class EmbeddingService:
    def __init__(self, model_name: str, dimension: int) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self._model: TextEmbedding | None = None

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            logger.info("loading_embedding_model", model=self.model_name)
            self._model = TextEmbedding(self.model_name)
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            results.extend(emb.tolist() for emb in self.model.embed(batch))
            gc.collect()
        return results

    async def embed_text(self, text: str) -> list[float]:
        try:
            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(None, lambda: self._encode([text]))
            return embeddings[0]
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed text: {exc}") from exc

    async def embed_batch(self, texts: list[str], batch_size: int = _BATCH_SIZE) -> list[list[float]]:
        if not texts:
            return []
        try:
            logger.info("embedding_batch", count=len(texts))
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self._encode(texts))
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed batch of {len(texts)} texts: {exc}") from exc


@lru_cache
def get_embedding_service() -> EmbeddingService:
    settings = get_settings()
    return EmbeddingService(
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
    )
