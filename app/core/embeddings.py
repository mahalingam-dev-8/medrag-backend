import asyncio
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.utils.exceptions import EmbeddingError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str, dimension: int) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("loading_embedding_model", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Synchronous encode — always called via run_in_executor from async code."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )

    async def embed_text(self, text: str) -> list[float]:
        """Async wrapper — offloads CPU-heavy encoding to a thread pool.

        NestJS analogy: like wrapping a blocking crypto.pbkdf2 call
        inside util.promisify so it doesn't block the event loop.
        """
        try:
            loop = asyncio.get_running_loop()
            embeddings: np.ndarray = await loop.run_in_executor(
                None, lambda: self._encode([text])
            )
            return embeddings[0].tolist()
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed text: {exc}") from exc

    async def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Async wrapper — offloads CPU-heavy batch encoding to a thread pool."""
        if not texts:
            return []
        try:
            logger.info("embedding_batch", count=len(texts), batch_size=batch_size)
            loop = asyncio.get_running_loop()
            embeddings: np.ndarray = await loop.run_in_executor(
                None, lambda: self._encode(texts, batch_size)
            )
            return embeddings.tolist()
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed batch of {len(texts)} texts: {exc}") from exc


@lru_cache
def get_embedding_service() -> EmbeddingService:
    settings = get_settings()
    return EmbeddingService(
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
    )
