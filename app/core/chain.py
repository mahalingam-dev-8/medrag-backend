import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

from app.config import get_settings
from app.core.generator import Generator
from app.core.reranker import get_reranker
from app.core.retriever import Retriever
from app.utils.logger import get_logger

logger = get_logger(__name__)

_generator = Generator()


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict] = field(default_factory=list)
    query: str = ""


class RAGChain:
    """Full RAG pipeline: retrieve → rerank → generate.

    Retrieves retrieval_candidate_k candidates (default 20), reranks with a
    cross-encoder, then sends the best top_k (default 5) to the LLM.
    """

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    async def run(
        self,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        document_ids: list[uuid.UUID] | None = None,
        model: str | None = None,
    ) -> RAGResponse:
        settings = get_settings()
        final_top_k = top_k or settings.top_k

        # 1. Retrieve more candidates than needed — reranker picks the best ones
        results = await self._retriever.retrieve(
            query=query,
            top_k=settings.retrieval_candidate_k,
            similarity_threshold=similarity_threshold,
            document_ids=document_ids,
        )

        # 2. Rerank candidates with cross-encoder, return best final_top_k
        if results:
            reranked = await get_reranker().rerank(query=query, results=results, top_k=final_top_k)
            context_chunks = self._retriever.format_context(reranked)
        else:
            logger.warning("no_chunks_retrieved", query=query)
            context_chunks = []

        # 3. Generate — GPT responds even with empty context
        answer = await _generator.generate(
            question=query,
            context_chunks=context_chunks,
            model=model,
        )

        return RAGResponse(answer=answer, sources=context_chunks, query=query)

    async def run_with_history(
        self,
        query: str,
        history: list[dict],
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        document_ids: list[uuid.UUID] | None = None,
        model: str | None = None,
    ) -> RAGResponse:
        settings = get_settings()
        final_top_k = top_k or settings.top_k

        results = await self._retriever.retrieve(
            query=query,
            top_k=settings.retrieval_candidate_k,
            similarity_threshold=similarity_threshold,
            document_ids=document_ids,
        )

        if results:
            reranked = await get_reranker().rerank(query=query, results=results, top_k=final_top_k)
            context_chunks = self._retriever.format_context(reranked)
        else:
            logger.warning("no_chunks_retrieved", query=query)
            context_chunks = []

        answer = await _generator.generate(
            question=query,
            context_chunks=context_chunks,
            model=model,
            history=history,
        )

        return RAGResponse(answer=answer, sources=context_chunks, query=query)

    async def run_stream(
        self,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        document_ids: list[uuid.UUID] | None = None,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        settings = get_settings()
        final_top_k = top_k or settings.top_k

        results = await self._retriever.retrieve(
            query=query,
            top_k=settings.retrieval_candidate_k,
            similarity_threshold=similarity_threshold,
            document_ids=document_ids,
        )

        if results:
            reranked = await get_reranker().rerank(query=query, results=results, top_k=final_top_k)
            context_chunks = self._retriever.format_context(reranked)
        else:
            logger.warning("no_chunks_retrieved", query=query)
            context_chunks = []

        async for token in _generator.stream(
            question=query,
            context_chunks=context_chunks,
            model=model,
        ):
            yield token
