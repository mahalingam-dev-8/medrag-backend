import asyncio
import uuid

from app.config import get_settings
from app.core.embeddings import get_embedding_service
from app.db.repositories.chunk_repo import ChunkRepository, SimilarChunk
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class Retriever:
    """Converts a text query into a vector and searches the chunk repository.

    Receives a ChunkRepository in __init__ — no session, no direct DB access.

    NestJS equivalent:
        constructor(
            @InjectRepository(Chunk) private chunkRepo: ChunkRepository,
        ) {}
    """

    def __init__(self, chunk_repo: ChunkRepository) -> None:
        self._chunk_repo = chunk_repo
        self._embedding_service = get_embedding_service()

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[SimilarChunk]:
        k = top_k or settings.retrieval_candidate_k
        threshold = similarity_threshold or settings.similarity_threshold

        logger.info("retrieving_chunks", query_preview=query[:80], top_k=k, hybrid=settings.hybrid_search_enabled)

        if settings.hybrid_search_enabled:
            return await self._hybrid_retrieve(query, k, threshold, document_ids)

        query_embedding = await self._embedding_service.embed_text(query)
        results = await self._chunk_repo.similarity_search(
            query_embedding=query_embedding,
            top_k=k,
            similarity_threshold=threshold,
            document_ids=document_ids,
        )
        logger.info("retrieval_complete", results_count=len(results))
        return results

    async def _hybrid_retrieve(
        self,
        query: str,
        top_k: int,
        similarity_threshold: float,
        document_ids: list[uuid.UUID] | None,
    ) -> list[SimilarChunk]:
        """Runs vector search and full-text search in parallel, merges with RRF."""
        query_embedding = await self._embedding_service.embed_text(query)

        vector_results, bm25_results = await asyncio.gather(
            self._chunk_repo.similarity_search(
                query_embedding=query_embedding,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                document_ids=document_ids,
            ),
            self._chunk_repo.fulltext_search(
                query=query,
                top_k=top_k,
                document_ids=document_ids,
            ),
        )

        merged = self._reciprocal_rank_fusion(vector_results, bm25_results, top_k=top_k)
        logger.info(
            "hybrid_retrieval_complete",
            vector=len(vector_results),
            bm25=len(bm25_results),
            merged=len(merged),
        )
        return merged

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[SimilarChunk],
        bm25_results: list[SimilarChunk],
        top_k: int,
        rrf_k: int = 60,
    ) -> list[SimilarChunk]:
        """Merges two ranked lists using Reciprocal Rank Fusion.

        Score = 1/(rank + rrf_k) summed across both lists.
        Chunks appearing high in both lists get the highest combined score.
        rrf_k=60 is the standard default from the original RRF paper.
        """
        scores: dict[str, float] = {}
        chunks: dict[str, SimilarChunk] = {}

        for rank, result in enumerate(vector_results):
            chunk_id = str(result.chunk.id)
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (rank + rrf_k)
            chunks[chunk_id] = result

        for rank, result in enumerate(bm25_results):
            chunk_id = str(result.chunk.id)
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (rank + rrf_k)
            if chunk_id not in chunks:
                chunks[chunk_id] = result

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [chunks[chunk_id] for chunk_id in sorted_ids[:top_k]]

    def format_context(self, results: list[SimilarChunk]) -> list[dict]:
        return [
            {
                "content": r.chunk.content,
                "source": r.document_filename,
                "page_number": r.chunk.page_number,
                "section_title": r.chunk.section_title,
                "similarity": r.similarity,
                "chunk_id": str(r.chunk.id),
                "document_id": str(r.chunk.document_id),
            }
            for r in results
        ]
