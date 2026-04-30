import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.retriever import Retriever
from app.db.repositories.chunk_repo import ChunkRepository
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SearchService:
    def __init__(self, session: AsyncSession) -> None:
        self._chunk_repo = ChunkRepository(session)
        self._retriever = Retriever(self._chunk_repo)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        document_ids: list[str] | None = None,
    ) -> list[dict]:
        doc_uuids = [uuid.UUID(did) for did in document_ids] if document_ids else None
        results = await self._retriever.retrieve(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            document_ids=doc_uuids,
        )
        return self._retriever.format_context(results)
