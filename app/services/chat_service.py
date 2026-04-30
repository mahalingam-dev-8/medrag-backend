import uuid
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.chain import RAGChain, RAGResponse
from app.core.retriever import Retriever
from app.db.repositories.chunk_repo import ChunkRepository
from app.db.repositories.session_repo import SessionRepository


class ChatService:
    """Orchestrates the chat flow.

    Wires Retriever → RAGChain in __init__.
    No method ever receives or passes a session.

    NestJS equivalent:
        constructor(
            @InjectRepository(Chunk) private chunkRepo: ChunkRepository,
        ) {}
    """

    def __init__(self, session: AsyncSession) -> None:
        self._chunk_repo = ChunkRepository(session)
        self._chain = RAGChain(Retriever(self._chunk_repo))
        self._session_repo = SessionRepository(session)

    async def stream_answer(
        self,
        question: str,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        doc_uuids = [uuid.UUID(did) for did in document_ids] if document_ids else None
        async for token in self._chain.run_stream(
            query=question,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            document_ids=doc_uuids,
            model=model,
        ):
            yield token

    async def answer(
        self,
        question: str,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        model: str = "llama-3.3-70b-versatile",
    ) -> RAGResponse:
        doc_uuids = [uuid.UUID(did) for did in document_ids] if document_ids else None
        return await self._chain.run(
            query=question,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            document_ids=doc_uuids,
            model=model,
        )

    async def session_chat(
        self,
        session_id: uuid.UUID,
        question: str,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        model: str | None = None,
    ) -> RAGResponse:
        # Load prior conversation history from DB
        messages = await self._session_repo.get_messages(session_id)
        history = [{"role": m.role, "content": m.content} for m in messages]

        # Auto-title session from first question (first 60 chars)
        if not messages:
            await self._session_repo.set_title(session_id, question[:60])

        # Save user message before generating
        await self._session_repo.add_message(session_id, "user", question)

        # Run RAG with full conversation history
        doc_uuids = [uuid.UUID(did) for did in document_ids] if document_ids else None
        result = await self._chain.run_with_history(
            query=question,
            history=history,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            document_ids=doc_uuids,
            model=model,
        )

        # Save assistant response
        await self._session_repo.add_message(
            session_id, "assistant", result.answer, sources=result.sources
        )

        return result
