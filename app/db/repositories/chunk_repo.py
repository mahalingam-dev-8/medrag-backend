import uuid
from dataclasses import dataclass

from pgvector.sqlalchemy import Vector
from sqlalchemy import cast, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.utils.exceptions import ChunkNotFoundError


@dataclass
class SimilarChunk:
    chunk: Chunk
    similarity: float
    document_filename: str
    document_source: str


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_batch(self, chunks: list[dict]) -> list[Chunk]:
        objects = [
            Chunk(
                document_id=c["document_id"],
                content=c["content"],
                chunk_index=c["chunk_index"],
                page_number=c.get("page_number"),
                section_title=c.get("section_title"),
                token_count=c.get("token_count", 0),
                embedding=c.get("embedding"),
                metadata_=c.get("metadata", {}),
            )
            for c in chunks
        ]
        self.session.add_all(objects)
        await self.session.flush()
        return objects

    async def get_by_id(self, chunk_id: uuid.UUID) -> Chunk:
        result = await self.session.execute(
            select(Chunk).where(Chunk.id == chunk_id)
        )
        chunk = result.scalar_one_or_none()
        if chunk is None:
            raise ChunkNotFoundError(str(chunk_id))
        return chunk

    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[SimilarChunk]:
        """Cosine similarity search using pgvector."""
        embedding_cast = cast(query_embedding, Vector(len(query_embedding)))

        stmt = (
            select(
                Chunk,
                (1 - Chunk.embedding.cosine_distance(embedding_cast)).label("similarity"),
                Document.filename.label("doc_filename"),
                Document.source.label("doc_source"),
            )
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.embedding.is_not(None))
            .where(
                (1 - Chunk.embedding.cosine_distance(embedding_cast)) >= similarity_threshold
            )
            .order_by(Chunk.embedding.cosine_distance(embedding_cast))
            .limit(top_k)
        )

        if document_ids:
            stmt = stmt.where(Chunk.document_id.in_(document_ids))

        result = await self.session.execute(stmt)
        rows = result.all()

        return [
            SimilarChunk(
                chunk=row.Chunk,
                similarity=float(row.similarity),
                document_filename=row.doc_filename,
                document_source=row.doc_source,
            )
            for row in rows
        ]

    async def fulltext_search(
        self,
        query: str,
        top_k: int = 20,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[SimilarChunk]:
        """Keyword search using PostgreSQL full-text search (tsvector/tsquery)."""
        stmt = (
            select(
                Chunk,
                func.ts_rank(
                    func.to_tsvector("english", Chunk.content),
                    func.plainto_tsquery("english", query),
                ).label("rank"),
                Document.filename.label("doc_filename"),
                Document.source.label("doc_source"),
            )
            .join(Document, Chunk.document_id == Document.id)
            .where(
                func.to_tsvector("english", Chunk.content).op("@@")(
                    func.plainto_tsquery("english", query)
                )
            )
            .order_by(text("rank DESC"))
            .limit(top_k)
        )

        if document_ids:
            stmt = stmt.where(Chunk.document_id.in_(document_ids))

        result = await self.session.execute(stmt)
        rows = result.all()

        return [
            SimilarChunk(
                chunk=row.Chunk,
                similarity=float(row.rank),
                document_filename=row.doc_filename,
                document_source=row.doc_source,
            )
            for row in rows
        ]

    async def get_by_document(self, document_id: uuid.UUID) -> list[Chunk]:
        result = await self.session.execute(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        )
        return list(result.scalars().all())

    async def delete_by_document(self, document_id: uuid.UUID) -> int:
        chunks = await self.get_by_document(document_id)
        for chunk in chunks:
            await self.session.delete(chunk)
        return len(chunks)
