import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Document
from app.utils.exceptions import DocumentNotFoundError


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        filename: str,
        source: str,
        doc_type: str = "pdf",
        metadata: dict | None = None,
    ) -> Document:
        doc = Document(
            filename=filename,
            source=source,
            doc_type=doc_type,
            metadata_=metadata or {},
        )
        self.session.add(doc)
        await self.session.flush()
        return doc

    async def get_by_id(self, document_id: uuid.UUID) -> Document:
        result = await self.session.execute(
            select(Document)
            .where(Document.id == document_id)
            .options(selectinload(Document.chunks))
        )
        doc = result.scalar_one_or_none()
        if doc is None:
            raise DocumentNotFoundError(str(document_id))
        return doc

    async def get_by_source(self, source: str) -> Document | None:
        result = await self.session.execute(
            select(Document).where(Document.source == source)
        )
        return result.scalar_one_or_none()

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[Document]:
        result = await self.session.execute(
            select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
        )
        return list(result.scalars().all())

    async def update_chunk_count(self, document_id: uuid.UUID, total_chunks: int) -> None:
        await self.session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(total_chunks=total_chunks)
        )

    async def delete(self, document_id: uuid.UUID) -> None:
        doc = await self.get_by_id(document_id)
        self.session.delete(doc)  # delete() is sync — no await needed
