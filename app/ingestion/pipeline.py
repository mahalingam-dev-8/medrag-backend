from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.core.embeddings import EmbeddingService, get_embedding_service
from app.db.repositories.chunk_repo import ChunkRepository
from app.db.repositories.document_repo import DocumentRepository
from app.ingestion.chunker import chunk_document
from app.ingestion.loader import LoadedDocument, load_bytes, load_file
from app.ingestion.metadata import detect_section_title, extract_document_metadata
from app.utils.exceptions import IngestionError
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class IngestionResult:
    document_id: str
    filename: str
    total_chunks: int
    skipped: bool = False


class IngestionPipeline:
    """Orchestrates: load → chunk → embed → store.

    Receives repos in __init__ — never touches the session directly.
    This mirrors NestJS where a service receives injected repos and
    never calls the database driver itself.
    """

    def __init__(
        self,
        doc_repo: DocumentRepository,
        chunk_repo: ChunkRepository,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self._doc_repo = doc_repo
        self._chunk_repo = chunk_repo
        self._embedding_service = embedding_service or get_embedding_service()

    async def run_from_path(
        self,
        path: Path | str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> IngestionResult:
        doc = load_file(Path(path))
        return await self._run(doc, chunk_size, chunk_overlap)

    async def run_from_bytes(
        self,
        data: bytes,
        filename: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> IngestionResult:
        doc = load_bytes(data, filename)
        return await self._run(doc, chunk_size, chunk_overlap)

    async def _run(
        self,
        doc: LoadedDocument,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> IngestionResult:
        # Idempotency: skip if already ingested
        existing = await self._doc_repo.get_by_source(doc.source)
        if existing is not None:
            logger.info("document_already_ingested", source=doc.source)
            return IngestionResult(
                document_id=str(existing.id),
                filename=existing.filename,
                total_chunks=existing.total_chunks,
                skipped=True,
            )

        # 1. Save document record
        db_doc = await self._doc_repo.create(
            filename=doc.filename,
            source=doc.source,
            doc_type=doc.doc_type,
            metadata=extract_document_metadata(doc),
        )
        logger.info("document_created", document_id=str(db_doc.id), filename=doc.filename)

        # 2. Chunk
        cs = chunk_size or settings.chunk_size
        co = chunk_overlap or settings.chunk_overlap
        chunks = chunk_document(doc, chunk_size=cs, chunk_overlap=co)

        if not chunks:
            raise IngestionError(f"No text extracted from {doc.filename}")

        # 3. Embed
        embeddings = await self._embedding_service.embed_batch([c.content for c in chunks])

        # 4. Persist chunks via repo
        chunk_dicts = [
            {
                "document_id": db_doc.id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "section_title": detect_section_title(chunk.content),
                "token_count": chunk.token_count,
                "embedding": embeddings[i],
                "metadata": chunk.metadata,
            }
            for i, chunk in enumerate(chunks)
        ]
        await self._chunk_repo.create_batch(chunk_dicts)

        # 5. Update document chunk count via repo
        await self._doc_repo.update_chunk_count(db_doc.id, len(chunks))

        logger.info(
            "ingestion_complete",
            document_id=str(db_doc.id),
            filename=doc.filename,
            chunks=len(chunks),
        )
        return IngestionResult(
            document_id=str(db_doc.id),
            filename=doc.filename,
            total_chunks=len(chunks),
        )
