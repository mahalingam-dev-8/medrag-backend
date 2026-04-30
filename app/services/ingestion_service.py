import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.chunk_repo import ChunkRepository
from app.db.repositories.document_repo import DocumentRepository
from app.ingestion.pipeline import IngestionPipeline, IngestionResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


class IngestionService:
    """Business logic for document ingestion.

    Wires together repos and the pipeline in __init__.
    No method ever receives or passes a session — all DB access
    goes through self._doc_repo / self._chunk_repo.

    NestJS equivalent:
        constructor(
            @InjectRepository(Document) private docRepo: DocumentRepository,
            @InjectRepository(Chunk)    private chunkRepo: ChunkRepository,
        ) {}
    """

    def __init__(self, session: AsyncSession) -> None:
        self._doc_repo = DocumentRepository(session)
        self._chunk_repo = ChunkRepository(session)
        self._pipeline = IngestionPipeline(self._doc_repo, self._chunk_repo)

    async def ingest_file(self, path: Path | str) -> IngestionResult:
        return await self._pipeline.run_from_path(path)

    async def ingest_upload(self, data: bytes, filename: str) -> IngestionResult:
        return await self._pipeline.run_from_bytes(data, filename)

    async def ingest_directory(self, directory: Path | str) -> list[IngestionResult]:
        results = []
        for file_path in sorted(Path(directory).rglob("*")):
            if file_path.suffix.lower() not in {".pdf", ".txt", ".md"}:
                continue
            try:
                results.append(await self.ingest_file(file_path))
            except Exception as exc:
                logger.error("file_ingestion_failed", path=str(file_path), error=str(exc))
        return results

    async def list_documents(self, limit: int = 50, offset: int = 0) -> list:
        return await self._doc_repo.list_all(limit=limit, offset=offset)

    async def delete_document(self, document_id: uuid.UUID) -> None:
        await self._doc_repo.delete(document_id)
