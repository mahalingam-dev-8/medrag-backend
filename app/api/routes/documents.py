import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.api.dependencies import get_ingestion_service
from app.db.database import get_db
from app.services.ingestion_service import IngestionService
from app.utils.exceptions import DocumentNotFoundError
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


class IngestResponse(BaseModel):
    message: str
    filename: str
    status: str  # always "processing" — frontend polls GET /documents/ to detect completion


class DocumentResponse(BaseModel):
    id: str
    filename: str
    source: str
    doc_type: str
    total_chunks: int


async def _ingest_in_background(data: bytes, filename: str) -> None:
    """Runs after the 202 response is sent, with its own DB session."""
    try:
        async for session in get_db():
            service = IngestionService(session)
            result = await service.ingest_upload(data, filename)
            logger.info("background_ingest_complete", filename=filename, chunks=result.total_chunks)
    except Exception as exc:
        logger.error("background_ingest_failed", filename=filename, error=str(exc))


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    """
    Accepts the file and returns immediately.
    Embedding + storage runs in the background.
    Frontend polls GET /documents/ every few seconds until the document appears.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    data = await file.read()
    background_tasks.add_task(_ingest_in_background, data, file.filename)

    return IngestResponse(
        message="Document received and queued for processing",
        filename=file.filename,
        status="processing",
    )


@router.get("/", response_model=list[DocumentResponse])
async def list_documents(
    limit: int = 50,
    offset: int = 0,
    service: IngestionService = Depends(get_ingestion_service),
) -> list[DocumentResponse]:
    docs = await service.list_documents(limit=limit, offset=offset)
    return [
        DocumentResponse(
            id=str(d.id),
            filename=d.filename,
            source=d.source,
            doc_type=d.doc_type,
            total_chunks=d.total_chunks,
        )
        for d in docs
    ]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    service: IngestionService = Depends(get_ingestion_service),
) -> None:
    try:
        await service.delete_document(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
