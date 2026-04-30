import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.api.dependencies import get_ingestion_service
from app.services.ingestion_service import IngestionService
from app.utils.exceptions import DocumentNotFoundError

router = APIRouter(prefix="/documents", tags=["documents"])


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    total_chunks: int
    skipped: bool


class DocumentResponse(BaseModel):
    id: str
    filename: str
    source: str
    doc_type: str
    total_chunks: int


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(
    file: UploadFile,
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    data = await file.read()
    result = await service.ingest_upload(data, file.filename)
    return IngestResponse(
        document_id=result.document_id,
        filename=result.filename,
        total_chunks=result.total_chunks,
        skipped=result.skipped,
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
