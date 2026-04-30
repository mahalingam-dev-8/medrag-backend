from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.dependencies import get_search_service
from app.services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    document_ids: list[str] | None = None


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    content: str
    similarity: float
    page_number: int | None
    section_title: str | None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total: int


@router.post("/", response_model=SearchResponse)
async def vector_search(
    request: SearchRequest,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    context = await service.search(
        query=request.query,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
        document_ids=request.document_ids,
    )
    return SearchResponse(
        query=request.query,
        results=[
            SearchResult(
                chunk_id=c["chunk_id"],
                document_id=c["document_id"],
                source=c["source"],
                content=c["content"],
                similarity=c["similarity"],
                page_number=c.get("page_number"),
                section_title=c.get("section_title"),
            )
            for c in context
        ],
        total=len(context),
    )
