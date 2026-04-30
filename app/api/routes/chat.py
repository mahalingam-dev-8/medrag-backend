from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.dependencies import get_chat_service
from app.services.chat_service import ChatService
from app.utils.exceptions import GenerationError, RerankingError, RetrievalError

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    document_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    model: str | None = Field(default=None)  # None = use OPENAI_MODEL from config


class SourceReference(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    page_number: int | None
    section_title: str | None
    similarity: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    question: str


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    try:
        result = await service.answer(
            question=request.question,
            document_ids=request.document_ids,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            model=request.model,
        )
    except (RetrievalError, GenerationError, RerankingError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        answer=result.answer,
        question=result.query,
        sources=[
            SourceReference(
                chunk_id=s["chunk_id"],
                document_id=s["document_id"],
                source=s["source"],
                page_number=s.get("page_number"),
                section_title=s.get("section_title"),
                similarity=s["similarity"],
            )
            for s in result.sources
        ],
    )


@router.post("/stream/")
async def chat_stream(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    async def generate():
        try:
            async for token in service.stream_answer(
                question=request.question,
                document_ids=request.document_ids,
                top_k=request.top_k,
                similarity_threshold=request.similarity_threshold,
                model=request.model,
            ):
                yield f"data: {token}\n\n"
        except (RetrievalError, GenerationError, RerankingError) as exc:
            yield f"data: [ERROR] {str(exc)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
