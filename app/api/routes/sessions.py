import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import get_chat_service
from app.db.repositories.session_repo import SessionRepository
from app.services.chat_service import ChatService
from app.utils.exceptions import GenerationError, RerankingError, RetrievalError, SessionNotFoundError

router = APIRouter(prefix="/sessions", tags=["sessions"])


class CreateSessionRequest(BaseModel):
    title: str | None = None


class SessionResponse(BaseModel):
    id: uuid.UUID
    title: str | None
    is_active: bool


class MessageResponse(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    sources: list


class SessionDetailResponse(BaseModel):
    id: uuid.UUID
    title: str | None
    is_active: bool
    messages: list[MessageResponse]


class SessionChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    document_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    model: str | None = None


class SessionChatResponse(BaseModel):
    answer: str
    sources: list
    session_id: uuid.UUID


@router.post("/", response_model=SessionResponse)
async def create_session(
    request: CreateSessionRequest,
    service: ChatService = Depends(get_chat_service),
) -> SessionResponse:
    session = await service._session_repo.create_session(title=request.title)
    return SessionResponse(id=session.id, title=session.title, is_active=session.is_active)


@router.get("/{session_id}/", response_model=SessionDetailResponse)
async def get_session(
    session_id: uuid.UUID,
    service: ChatService = Depends(get_chat_service),
) -> SessionDetailResponse:
    try:
        session = await service._session_repo.get_session(session_id)
        messages = await service._session_repo.get_messages(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return SessionDetailResponse(
        id=session.id,
        title=session.title,
        is_active=session.is_active,
        messages=[
            MessageResponse(id=m.id, role=m.role, content=m.content, sources=m.sources)
            for m in messages
        ],
    )


@router.post("/{session_id}/chat/", response_model=SessionChatResponse)
async def session_chat(
    session_id: uuid.UUID,
    request: SessionChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> SessionChatResponse:
    try:
        result = await service.session_chat(
            session_id=session_id,
            question=request.question,
            document_ids=request.document_ids,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            model=request.model,
        )
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (RetrievalError, GenerationError, RerankingError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SessionChatResponse(
        answer=result.answer,
        sources=result.sources,
        session_id=session_id,
    )


@router.delete("/{session_id}/", status_code=204)
async def deactivate_session(
    session_id: uuid.UUID,
    service: ChatService = Depends(get_chat_service),
) -> None:
    try:
        await service._session_repo.deactivate_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
