from collections.abc import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.services.chat_service import ChatService
from app.services.ingestion_service import IngestionService
from app.services.search_service import SearchService


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_db():
        yield session


async def get_chat_service(
    session: AsyncSession = Depends(get_session),
) -> ChatService:
    return ChatService(session)


async def get_ingestion_service(
    session: AsyncSession = Depends(get_session),
) -> IngestionService:
    return IngestionService(session)


async def get_search_service(
    session: AsyncSession = Depends(get_session),
) -> SearchService:
    return SearchService(session)
