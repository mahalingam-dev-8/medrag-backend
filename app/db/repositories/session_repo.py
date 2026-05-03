import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChatMessage, ChatSession
from app.utils.exceptions import SessionNotFoundError


class SessionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_session(self, title: str | None = None) -> ChatSession:
        obj = ChatSession(title=title, is_active=True)
        self.session.add(obj)
        await self.session.flush()
        return obj

    async def get_session(self, session_id: uuid.UUID) -> ChatSession:
        result = await self.session.execute(
            select(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.is_active == True,  # noqa: E712
            )
        )
        chat_session = result.scalar_one_or_none()
        if chat_session is None:
            raise SessionNotFoundError(str(session_id))
        return chat_session

    async def get_messages(self, session_id: uuid.UUID, limit: int = 20) -> list[ChatMessage]:
        result = await self.session.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def add_message(
        self,
        session_id: uuid.UUID,
        role: str,
        content: str,
        sources: list | None = None,
    ) -> ChatMessage:
        obj = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            sources=sources or [],
        )
        self.session.add(obj)
        await self.session.flush()
        return obj

    async def set_title(self, session_id: uuid.UUID, title: str) -> None:
        chat_session = await self.get_session(session_id)
        chat_session.title = title

    async def list_sessions(self, limit: int = 50) -> list[ChatSession]:
        result = await self.session.execute(
            select(ChatSession)
            .where(ChatSession.is_active == True)  # noqa: E712
            .order_by(ChatSession.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def deactivate_session(self, session_id: uuid.UUID) -> None:
        chat_session = await self.get_session(session_id)
        chat_session.is_active = False
