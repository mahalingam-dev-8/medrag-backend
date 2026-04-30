"""Redis query caching (Phase 3 — placeholder)."""

import hashlib
import json
from typing import Any

from app.utils.logger import get_logger

logger = get_logger(__name__)


def _cache_key(query: str, **kwargs: Any) -> str:
    payload = json.dumps({"query": query, **kwargs}, sort_keys=True)
    return "rag:" + hashlib.sha256(payload.encode()).hexdigest()


class CacheService:
    """Wraps Redis for caching RAG responses.

    The redis client is injected so this class stays testable.
    Pass a connected redis.asyncio.Redis instance in Phase 3.
    """

    def __init__(self, redis_client: Any | None = None) -> None:
        self.redis = redis_client
        self.ttl = 3600  # 1 hour default

    async def get(self, query: str, **kwargs: Any) -> dict | None:
        if self.redis is None:
            return None
        key = _cache_key(query, **kwargs)
        try:
            raw = await self.redis.get(key)
            if raw:
                logger.info("cache_hit", key=key)
                return json.loads(raw)
        except Exception as exc:
            logger.warning("cache_get_error", error=str(exc))
        return None

    async def set(self, query: str, value: dict, **kwargs: Any) -> None:
        if self.redis is None:
            return
        key = _cache_key(query, **kwargs)
        try:
            await self.redis.setex(key, self.ttl, json.dumps(value))
        except Exception as exc:
            logger.warning("cache_set_error", error=str(exc))

    async def invalidate_all(self) -> int:
        if self.redis is None:
            return 0
        try:
            keys = await self.redis.keys("rag:*")
            if keys:
                return await self.redis.delete(*keys)
        except Exception as exc:
            logger.warning("cache_invalidate_error", error=str(exc))
        return 0
