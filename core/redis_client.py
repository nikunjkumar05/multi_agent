import logging

import redis.asyncio as aioredis

from core.config import settings

logger = logging.getLogger(__name__)

_redis_client: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis | None:
    global _redis_client
    try:
        if _redis_client is None:
            _redis_client = aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
            )
        return _redis_client
    except Exception:
        logger.warning("Redis unavailable, events will be skipped")
        return None


async def close_redis() -> None:
    global _redis_client
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None