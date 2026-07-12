import logging
from typing import Any

from core.events import EventBroadcaster
from core.redis_client import get_redis

log = logging.getLogger(__name__)

_broadcaster_cache: EventBroadcaster | None = None
_cached_redis: Any = None


async def emit_event(task_id: str, event_type: str, data: dict[str, Any]) -> None:
    global _broadcaster_cache, _cached_redis
    if not task_id:
        return
    try:
        redis = await get_redis()
        if redis is None:
            return
        # Reuse cached broadcaster if same Redis instance
        if _broadcaster_cache is None or _cached_redis is not redis:
            _broadcaster_cache = EventBroadcaster(redis)
            _cached_redis = redis
        await _broadcaster_cache.publish(task_id, event_type, data)
    except Exception:
        log.warning("emit_event failed for %s/%s", task_id, event_type, exc_info=True)
