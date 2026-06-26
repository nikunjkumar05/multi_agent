import logging
from typing import Any

from core.events import EventBroadcaster
from core.redis_client import get_redis

log = logging.getLogger(__name__)


async def emit_event(task_id: str, event_type: str, data: dict[str, Any]) -> None:
    if not task_id:
        return
    try:
        redis = await get_redis()
        if redis is None:
            return
        broadcaster = EventBroadcaster(redis)
        await broadcaster.publish(task_id, event_type, data)
    except Exception:
        log.warning("emit_event failed for %s/%s", task_id, event_type, exc_info=True)
