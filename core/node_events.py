from typing import Any

from core.events import EventBroadcaster
from core.redis_client import get_redis


async def emit_event(task_id: str, event_type: str, data: dict[str, Any]) -> None:
    if not task_id:
        return
    redis = await get_redis()
    if redis is None:
        return
    broadcaster = EventBroadcaster(redis)
    await broadcaster.publish(task_id, event_type, data)
