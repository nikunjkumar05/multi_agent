import json
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as aioredis


class EventBroadcaster:
    def __init__(self, redis: aioredis.Redis) -> None:
        self.redis = redis

    async def publish(self, task_id: str, event_type: str, data: dict[str, Any]) -> None:
        event = {
            "event_type": event_type,
            "timestamp": datetime.now(UTC).isoformat(),
            "data": data,
        }
        event_json = json.dumps(event, default=str)
        key = f"events:{task_id}:log"
        await self.redis.lpush(key, event_json)
        await self.redis.ltrim(key, 0, 99)
        await self.redis.expire(key, 3600)

        await self.redis.publish(f"events:{task_id}", event_json)

    async def get_history(self, task_id: str) -> list[dict[str, Any]]:
        key = f"events:{task_id}:log"
        raw_list = await self.redis.lrange(key, 0, -1)
        return [json.loads(item) for item in reversed(raw_list)]
    
    async def subscribe(self, task_id: str) -> AsyncGenerator[dict[str, Any], None]:
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(f"events:{task_id}")
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield json.loads(message["data"])
        finally:
            await pubsub.unsubscribe(f"events:{task_id}")
            await pubsub.close()