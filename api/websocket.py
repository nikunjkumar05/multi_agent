import asyncio
import logging
from datetime import UTC, datetime

import jwt
from fastapi import APIRouter, Query, WebSocket

from core.config import DEFAULT_JWT_SECRET
from core.events import EventBroadcaster
from core.redis_client import get_redis

log = logging.getLogger(__name__)

router = APIRouter()

_HEARTBEAT_INTERVAL = 20  # seconds


def _validate_ws_token(token: str | None) -> bool:
    """Validate JWT token from query param. Returns True if valid or auth disabled."""
    from core.config import settings
    if settings.jwt_secret == DEFAULT_JWT_SECRET:
        return True
    if token is None:
        return False
    try:
        jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        return True
    except jwt.PyJWTError:
        return False


@router.websocket("/ws/{task_id}")
async def websocket_endpoint(
    ws: WebSocket, task_id: str, token: str | None = Query(default=None)
) -> None:
    if not _validate_ws_token(token):
        await ws.accept()
        await ws.close(code=4001, reason="Unauthorized")
        return

    await ws.accept()
    redis = await get_redis()
    if redis is None:
        await ws.close(code=1011, reason="Redis unavailable")
        return

    broadcaster = EventBroadcaster(redis)

    # Send backlog of events that occurred before the client connected
    try:
        history = await broadcaster.get_history(task_id)
        for event in history:
            await ws.send_json(event)
    except Exception:
        # Client disconnected during backlog send — close cleanly
        return

    # Shared flag: heartbeat and main loop both check this before sending.
    # Once set, no further ws.send_json() calls are attempted.
    closed = asyncio.Event()

    async def _heartbeat() -> None:
        """Keeps the connection alive through proxy idle-timeout windows."""
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            if closed.is_set():
                break
            try:
                await ws.send_json({
                    "event_type": "ping",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "data": {},
                })
            except Exception:
                # WS dead — signal main loop to stop
                closed.set()
                break

    hb_task = asyncio.create_task(_heartbeat())
    pubsub = None
    try:
        # Manual PubSub lifecycle so we can close it in finally
        # (the generator's finally block may not run if we break out)
        pubsub = redis.pubsub()
        await pubsub.subscribe(f"events:{task_id}")
        async for message in pubsub.listen():
            if closed.is_set():
                break
            if message["type"] != "message":
                continue
            try:
                event = __import__("json").loads(message["data"])
                await ws.send_json(event)
            except Exception:
                # WS dead — stop immediately
                closed.set()
                break
    except Exception:
        # Catch-all: any exception from WS or PubSub kills the loop cleanly
        closed.set()
    finally:
        hb_task.cancel()
        # Suppress CancelledError from heartbeat task
        try:
            await hb_task
        except asyncio.CancelledError:
            pass
        # Explicitly close PubSub to unsubscribe from Redis channel
        if pubsub is not None:
            try:
                await pubsub.unsubscribe(f"events:{task_id}")
                await pubsub.close()
            except Exception:
                pass
