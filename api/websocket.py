import asyncio
from datetime import datetime, timezone

import jwt
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from core.events import EventBroadcaster
from core.redis_client import get_redis

router = APIRouter()

_HEARTBEAT_INTERVAL = 20  # seconds
_DEFAULT_SECRET = "change-me-in-production"


def _validate_ws_token(token: str | None) -> bool:
    """Validate JWT token from query param. Returns True if valid or auth disabled."""
    from core.config import settings
    if settings.jwt_secret == _DEFAULT_SECRET:
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
    except WebSocketDisconnect:
        return

    async def _heartbeat() -> None:
        """Keeps the connection alive through proxy idle-timeout windows."""
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            try:
                await ws.send_json({
                    "event_type": "ping",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {},
                })
            except Exception:
                break

    hb_task = asyncio.create_task(_heartbeat())
    try:
        async for event in broadcaster.subscribe(task_id):
            await ws.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        hb_task.cancel()
