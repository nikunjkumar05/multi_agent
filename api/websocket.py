from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.events import EventBroadcaster
from core.redis_client import get_redis

router = APIRouter()

@router.websocket("/ws/{task_id}")
async def websocket_endpoint(ws: WebSocket, task_id: str) -> None:
    await ws.accept()
    redis = await get_redis()
    if redis is None:
        await ws.close(code=1011, reason="Redis unavailable")
        return

    broadcaster = EventBroadcaster(redis)

    try:
        history = await broadcaster.get_history(task_id)
        for event in history:
            await ws.send_json(event)
    except WebSocketDisconnect:
        return

    try:
        async for event in broadcaster.subscribe(task_id):
            await ws.send_json(event)
    except WebSocketDisconnect:
        pass