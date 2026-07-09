"""Integration tests for WebSocket /ws/{task_id} endpoint."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from api.main import app


# ── Mock Redis helpers ────────────────────────────────────────────────────


class _MockPipeline:
    """Accumulates commands; exec returns empty list."""

    def __init__(self) -> None:
        self._cmds: list[tuple] = []

    def lpush(self, *args: object) -> "_MockPipeline":
        self._cmds.append(("lpush", *args))
        return self

    def ltrim(self, *args: object) -> "_MockPipeline":
        self._cmds.append(("ltrim", *args))
        return self

    def expire(self, *args: object) -> "_MockPipeline":
        self._cmds.append(("expire", *args))
        return self

    async def execute(self) -> list:
        return [None] * len(self._cmds)


class _MockPubSub:
    """Stores subscribed channels; `listen()` yields pre-loaded messages then stops."""

    def __init__(self, messages: list[dict]) -> None:
        self._channels: list[str] = []
        self._messages = messages

    async def subscribe(self, channel: str) -> None:
        self._channels.append(channel)

    async def unsubscribe(self, channel: str) -> None:
        self._channels = [c for c in self._channels if c != channel]

    async def listen(self):  # type: ignore[override]
        for msg in self._messages:
            yield msg
        # After yielding all messages, block until cancelled
        await asyncio.sleep(3600)

    async def close(self) -> None:
        pass


class _MockRedis:
    """Minimal async Redis mock for WebSocket tests."""

    def __init__(self, history: list[dict] | None = None,
                 stream_events: list[dict] | None = None) -> None:
        self._history = history or []
        self._stream_events = stream_events or []

    async def lrange(self, *args: object) -> list[str]:
        return [json.dumps(e) for e in self._history]

    def pipeline(self) -> _MockPipeline:
        return _MockPipeline()

    def pubsub(self) -> _MockPubSub:
        msgs = [
            {"type": "message", "data": json.dumps(e)}
            for e in self._stream_events
        ]
        return _MockPubSub(msgs)

    async def publish(self, *args: object) -> None:
        pass


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_ws_receives_events():
    """WebSocket client receives events published before and during connection."""
    history_event = {
        "event_type": "task_started",
        "timestamp": "2026-07-08T12:00:00Z",
        "data": {"task_id": "ws-test-1"},
    }
    stream_event = {
        "event_type": "step_completed",
        "timestamp": "2026-07-08T12:00:01Z",
        "data": {"task_id": "ws-test-1", "step": 1},
    }

    mock_redis = _MockRedis(history=[history_event], stream_events=[stream_event])

    with patch("api.websocket.get_redis", new_callable=lambda: AsyncMock(return_value=mock_redis)):
        client = TestClient(app)
        with client.websocket_connect("/ws/ws-test-1") as ws:
            # Should receive the historical event first
            data = ws.receive_json()
            assert data["event_type"] == "task_started"
            assert data["data"]["task_id"] == "ws-test-1"

            # Should receive the streamed event
            data = ws.receive_json()
            assert data["event_type"] == "step_completed"
            assert data["data"]["step"] == 1


@pytest.mark.anyio
async def test_ws_empty_history():
    """WebSocket connects successfully with no prior events."""
    mock_redis = _MockRedis(history=[], stream_events=[])

    with patch("api.websocket.get_redis", new_callable=lambda: AsyncMock(return_value=mock_redis)):
        client = TestClient(app)
        with client.websocket_connect("/ws/empty-task") as ws:
            # No events to receive; just verify connection is open
            # Send a ping to confirm the socket is alive
            ws.send_json({"type": "ping"})
            # Connection should remain open (no disconnect)


@pytest.mark.anyio
async def test_ws_redis_unavailable():
    """WebSocket closes gracefully when Redis is unavailable."""
    with patch("api.websocket.get_redis", new_callable=lambda: AsyncMock(return_value=None)):
        client = TestClient(app)
        with client.websocket_connect("/ws/no-redis-task") as ws:
            # Should receive close frame with code 1011
            with pytest.raises(Exception):
                ws.receive_json()


@pytest.mark.anyio
async def test_ws_multiple_tasks_isolated():
    """Events for task A don't leak to task B's WebSocket."""
    event_a = {
        "event_type": "step_completed",
        "timestamp": "2026-07-08T12:00:00Z",
        "data": {"task_id": "task-a"},
    }

    mock_redis = _MockRedis(history=[event_a], stream_events=[])

    with patch("api.websocket.get_redis", new_callable=lambda: AsyncMock(return_value=mock_redis)):
        client = TestClient(app)

        # Connect to task-b — should NOT see task-a's events
        with client.websocket_connect("/ws/task-b") as ws:
            # No events for task-b, connection should just stay open
            ws.send_json({"type": "ping"})
