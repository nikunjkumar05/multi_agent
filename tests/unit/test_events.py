import json
from unittest.mock import AsyncMock

import pytest

from core.events import EventBroadcaster


class TestEventBroadcaster:
    @pytest.mark.asyncio
    async def test_publish_stores_event(self):
        mock_redis = AsyncMock()
        broadcaster = EventBroadcaster(mock_redis)
        await broadcaster.publish("task-1", "step_completed", {"step_id": 1})
        mock_redis.lpush.assert_called_once()
        mock_redis.ltrim.assert_called_once()
        mock_redis.expire.assert_called_once()
        mock_redis.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_stores_correct_json(self):
        mock_redis = AsyncMock()
        broadcaster = EventBroadcaster(mock_redis)
        await broadcaster.publish("task-1", "test_event", {"key": "value"})
        call_args = mock_redis.lpush.call_args
        event_json = call_args[0][1]
        event = json.loads(event_json)
        assert event["event_type"] == "test_event"
        assert event["data"]["key"] == "value"
        assert "timestamp" in event

    @pytest.mark.asyncio
    async def test_publish_uses_correct_redis_key(self):
        mock_redis = AsyncMock()
        broadcaster = EventBroadcaster(mock_redis)
        await broadcaster.publish("task-123", "test", {})
        key = mock_redis.lpush.call_args[0][0]
        assert key == "events:task-123:log"

    @pytest.mark.asyncio
    async def test_publish_publishes_to_correct_channel(self):
        mock_redis = AsyncMock()
        broadcaster = EventBroadcaster(mock_redis)
        await broadcaster.publish("task-123", "test", {})
        channel = mock_redis.publish.call_args[0][0]
        assert channel == "events:task-123"

    @pytest.mark.asyncio
    async def test_publish_sets_ttl(self):
        mock_redis = AsyncMock()
        broadcaster = EventBroadcaster(mock_redis)
        await broadcaster.publish("task-1", "test", {})
        mock_redis.expire.assert_called_once_with("events:task-1:log", 3600)

    @pytest.mark.asyncio
    async def test_publish_trims_to_100(self):
        mock_redis = AsyncMock()
        broadcaster = EventBroadcaster(mock_redis)
        await broadcaster.publish("task-1", "test", {})
        mock_redis.ltrim.assert_called_once_with("events:task-1:log", 0, 99)


class TestGetHistory:
    @pytest.mark.asyncio
    async def test_returns_empty_for_no_events(self):
        mock_redis = AsyncMock()
        mock_redis.lrange = AsyncMock(return_value=[])
        broadcaster = EventBroadcaster(mock_redis)
        history = await broadcaster.get_history("task-1")
        assert history == []

    @pytest.mark.asyncio
    async def test_returns_parsed_events(self):
        mock_redis = AsyncMock()
        event1 = json.dumps({"event_type": "step_started", "data": {}})
        event2 = json.dumps({"event_type": "step_completed", "data": {}})
        mock_redis.lrange = AsyncMock(return_value=[event2, event1])
        broadcaster = EventBroadcaster(mock_redis)
        history = await broadcaster.get_history("task-1")
        assert len(history) == 2
        assert history[0]["event_type"] == "step_started"
        assert history[1]["event_type"] == "step_completed"

    @pytest.mark.asyncio
    async def test_uses_correct_key(self):
        mock_redis = AsyncMock()
        mock_redis.lrange = AsyncMock(return_value=[])
        broadcaster = EventBroadcaster(mock_redis)
        await broadcaster.get_history("task-42")
        mock_redis.lrange.assert_called_once_with("events:task-42:log", 0, -1)
