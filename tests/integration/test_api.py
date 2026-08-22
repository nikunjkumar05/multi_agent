import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app

requires_api_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="requires OPENAI_API_KEY for live LLM calls",
)


@pytest.mark.anyio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


@pytest.mark.anyio
@requires_api_key
async def test_execute_returns_task_id():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/execute", json={"task": "What is 2+2?", "budget_usd": 0.05})
        assert r.status_code == 200
        data = r.json()
        assert "task_id" in data
        assert data["status"] == "pending"


@pytest.mark.anyio
async def test_get_task_not_found():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/tasks/nonexistent-id")
        assert r.status_code == 404


@pytest.mark.anyio
async def test_get_audit_not_found():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/audit/nonexistent-id")
        assert r.status_code == 404


@pytest.mark.anyio
@requires_api_key
async def test_execute_with_topology_override():
    with patch("core.redis_client.get_redis", new_callable=lambda: AsyncMock(return_value=None)):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post("/execute", json={
                "task": "Write a function",
                "budget_usd": 0.10,
                "topology": "pipeline",
            })
            assert r.status_code == 200
            data = r.json()
            assert "task_id" in data


@pytest.mark.anyio
async def test_rl_reset_endpoint():
    mock_redis = AsyncMock()
    mock_redis.delete = AsyncMock()

    async def mock_scan_iter(match):
        return
        yield

    mock_redis.scan_iter = mock_scan_iter

    # Mock pipeline for both reset() and get_stats() -> load()
    pipe = AsyncMock()
    pipe.hset = MagicMock()
    pipe.set = MagicMock()
    pipe.execute = AsyncMock(return_value=[
        # 5 arm results (empty dicts = fresh arms) + total_tasks
        {}, {}, {}, {}, {}, b"0"
    ])
    mock_redis.pipeline = AsyncMock(return_value=pipe)

    with patch("api.routes.rl.get_redis", new_callable=lambda: AsyncMock(return_value=mock_redis)):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Without confirm — should reject
            r = await client.post("/rl/reset")
            assert r.status_code == 200
            assert r.json()["error"] == "Pass confirm=yes to reset RL state"

            # With confirm=yes — should succeed
            r = await client.post("/rl/reset?confirm=yes")
            assert r.status_code == 200
            data = r.json()
            assert data["status"] == "reset"
            assert "arms" in data
            assert data["arms"]["total_tasks"] == 0
