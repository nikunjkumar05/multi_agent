import pytest
from httpx import AsyncClient, ASGITransport
from api.main import app


@pytest.mark.anyio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


@pytest.mark.anyio
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
async def test_execute_with_topology_override():
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
