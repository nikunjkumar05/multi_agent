import pytest
from fastapi.testclient import TestClient
from middleware.api.main import app

client = TestClient(app)


def test_health_check():
    """Test that the API is up and running."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_and_get_task():
    """Full lifecycle: create task (ephemeral budget) and fetch receipt."""
    payload = {
        "task_type": "code_generation",
        "prompt": "Write a python function to add two numbers",
        "budget_usd": 0.50,
        "timeout_seconds": 60,
    }

    response = client.post("/api/v1/tasks/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["selected_agent"] == "mock"
    assert "ws_url" not in data  # fake field removed

    # Budget receipt fields present even pre-execution
    assert data["budget_id"] is not None
    assert data["budget_remaining_usd"] is not None
    assert data["budget_exceeded"] is False

    task_id = data["task_id"]

    # TestClient runs background tasks before post() returns,
    # so the mock has already finished by now.
    get_response = client.get(f"/api/v1/tasks/{task_id}")
    assert get_response.status_code == 200
    get_data = get_response.json()
    assert get_data["task_id"] == task_id
    assert get_data["status"] in ["queued", "in_progress", "completed", "failed"]
    if get_data["status"] == "completed":
        assert get_data["cost_usd"] >= 0.0
        assert get_data["budget_spent_usd"] >= get_data["cost_usd"]
        assert get_data["budget_exceeded"] is False


def test_get_nonexistent_task():
    """Requesting an invalid task ID returns a 404."""
    response = client.get("/api/v1/tasks/task_invalid999")
    assert response.status_code == 404


# ── Budget endpoints ──────────────────────────────────────────────────

def test_create_and_get_budget():
    payload = {"name": "Demo", "owner": "intern", "max_cost_usd": 10.0, "max_tasks": 50}
    r = client.post("/api/v1/budgets", json=payload)
    assert r.status_code == 201
    data = r.json()
    assert data["budget_id"].startswith("budget_")
    assert data["max_cost_usd"] == 10.0
    assert data["remaining_usd"] == 10.0
    assert data["status"] == "active"

    gid = data["budget_id"]
    got = client.get(f"/api/v1/budgets/{gid}")
    assert got.status_code == 200
    assert got.json()["budget_id"] == gid


def test_list_budgets_by_owner():
    client.post("/api/v1/budgets", json={"name": "A", "owner": "alice", "max_cost_usd": 5.0})
    listed = client.get("/api/v1/budgets?owner=alice")
    assert listed.status_code == 200
    assert all(b["owner"] == "alice" for b in listed.json())


# ── Budget enforcement in the task flow ───────────────────────────────

def test_task_denied_when_budget_too_small():
    """Estimated cost above remaining budget -> HTTP 402, nothing queued."""
    tiny = client.post(
        "/api/v1/budgets",
        json={"name": "pocket change", "owner": "x", "max_cost_usd": 0.00001},
    ).json()["budget_id"]

    response = client.post(
        "/api/v1/tasks/",
        json={
            "task_type": "code_generation",
            "prompt": "Write a CSV parser",
            "budget_usd": 0.50,
            "budget_id": tiny,
        },
    )
    assert response.status_code == 402
    assert "exhausted" in response.json()["detail"]


def test_task_denied_on_unknown_budget():
    response = client.post(
        "/api/v1/tasks/",
        json={
            "task_type": "code_generation",
            "prompt": "hello",
            "budget_usd": 0.50,
            "budget_id": "budget_doesnotexist",
        },
    )
    assert response.status_code == 404


def test_warning_fired_near_limit():
    """Spent past warn_threshold (80%) but under hard limit -> allowed with warning."""
    gid = client.post(
        "/api/v1/budgets",
        json={"name": "warnme", "owner": "w", "max_cost_usd": 0.001},
    ).json()["budget_id"]

    # Pre-spend 90% directly through the manager (simulates prior tasks).
    from middleware.api.state import budget_manager

    budget_manager.get_budget(gid).record_usage(0.0009, 0)

    response = client.post(
        "/api/v1/tasks/",
        json={"task_type": "code_generation", "prompt": "hi", "budget_usd": 0.001, "budget_id": gid},
    )
    assert response.status_code == 200
    assert response.json()["warning"] is not None


def test_spend_recorded_to_persistent_budget():
    """Successful task deducts actual cost from the linked persistent budget."""
    gid = client.post(
        "/api/v1/budgets",
        json={"name": "wallet", "owner": "p", "max_cost_usd": 1.0},
    ).json()["budget_id"]

    created = client.post(
        "/api/v1/tasks/",
        json={"task_type": "code_generation", "prompt": "do it", "budget_usd": 0.5, "budget_id": gid},
    )
    assert created.status_code == 200

    after = client.get(f"/api/v1/budgets/{gid}").json()
    assert after["spent_usd"] > 0
    assert after["tasks_completed"] == 1
    assert after["remaining_usd"] < 1.0

    receipt = client.get(f"/api/v1/tasks/{created.json()['task_id']}").json()
    assert receipt["budget_id"] == gid
    assert abs(receipt["budget_spent_usd"] - after["spent_usd"]) < 1e-9


# ── Agents endpoint ───────────────────────────────────────────────────

def test_agents_endpoint_lists_mock():
    r = client.get("/api/v1/agents")
    assert r.status_code == 200
    body = r.json()
    ids = [a["agent_id"] for a in body["agents"]]
    assert "mock" in ids
    assert body["summary"]["total_agents"] >= 1


# ── Cancel semantics ──────────────────────────────────────────────────

def test_cancel_completed_task_conflict():
    """Cancelling an already-finished task returns 409 (cancel-race guard)."""
    from middleware.api.routes.tasks import tasks_db
    from middleware.models.schemas import TaskStatus

    payload = {
        "task_type": "debugging",
        "prompt": "Fix this error: IndexError",
        "budget_usd": 0.20,
    }
    response = client.post("/api/v1/tasks/", json=payload)
    assert response.status_code == 200
    task_id = response.json()["task_id"]

    # Background mock already completed the task inside post() above.
    assert tasks_db[task_id]["status"] == TaskStatus.COMPLETED

    cancel_response = client.delete(f"/api/v1/tasks/{task_id}")
    assert cancel_response.status_code == 409
    assert client.get(f"/api/v1/tasks/{task_id}").json()["status"] == "completed"


def test_cancel_queued_task():
    """A queued (not yet executed) task can be cancelled."""
    from middleware.api.routes.tasks import tasks_db
    from middleware.models.schemas import TaskStatus

    task_id = "task_cancelfake"
    tasks_db[task_id] = {
        "task_id": task_id,
        "status": TaskStatus.QUEUED,
        "estimated_cost_usd": 0.01,
        "estimated_tokens": 100,
        "selected_agent": "mock",
        "selected_tier": "standard",
        "output": None,
        "error": None,
        "cost_usd": None,
        "tokens_used": None,
        "latency_ms": None,
        "quality_score": None,
        "budget_id": None,
        "budget_spent_usd": None,
        "budget_remaining_usd": None,
        "budget_exceeded": False,
        "warning": None,
        "attempts": [],
    }
    try:
        cancel_response = client.delete(f"/api/v1/tasks/{task_id}")
        assert cancel_response.status_code == 200
        assert client.get(f"/api/v1/tasks/{task_id}").json()["status"] == "cancelled"
    finally:
        tasks_db.pop(task_id, None)


def test_background_worker_skips_cancelled_task():
    """The worker must never overwrite a CANCELLED status (race guard)."""
    import asyncio
    from middleware.adapters.base import AgentTask
    from middleware.api.routes.tasks import execute_task_background, tasks_db
    from middleware.models.schemas import TaskStatus

    task_id = "task_racefake"
    agent_task = AgentTask(task_id=task_id, prompt="x", budget_usd=1.0)
    tasks_db[task_id] = {"task_id": task_id, "status": TaskStatus.CANCELLED}
    try:
        asyncio.run(execute_task_background(task_id, ["mock"], agent_task))
        assert tasks_db[task_id]["status"] == TaskStatus.CANCELLED
    finally:
        tasks_db.pop(task_id, None)


def test_background_worker_blocks_on_exhausted_budget():
    """Worker refuses execution when the linked budget cannot afford the estimate."""
    import asyncio
    from middleware.adapters.base import AgentTask
    from middleware.api.routes.tasks import execute_task_background, tasks_db
    from middleware.api.state import budget_manager
    from middleware.models.schemas import TaskStatus

    gid = budget_manager.create_budget(name="dry", owner="t", max_cost_usd=0.00001).budget_id
    task_id = "task_dryfake"
    agent_task = AgentTask(task_id=task_id, prompt="x", budget_usd=1.0)
    tasks_db[task_id] = {
        "task_id": task_id,
        "status": TaskStatus.QUEUED,
        "estimated_cost_usd": 0.0001,  # exceeds the tiny limit
        "selected_agent": "mock",
        "budget_id": gid,
        "attempts": [],
    }
    try:
        asyncio.run(execute_task_background(task_id, ["mock"], agent_task))
        assert tasks_db[task_id]["status"] == TaskStatus.FAILED
        assert "Budget exhausted" in tasks_db[task_id]["error"]
    finally:
        tasks_db.pop(task_id, None)
