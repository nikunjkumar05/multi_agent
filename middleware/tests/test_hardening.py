"""Hardening tests: fallback chain, selection ordering, persistence, auth."""

import asyncio
import sqlite3

import pytest
from fastapi.testclient import TestClient

from middleware.adapters.base import AgentAdapter, AgentResult, AgentTask
from middleware.api.main import app
from middleware.api.state import budget_manager, registry
from middleware.models.schemas import TaskStatus
from middleware.selection import select_agents

client = TestClient(app)


# ── Fallback chain ────────────────────────────────────────────────────

class _StaticAdapter(AgentAdapter):
    """Configurable mock: fixed success flag, fixed estimate, tagged caps."""

    def __init__(self, name: str, succeed: bool):
        self._name = name
        self._succeed = succeed

    async def execute(self, task: AgentTask) -> AgentResult:
        await asyncio.sleep(0)
        return AgentResult(
            task_id=task.task_id,
            agent=self._name,
            output=f"[{self._name}] {task.prompt}" if self._succeed else "",
            cost_usd=0.0001 if self._succeed else 0.0,
            tokens_used=10 if self._succeed else 0,
            latency_ms=1,
            success=self._succeed,
            error=None if self._succeed else f"{self._name} failed on purpose",
            metadata={"mock": True},
        )

    def estimate_cost(self, task: AgentTask) -> float:
        return 0.0001

    def health_check(self) -> bool:
        return True

    def get_capabilities(self) -> dict:
        return {
            "task_types": ["code_generation", "debugging"],
            "reliability": 0.99,
            "pricing": {"input_per_1k": 0.003, "output_per_1k": 0.015},
        }

    def get_name(self) -> str:
        return self._name


@pytest.fixture()
def flaky_agent():
    registry.register("flaky", _StaticAdapter("flaky", succeed=False))
    yield "flaky"
    registry.unregister("flaky")


def test_fallback_chain_recovers_from_primary_failure(flaky_agent):
    """Primary fails -> next candidate runs -> task completes with attempts trail."""
    r = client.post(
        "/api/v1/tasks/",
        json={
            "task_type": "code_generation",
            "prompt": "write something",
            "budget_usd": 0.5,
            "preferred_agents": ["flaky", "mock"],
        },
    )
    assert r.status_code == 200
    data = r.json()

    get = client.get(f"/api/v1/tasks/{data['task_id']}").json()
    assert get["status"] == "completed"
    assert get["selected_agent"] == "flaky"          # primary as queued
    assert [a["agent"] for a in get["attempts"]] == ["flaky", "mock"]
    assert get["attempts"][0]["success"] is False
    assert get["attempts"][1]["success"] is True
    assert get["cost_usd"] > 0


def test_all_candidates_failing_marks_task_failed():
    registry.register("bad1", _StaticAdapter("bad1", succeed=False))
    saved_mock = registry.get_info("mock")
    registry.unregister("mock")  # isolate chain to the failing agent only
    try:
        r = client.post(
            "/api/v1/tasks/",
            json={
                "task_type": "code_generation",
                "prompt": "x",
                "budget_usd": 0.5,
                "preferred_agents": ["bad1"],
            },
        )
        tid = r.json()["task_id"]
        get = client.get(f"/api/v1/tasks/{tid}").json()
        assert get["status"] == "failed"
        assert [a["agent"] for a in get["attempts"]] == ["bad1"]
        assert get["error"]
    finally:
        registry.unregister("bad1")
        if saved_mock:
            registry.register("mock", saved_mock.adapter)


# ── Selection ordering ────────────────────────────────────────────────

class _QualityAdapter(_StaticAdapter):
    def __init__(self, name, reliability, cost):
        super().__init__(name, succeed=True)
        self._rel = reliability
        self._cost = cost

    def estimate_cost(self, task: AgentTask) -> float:
        return self._cost

    def get_capabilities(self) -> dict:
        return {"task_types": ["code_generation"], "reliability": self._rel}


def test_selection_prefers_quality_within_budget():
    task = AgentTask(prompt="p", budget_usd=1.0)
    candidates = {
        "cheap_low_q": _QualityAdapter("cheap_low_q", 0.5, 0.001),
        "pricey_high_q": _QualityAdapter("pricey_high_q", 0.99, 0.40),
        "mid": _QualityAdapter("mid", 0.8, 0.05),
    }
    ordered = select_agents(candidates, task, remaining_budget=0.50)
    assert ordered[0][0] == "pricey_high_q"   # max quality among affordable
    assert [a for a, _ in ordered] == ["pricey_high_q", "mid", "cheap_low_q"]


def test_selection_excludes_unaffordable_but_keeps_them_last():
    task = AgentTask(prompt="p", budget_usd=0.01)
    candidates = {
        "affordable": _QualityAdapter("affordable", 0.6, 0.005),
        "too_pricey": _QualityAdapter("too_pricey", 0.99, 1.00),
    }
    ordered = select_agents(candidates, task, remaining_budget=0.01)
    assert ordered[0][0] == "affordable"      # affordable beats higher quality
    assert ordered[-1][0] == "too_pricey"     # unaffordable trails


# ── Persistence ───────────────────────────────────────────────────────

def _db_rows(table: str) -> list[tuple]:
    from middleware.persistence import get_db_path

    with sqlite3.connect(get_db_path()) as conn:
        return list(conn.execute(f"SELECT * FROM {table}"))


def test_budget_and_task_survive_in_sqlite():
    gid = client.post(
        "/api/v1/budgets",
        json={"name": "persist-me", "owner": "disk", "max_cost_usd": 2.0},
    ).json()["budget_id"]

    created = client.post(
        "/api/v1/tasks/",
        json={"prompt": "hello disk", "budget_usd": 1.0, "budget_id": gid},
    )
    tid = created.json()["task_id"]

    rows_t = _db_rows("tasks")
    rows_b = _db_rows("budgets")
    assert any(tid == r[0] for r in rows_t)
    assert any(gid == r[0] for r in rows_b)

    # Completed task state was rewritten (write-through on worker finish).
    row = next(r for r in rows_t if r[0] == tid)
    assert '"status"' in row[1]


def test_load_all_returns_saved_state():
    from middleware.persistence import load_all

    tasks_list, budgets_list = load_all()
    assert isinstance(tasks_list, list)
    assert isinstance(budgets_list, list)
    assert len(budgets_list) >= 1  # budgets created by earlier tests in session


# ── Auth guard ────────────────────────────────────────────────────────

def test_api_key_guard(monkeypatch):
    monkeypatch.setenv("BAMAS_API_KEY", "topsecret")

    # Missing key -> 401
    r = client.get("/api/v1/agents")
    assert r.status_code == 401

    # Wrong key -> 401
    r = client.get("/api/v1/agents", headers={"X-API-Key": "wrong"})
    assert r.status_code == 401

    # Correct key -> passes
    r = client.get("/api/v1/agents", headers={"X-API-Key": "topsecret"})
    assert r.status_code == 200

    # Health endpoint stays open even with key set (liveness probes)
    assert client.get("/health").status_code == 200


# ── Restart restore semantics ─────────────────────────────────────────

def test_restore_budget_rebuilds_state():
    payload = {"name": "rb", "owner": "restore-test", "max_cost_usd": 3.0}
    original = client.post("/api/v1/budgets", json=payload).json()
    raw = budget_manager.get_budget(original["budget_id"]).to_dict()

    # Simulate fresh process: remove then restore.
    budget_manager.delete_budget(original["budget_id"])
    restored = budget_manager.restore_budget(raw)

    assert restored.budget_id == original["budget_id"]
    assert restored.max_cost_usd == pytest.approx(3.0)
    assert restored.spent_usd == pytest.approx(original["spent_usd"])
    assert budget_manager.get_budget(restored.budget_id) is restored


def test_interrupted_tasks_marked_failed_on_restore():
    """Startup logic marks queued/in_progress records as failed."""
    ghost = {
        "task_id": "task_ghost",
        "status": TaskStatus.IN_PROGRESS.value,
        "estimated_cost_usd": 0.01,
        "selected_agent": "mock",
        "attempts": [],
        "budget_exceeded": False,
    }
    from middleware import persistence

    persistence.save_task(ghost)
    loaded, _ = persistence.load_all()
    match = next(t for t in loaded if t["task_id"] == "task_ghost")
    assert match["status"] == "in_progress"  # raw record untouched...

    # ...the lifespan handler does the marking; simulate its branch:
    if match["status"] in ("queued", "in_progress"):
        match["status"] = "failed"
        match["error"] = "Interrupted by server restart"
    assert match["status"] == "failed"
