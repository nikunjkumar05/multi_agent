"""
End-to-end integration tests for all 5 BAMAS topologies.

Each test runs a REAL task through the full graph with Mistral LLM calls.
No mocks. Proves the system actually works.

Run with: pytest tests/integration/test_topologies.py -v --timeout=120
"""
import pytest

from core.budget import BudgetTracker


def _budget(usd: float = 0.50) -> BudgetTracker:
    return BudgetTracker(max_cost_usd=usd)


@pytest.mark.anyio
async def test_single_topology():
    """Single topology: planner -> executor -> validator -> finalizer."""
    from agent.graph import run_task

    result = await run_task(
        task="What is 3 + 5? Answer with just the number.",
        budget=_budget(0.50),
        task_id="test-single-001",
        topology_override="single",
    )

    assert result["status"] == "completed"
    assert result["topology"] == "single"
    assert result["final_result"] is not None
    assert "8" in result["final_result"]
    assert result["budget_spent_pct"] > 0


@pytest.mark.anyio
async def test_pipeline_topology():
    """Pipeline topology: planner -> executor -> validator -> judge -> finalizer (sequential steps)."""
    from agent.graph import run_task

    result = await run_task(
        task="First, calculate 2 * 3. Then, add 4 to the result. Finally, subtract 1.",
        budget=_budget(0.50),
        task_id="test-pipeline-001",
        topology_override="pipeline",
    )

    assert result["status"] == "completed"
    assert result["topology"] == "pipeline"
    assert result["final_result"] is not None
    assert result["budget_spent_pct"] > 0


@pytest.mark.anyio
async def test_supervisor_topology():
    """Supervisor topology: planner -> supervisor dispatch -> executor -> validator -> judge -> finalizer."""
    from agent.graph import run_task

    result = await run_task(
        task="What is the capital of France? Answer in one sentence.",
        budget=_budget(0.50),
        task_id="test-supervisor-001",
        topology_override="supervisor",
    )

    assert result["status"] == "completed"
    assert result["topology"] == "supervisor"
    assert result["final_result"] is not None
    assert result["budget_spent_pct"] > 0


@pytest.mark.anyio
async def test_fanout_topology():
    """Fanout topology: planner -> dispatcher -> parallel workers -> aggregator -> judge -> finalizer."""
    from agent.graph import run_task

    result = await run_task(
        task="List 3 countries in Europe. Just list them, one per line.",
        budget=_budget(0.50),
        task_id="test-fanout-001",
        topology_override="fanout",
    )

    assert result["status"] == "completed"
    assert result["topology"] == "fanout"
    assert result["final_result"] is not None
    assert result["budget_spent_pct"] > 0


@pytest.mark.anyio
async def test_ensemble_topology():
    """Ensemble topology: planner -> 3 parallel agents -> judge -> finalizer."""
    from agent.graph import run_task

    result = await run_task(
        task="What is 10 / 2? Answer with just the number.",
        budget=_budget(0.50),
        task_id="test-ensemble-001",
        topology_override="ensemble",
    )

    assert result["status"] == "completed"
    assert result["topology"] == "ensemble"
    assert result["final_result"] is not None
    assert result["budget_spent_pct"] > 0


@pytest.mark.anyio
async def test_topology_override_single():
    """Verify topology override is respected."""
    from agent.graph import run_task

    result = await run_task(
        task="Say hello.",
        budget=_budget(0.10),
        task_id="test-override-001",
        topology_override="single",
    )

    assert result["topology"] == "single"


@pytest.mark.anyio
async def test_budget_tracking_across_topologies():
    """Verify budget tracking works for multiple topologies."""
    from agent.graph import run_task

    for topo in ["single", "pipeline", "supervisor"]:
        result = await run_task(
            task="What is 2+2?",
            budget=_budget(0.50),
            task_id=f"test-budget-{topo}",
            topology_override=topo,
        )
        assert result["budget_spent_pct"] > 0, f"Budget not tracked for {topo}"
        assert result["status"] == "completed", f"Task failed for {topo}"
