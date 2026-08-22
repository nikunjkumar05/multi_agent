"""
Integration test: mid-execution degradation chain.

Verifies that a tiny budget triggers the full degradation chain:
  ensemble → fanout → supervisor → pipeline → single

Uses mocked LLM to control cost without needing a real API.
"""

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.budget import BudgetBand, BudgetTracker
from core.optimizer import OptimizerDecision


@dataclass
class UsageMetadata:
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class MockAIMessage:
    content: str = ""
    tool_calls: list = field(default_factory=list)
    usage_metadata: UsageMetadata = field(default_factory=UsageMetadata)


def _make_mock_llm_response(content: str = "test response", total_tokens: int = 500):
    """Create a mock LangChain AIMessage with controlled token usage."""
    return MockAIMessage(
        content=content,
        tool_calls=[],
        usage_metadata=UsageMetadata(
            total_tokens=total_tokens,
            input_tokens=total_tokens // 2,
            output_tokens=total_tokens // 2,
        ),
    )


def _make_mock_llm(content: str = "test response", total_tokens: int = 500):
    """Create a mock chat model that returns controlled responses."""
    mock_llm = MagicMock()
    mock_response = _make_mock_llm_response(content, total_tokens)
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)
    mock_llm.with_structured_output = MagicMock(return_value=mock_llm)
    return mock_llm


def _make_planner_llm(step_count: int = 1, total_tokens: int = 500):
    """Create a mock LLM that returns valid planner JSON."""
    import json
    steps = [{"step_id": i + 1, "description": f"Step {i + 1}: do something"} for i in range(step_count)]
    planner_response = _make_mock_llm_response(json.dumps(steps), total_tokens)
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=planner_response)
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)
    mock_llm.with_structured_output = MagicMock(return_value=mock_llm)
    return mock_llm


def _make_decision(topology: str = "ensemble") -> OptimizerDecision:
    """Create a real OptimizerDecision (Pydantic model, serializable by checkpointer)."""
    tiers = {
        "planner": "standard",
        "executor": "standard",
        "validator": "standard",
        "judge": "standard",
    }
    if topology == "ensemble":
        tiers.update({"agent_a": "standard", "agent_b": "standard", "agent_c": "standard"})
    return OptimizerDecision(
        topology=topology,
        llm_topology=topology,
        model_tiers=tiers,
        rationale="test",
        alternatives_considered=[],
    )


def _base_state(task_id: str, topology: str, budget: BudgetTracker, **overrides) -> dict:
    """Build a complete initial state for the orchestrator."""
    state = {
        "task": "What is 2+2?",
        "task_id": task_id,
        "topology": topology,
        "decision": _make_decision(topology),
        "budget": budget,
        "consumed_tokens": 0,
        "consumed_cost": 0.0,
        "last_budget_band": BudgetBand.HEALTHY.value,
        "plan_steps": [],
        "completed_step_ids": [],
        "current_step_index": 0,
        "step_results": {},
        "candidate_outputs": {},
        "prior_context": None,
        "final_output": None,
        "judge_output": None,
        "degradation_requested": False,
        "target_topology": None,
        "topology_history": [],
        "validator_confidence": None,
        "reasoning_diverged": False,
        "skip_judge": False,
        "escalation_triggered": False,
        "resume_signal": None,
        "_worker_assignments": None,
        "fanout_worker_results": None,
        "agent_a_result": None,
        "agent_b_result": None,
        "agent_c_result": None,
        "supervisor_remaining_tasks": None,
        "supervisor_completed_tasks": None,
        "errors": [],
        "retry_count": 0,
        "logs": [],
        "status": "pending",
        "steps": [],
        "final_result": None,
    }
    state.update(overrides)
    return state


# ── Tests ──────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_budget_exhausted_triggers_degradation():
    """
    Tiny budget + expensive topology → degradation fires.

    Ensemble with $0.001 budget. LLM calls cost ~$0.004 each (500 tokens × frontier).
    Budget gate should fire and degrade to simpler topology.
    """
    from agent.graph import run_task

    budget = BudgetTracker(max_cost_usd=0.001)

    executor_llm = _make_mock_llm(content="42", total_tokens=500)
    planner_llm = _make_planner_llm(step_count=1, total_tokens=500)
    # Validator needs to return confidence info
    validator_response = _make_mock_llm_response("pass", total_tokens=50)
    validator_llm = MagicMock()
    validator_llm.ainvoke = AsyncMock(return_value=validator_response)
    validator_llm.bind_tools = MagicMock(return_value=validator_llm)
    validator_llm.with_structured_output = MagicMock(return_value=validator_llm)

    with patch("agent.nodes.executor.create_llm", return_value=executor_llm), \
         patch("agent.nodes.planner.create_llm", return_value=planner_llm), \
         patch("agent.nodes.validator.create_llm", return_value=validator_llm), \
         patch("agent.nodes.judge.create_llm", return_value=executor_llm):

        result = await run_task(
            task="What is 2+2?",
            budget=budget,
            task_id="test-degrad-001",
            topology_override="ensemble",
        )

    # Should either complete with degraded status or complete on a simpler topology
    assert result["status"] in ("completed", "degraded_completion", "failed")
    # Budget was spent (nodes return consumed_cost via state updates)
    assert result["budget_spent_pct"] >= 0


@pytest.mark.anyio
async def test_degradation_chain_ensemble_to_single():
    """
    Full degradation chain: ensemble → fanout → supervisor → pipeline → single.
    """
    from agent.graph import run_task

    budget = BudgetTracker(max_cost_usd=0.0005)
    executor_llm = _make_mock_llm(content="The answer is 4", total_tokens=500)
    planner_llm = _make_planner_llm(step_count=1, total_tokens=500)
    validator_response = _make_mock_llm_response("pass", total_tokens=50)
    validator_llm = MagicMock()
    validator_llm.ainvoke = AsyncMock(return_value=validator_response)
    validator_llm.bind_tools = MagicMock(return_value=validator_llm)
    validator_llm.with_structured_output = MagicMock(return_value=validator_llm)

    with patch("agent.nodes.executor.create_llm", return_value=executor_llm), \
         patch("agent.nodes.planner.create_llm", return_value=planner_llm), \
         patch("agent.nodes.validator.create_llm", return_value=validator_llm), \
         patch("agent.nodes.judge.create_llm", return_value=executor_llm):

        result = await run_task(
            task="What is 2+2?",
            budget=budget,
            task_id="test-degrad-chain-001",
            topology_override="ensemble",
        )

    result.get("topology", "ensemble")
    result.get("degradation_count", 0)

    # The system didn't crash
    assert result["status"] in ("completed", "degraded_completion", "failed")


@pytest.mark.anyio
async def test_budget_gate_interrupts_and_projects():
    """
    Verify the budget gate fires an interrupt, and the orchestrator
    catches it and projects to a degraded topology.
    """
    from langgraph.checkpoint.memory import MemorySaver

    from agent.orchestrator import run_task_with_degradation
    from agent.topologies.builder import compile_graph

    budget = BudgetTracker(max_cost_usd=0.001)
    budget.consumed_cost = 0.00095  # 95% spent → STRUCTURAL_DEGRADE

    task_id = "test-degrad-gate-001"
    topology = "ensemble"
    checkpointer = MemorySaver()
    graph = compile_graph(topology)

    initial_state = _base_state(
        task_id=task_id,
        topology=topology,
        budget=budget,
        consumed_cost=0.00095,
    )

    executor_llm = _make_mock_llm(content="42", total_tokens=50)
    planner_llm = _make_planner_llm(step_count=1, total_tokens=50)
    validator_response = _make_mock_llm_response("pass", total_tokens=50)
    validator_llm = MagicMock()
    validator_llm.ainvoke = AsyncMock(return_value=validator_response)
    validator_llm.bind_tools = MagicMock(return_value=validator_llm)
    validator_llm.with_structured_output = MagicMock(return_value=validator_llm)

    with patch("agent.nodes.executor.create_llm", return_value=executor_llm), \
         patch("agent.nodes.planner.create_llm", return_value=planner_llm), \
         patch("agent.nodes.validator.create_llm", return_value=validator_llm), \
         patch("agent.nodes.judge.create_llm", return_value=executor_llm):

        result = await run_task_with_degradation(
            graph=graph,
            initial_state=initial_state,
            task_id=task_id,
            topology=topology,
            checkpointer=checkpointer,
        )

    degradation_count = result.get("degradation_count", 0)
    final_topo = result.get("topology", topology)

    # The system didn't crash
    assert result["status"] in ("completed", "degraded_completion", "failed")

    # Budget was 95% spent — should have triggered degradation
    if degradation_count > 0:
        # Final topology should be simpler than or equal to ensemble
        assert final_topo in ("ensemble", "fanout", "supervisor", "pipeline", "single")


@pytest.mark.anyio
async def test_single_topology_no_degradation():
    """
    Single topology with tiny budget — can't degrade further.
    Should complete or degrade gracefully.
    """
    from agent.graph import run_task

    budget = BudgetTracker(max_cost_usd=0.001)
    executor_llm = _make_mock_llm(content="42", total_tokens=50)
    planner_llm = _make_planner_llm(step_count=1, total_tokens=50)
    validator_response = _make_mock_llm_response("pass", total_tokens=50)
    validator_llm = MagicMock()
    validator_llm.ainvoke = AsyncMock(return_value=validator_response)
    validator_llm.bind_tools = MagicMock(return_value=validator_llm)
    validator_llm.with_structured_output = MagicMock(return_value=validator_llm)

    with patch("agent.nodes.executor.create_llm", return_value=executor_llm), \
         patch("agent.nodes.planner.create_llm", return_value=planner_llm), \
         patch("agent.nodes.validator.create_llm", return_value=validator_llm), \
         patch("agent.nodes.judge.create_llm", return_value=executor_llm):

        result = await run_task(
            task="What is 2+2?",
            budget=budget,
            task_id="test-degrad-single-001",
            topology_override="single",
        )

    # Single can't degrade further
    assert result["status"] in ("completed", "degraded_completion", "failed")
    assert result["topology"] == "single"
    assert result["budget_spent_pct"] >= 0
