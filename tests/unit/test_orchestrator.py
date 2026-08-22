"""Tests for the orchestrator interrupt/proj/resume loop."""

import uuid
from typing import Annotated, TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import interrupt

from core.budget import next_topology as _next_topology


# ── Shared test state ──

class OrchState(TypedDict):
    task: str
    step_results: Annotated[dict, lambda a, b: {**(a or {}), **(b or {})}]
    topology: str
    final_output: str | None
    skip_judge: bool


# ── Helper nodes ──

def make_node(name: str, output: str):
    def node(state: OrchState):
        results = dict(state.get("step_results", {}))
        results[name] = output
        return {"step_results": results, "final_output": output}
    return node


def make_interrupt_node(band: str = "structural_degrade", from_topo: str = "ensemble", to_topo: str = "fanout"):
    def node(state: OrchState):
        interrupt({
            "reason": band,
            "band": band,
            "from_topology": from_topo,
            "to_topology": to_topo,
        })
        return {}
    return node


def make_skip_judge_node():
    def node(state: OrchState):
        interrupt({
            "reason": "skip_judge",
            "band": "critical",
            "from_topology": "single",
        })
        return {"skip_judge": True}
    return node


# ── Tests ──

class TestOrchestratorLoop:
    @pytest.mark.asyncio
    async def test_normal_completion(self):
        """Graph completes without interrupt — orchestrator returns result."""
        checkpointer = MemorySaver()
        builder = StateGraph(OrchState)
        builder.add_node("work", make_node("work", "hello"))
        builder.add_edge(START, "work")
        graph = builder.compile(checkpointer=checkpointer)

        from agent.orchestrator import run_task_with_degradation
        result = await run_task_with_degradation(
            graph=graph,
            initial_state={"task": "test", "step_results": {}, "topology": "single", "final_output": None, "skip_judge": False},
            task_id=str(uuid.uuid4()),
            topology="single",
            checkpointer=checkpointer,
        )

        assert result["status"] == "completed"
        assert result["final_output"] == "hello"
        assert result["degradation_count"] == 0

    @pytest.mark.asyncio
    async def test_projection_preserves_step_results(self):
        """Verify projection + update_state preserves accumulated step_results."""
        checkpointer = MemorySaver()
        task_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": task_id}}

        # Graph A: work → interrupt
        builder_a = StateGraph(OrchState)
        builder_a.add_node("work", make_node("a_work", "a_done"))
        builder_a.add_node("gate", make_interrupt_node("structural_degrade", "ensemble", "fanout"))
        builder_a.add_edge(START, "work")
        builder_a.add_edge("work", "gate")
        graph_a = builder_a.compile(checkpointer=checkpointer)

        initial_state = {"task": "test", "step_results": {}, "topology": "ensemble", "final_output": None, "skip_judge": False}
        graph_a.update_state(config, initial_state)
        result_a = await graph_a.ainvoke(None, config)

        assert "__interrupt__" in result_a
        state_a = graph_a.get_state(config).values
        assert state_a["step_results"] == {"a_work": "a_done"}

        # Project ensemble → fanout
        from core.projections import project_state
        projected = project_state(state_a, "ensemble", "fanout")
        assert projected["topology"] == "fanout"
        # Must NOT include annotated fields
        assert "step_results" not in projected

        # Graph B: work node (simulates fanout)
        builder_b = StateGraph(OrchState)
        builder_b.add_node("work", make_node("b_work", "b_done"))
        builder_b.add_edge(START, "work")
        graph_b = builder_b.compile(checkpointer=checkpointer)

        graph_b.update_state(config, projected, as_node=START)
        result_b = await graph_b.ainvoke(None, config)

        assert "__interrupt__" not in result_b
        final = graph_b.get_state(config).values
        # step_results should accumulate (merge_dicts annotated)
        assert "a_work" in final["step_results"]
        assert "b_work" in final["step_results"]
        assert final["topology"] == "fanout"

    @pytest.mark.asyncio
    async def test_degraded_completion_on_single(self):
        """CRITICAL on single → degraded_completion with best available output."""
        checkpointer = MemorySaver()
        builder = StateGraph(OrchState)
        builder.add_node("work", make_node("work", "partial_result"))
        builder.add_node("gate", make_skip_judge_node())
        builder.add_edge(START, "work")
        builder.add_edge("work", "gate")
        graph = builder.compile(checkpointer=checkpointer)

        from agent.orchestrator import run_task_with_degradation
        result = await run_task_with_degradation(
            graph=graph,
            initial_state={"task": "test", "step_results": {}, "topology": "single", "final_output": None, "skip_judge": False},
            task_id=str(uuid.uuid4()),
            topology="single",
            checkpointer=checkpointer,
        )

        assert result["status"] == "degraded_completion"
        assert result["final_output"] == "partial_result"

    @pytest.mark.asyncio
    async def test_emergency_single_projection(self):
        """EMERGENCY_SINGLE interrupt → projects to single with prior_context."""
        from core.projections import project_ensemble_to_single

        # Simulate ensemble state with candidate outputs
        state = {
            "topology": "ensemble",
            "candidate_outputs": {
                "a": {"output": "ensemble_work_a", "confidence": 0.9, "tool_calls_count": 0, "tool_errors_count": 0},
                "b": {"output": "ensemble_work_b", "confidence": 0.7, "tool_calls_count": 0, "tool_errors_count": 0},
            },
        }
        projected = project_ensemble_to_single(state)
        assert projected["topology"] == "single"
        assert projected["prior_context"] is not None
        assert "ensemble_work_a" in projected["prior_context"]  # higher confidence wins

    @pytest.mark.asyncio
    async def test_multi_step_degradation_chain(self):
        """Verify state accumulates across A → B → C degrade chain."""
        checkpointer = MemorySaver()
        task_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": task_id}}

        # Graph A
        builder_a = StateGraph(OrchState)
        builder_a.add_node("work", make_node("a_work", "a_done"))
        builder_a.add_node("gate", make_interrupt_node("structural_degrade", "ensemble", "fanout"))
        builder_a.add_edge(START, "work")
        builder_a.add_edge("work", "gate")
        graph_a = builder_a.compile(checkpointer=checkpointer)

        initial_state = {"task": "test", "step_results": {}, "topology": "ensemble", "final_output": None, "skip_judge": False}
        graph_a.update_state(config, initial_state)
        await graph_a.ainvoke(None, config)

        state_a = graph_a.get_state(config).values
        assert state_a["step_results"] == {"a_work": "a_done"}

        # Project ensemble → fanout, then fanout → supervisor
        from core.projections import project_state
        projected_fanout = project_state(state_a, "ensemble", "fanout")

        # Graph B (fanout)
        builder_b = StateGraph(OrchState)
        builder_b.add_node("work", make_node("b_work", "b_done"))
        builder_b.add_node("gate", make_interrupt_node("structural_degrade", "fanout", "supervisor"))
        builder_b.add_edge(START, "work")
        builder_b.add_edge("work", "gate")
        graph_b = builder_b.compile(checkpointer=checkpointer)

        graph_b.update_state(config, projected_fanout, as_node=START)
        await graph_b.ainvoke(None, config)

        state_b = graph_b.get_state(config).values
        assert "a_work" in state_b["step_results"]
        assert "b_work" in state_b["step_results"]

        # Project fanout → supervisor
        projected_supervisor = project_state(state_b, "fanout", "supervisor")

        # Graph C (supervisor) — simple completion
        builder_c = StateGraph(OrchState)
        builder_c.add_node("work", make_node("c_work", "c_done"))
        builder_c.add_edge(START, "work")
        graph_c = builder_c.compile(checkpointer=checkpointer)

        graph_c.update_state(config, projected_supervisor, as_node=START)
        result_c = await graph_c.ainvoke(None, config)

        final = graph_c.get_state(config).values
        assert "a_work" in final["step_results"]
        assert "b_work" in final["step_results"]
        assert "c_work" in final["step_results"]
        assert final["topology"] == "supervisor"


class TestNextTopology:
    def test_ensemble_to_fanout(self):
        assert _next_topology("ensemble") == "fanout"

    def test_fanout_to_feedback(self):
        assert _next_topology("fanout") == "feedback"

    def test_feedback_to_supervisor(self):
        assert _next_topology("feedback") == "supervisor"

    def test_supervisor_to_pipeline(self):
        assert _next_topology("supervisor") == "pipeline"

    def test_pipeline_to_single(self):
        assert _next_topology("pipeline") == "single"

    def test_single_stays_single(self):
        assert _next_topology("single") == "single"

    def test_unknown_defaults_to_single(self):
        assert _next_topology("nonexistent") == "single"


# ── Quality degradation (Gap 6) ──

class TestQualityDegradation:
    def test_no_degradation_returns_none(self):
        from agent.orchestrator import _estimate_quality_degradation
        result = _estimate_quality_degradation({"topology": "single"}, 0)
        assert result is None

    def test_single_degradation(self):
        from agent.orchestrator import _estimate_quality_degradation
        result = _estimate_quality_degradation({"topology": "single"}, 1)
        assert result is not None
        assert result["degradation_count"] == 1
        assert result["estimated_quality"] < 1.0

    def test_multiple_degradations(self):
        from agent.orchestrator import _estimate_quality_degradation
        result1 = _estimate_quality_degradation({"topology": "single"}, 1)
        result2 = _estimate_quality_degradation({"topology": "single"}, 2)
        assert result2["estimated_quality"] < result1["estimated_quality"]

    def test_topology_quality_factors(self):
        from agent.orchestrator import _estimate_quality_degradation
        for topo, factor in [("ensemble", 1.0), ("fanout", 0.8), ("single", 0.5)]:
            result = _estimate_quality_degradation({"topology": topo}, 0)
            assert result is None  # No degradation
            result = _estimate_quality_degradation({"topology": topo}, 1)
            assert result["topology_quality_factor"] == factor

    def test_quality_in_build_result(self):
        from agent.orchestrator import _build_result
        result = _build_result(
            {"status": "completed", "final_output": "test"},
            "single",
            degradation_count=1,
        )
        assert "quality_degradation" in result
        assert result["quality_degradation"] is not None
        assert "estimated_quality" in result["quality_degradation"]
