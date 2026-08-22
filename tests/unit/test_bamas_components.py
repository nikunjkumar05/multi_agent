"""Tests for BAMAS mid-execution components: projections, budget gate, entry router."""

import pytest

from agent.nodes.budget_gate import BudgetGateAction, evaluate_gate
from agent.nodes.entry_router import entry_router_node, route_next_step
from core.budget import BudgetTracker
from core.projections import (
    _select_best_candidate,
    get_valid_projection_edges,
    project_ensemble_to_fanout,
    project_ensemble_to_single,
    project_fanout_to_single,
    project_fanout_to_supervisor,
    project_pipeline_to_single,
    project_state,
    project_supervisor_to_single,
)

# ── Projection Tests ──────────────────────────────────────────────────

class TestProjections:
    def test_project_ensemble_to_fanout(self):
        state = {
            "topology": "ensemble",
            "candidate_outputs": {"a": {"output": "test"}},
            "agent_a_result": {"output": "a"},
            "agent_b_result": {"output": "b"},
            "agent_c_result": {"output": "c"},
            "step_results": {1: "done"},
        }
        result = project_ensemble_to_fanout(state)
        assert result["topology"] == "fanout"
        assert result["agent_a_result"] is None
        assert result["agent_b_result"] is None
        assert result["agent_c_result"] is None

    def test_project_ensemble_to_single_preserves_best(self):
        state = {
            "topology": "ensemble",
            "candidate_outputs": {
                "a": {"output": "best output here", "confidence": 0.9, "tool_calls_count": 2, "tool_errors_count": 0},
                "b": {"output": "ok", "confidence": 0.5, "tool_calls_count": 1, "tool_errors_count": 1},
            },
        }
        result = project_ensemble_to_single(state)
        assert result["topology"] == "single"
        assert result["prior_context"] is not None
        assert "best output here" in result["prior_context"]

    def test_project_fanout_to_supervisor_collapses_queue(self):
        state = {
            "topology": "fanout",
            "step_results": {1: "done", 2: "done"},
            "plan_steps": [
                {"step_id": 1, "description": "Step 1", "status": "completed"},
                {"step_id": 2, "description": "Step 2", "status": "completed"},
                {"step_id": 3, "description": "Step 3", "status": "pending"},
            ],
        }
        result = project_fanout_to_supervisor(state)
        assert result["topology"] == "supervisor"
        assert result["supervisor_remaining_tasks"] is not None
        assert len(result["supervisor_remaining_tasks"]) == 1
        assert result["prior_context"] is not None

    def test_project_fanout_to_single_aggregates(self):
        state = {
            "topology": "fanout",
            "step_results": {1: "result_a", 2: "result_b"},
        }
        result = project_fanout_to_single(state)
        assert result["topology"] == "single"
        assert result["prior_context"] is not None
        assert "result_a" in result["prior_context"]

    def test_project_supervisor_to_single(self):
        state = {
            "topology": "supervisor",
            "supervisor_remaining_tasks": ["task1", "task2"],
            "supervisor_completed_tasks": [{"step_id": 1, "description": "done"}],
        }
        result = project_supervisor_to_single(state)
        assert result["topology"] == "single"
        assert result["prior_context"] is not None

    def test_project_pipeline_to_single(self):
        state = {"topology": "pipeline"}
        result = project_pipeline_to_single(state)
        assert result["topology"] == "single"

    def test_project_state_excludes_topology_history(self):
        """topology_history is annotated with operator.add — preserved by checkpointer, not projection."""
        state = {"topology": "ensemble", "candidate_outputs": {}}
        result = project_state(state, "ensemble", "single")
        assert "topology_history" not in result

    def test_project_state_raises_on_invalid_edge(self):
        state = {"topology": "single"}
        with pytest.raises(ValueError, match="Unknown projection edge"):
            project_state(state, "single", "ensemble")

    def test_valid_projection_edges(self):
        edges = get_valid_projection_edges()
        assert ("ensemble", "fanout") in edges
        assert ("fanout", "single") in edges
        assert ("pipeline", "single") in edges
        assert len(edges) == 15


class TestSelectBestCandidate:
    def test_returns_none_for_empty(self):
        assert _select_best_candidate({}) is None

    def test_returns_none_for_non_dict_values(self):
        assert _select_best_candidate({"a": "not a dict"}) is None

    def test_picks_highest_score(self):
        candidates = {
            "a": {"output": "short", "confidence": 0.5, "tool_calls_count": 0, "tool_errors_count": 0},
            "b": {"output": "much longer output with more content that exceeds threshold", "confidence": 0.9, "tool_calls_count": 3, "tool_errors_count": 0},
        }
        best = _select_best_candidate(candidates)
        assert best is not None
        assert best["confidence"] == 0.9

    def test_penalizes_errors(self):
        candidates = {
            "a": {"output": "good output", "confidence": 0.8, "tool_calls_count": 4, "tool_errors_count": 3},
            "b": {"output": "good output", "confidence": 0.7, "tool_calls_count": 2, "tool_errors_count": 0},
        }
        best = _select_best_candidate(candidates)
        assert best["tool_errors_count"] == 0


# ── Budget Gate Tests ─────────────────────────────────────────────────

class TestBudgetGate:
    def test_healthy_returns_continue(self):
        budget = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.5)
        state = {"budget": budget, "topology": "single"}
        assert evaluate_gate(state) == BudgetGateAction.CONTINUE

    def test_tier_downgrade_returns_continue(self):
        budget = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.75)
        state = {"budget": budget, "topology": "ensemble"}
        assert evaluate_gate(state) == BudgetGateAction.CONTINUE

    def test_structural_degrade_returns_pause(self):
        budget = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.95)
        state = {"budget": budget, "topology": "ensemble"}
        assert evaluate_gate(state) == BudgetGateAction.PAUSE

    def test_structural_degrade_on_single_returns_continue(self):
        budget = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.95)
        state = {"budget": budget, "topology": "single"}
        assert evaluate_gate(state) == BudgetGateAction.CONTINUE

    def test_critical_returns_emergency_single(self):
        budget = BudgetTracker(max_cost_usd=1.0, consumed_cost=1.0)
        state = {"budget": budget, "topology": "ensemble"}
        assert evaluate_gate(state) == BudgetGateAction.EMERGENCY_SINGLE

    def test_critical_on_single_returns_skip_judge(self):
        budget = BudgetTracker(max_cost_usd=1.0, consumed_cost=1.0)
        state = {"budget": budget, "topology": "single"}
        assert evaluate_gate(state) == BudgetGateAction.SKIP_JUDGE

    def test_no_budget_returns_continue(self):
        state = {"topology": "single"}
        assert evaluate_gate(state) == BudgetGateAction.CONTINUE


# ── Entry Router Tests ────────────────────────────────────────────────

class TestEntryRouter:
    def test_cold_start_returns_next_node(self):
        result = entry_router_node(None)
        assert result["next_node"] == "planner"

    def test_cold_start_with_empty_state(self):
        result = entry_router_node({})
        assert result == {}

    def test_resume_sets_next_index(self):
        state = {"completed_step_ids": [0, 1, 2]}
        result = entry_router_node(state)
        assert result["current_step_index"] == 3

    def test_route_next_step_cold_start(self):
        state = {}
        assert route_next_step(state) == "planner"

    def test_route_next_step_all_done_no_judge(self):
        state = {
            "completed_step_ids": [0, 1],
            "plan_steps": [
                {"step_id": 0, "type": "execution"},
                {"step_id": 1, "type": "execution"},
            ],
            "skip_judge": True,
        }
        assert route_next_step(state) == "finalizer"

    def test_route_next_step_all_done_with_judge(self):
        state = {
            "completed_step_ids": [0],
            "plan_steps": [{"step_id": 0, "type": "execution"}],
            "skip_judge": False,
        }
        assert route_next_step(state) == "judge"

    def test_route_next_step_to_executor(self):
        state = {
            "completed_step_ids": [0],
            "plan_steps": [
                {"step_id": 0, "type": "execution"},
                {"step_id": 1, "type": "execution"},
            ],
        }
        assert route_next_step(state) == "executor"

    def test_route_next_step_to_validator(self):
        state = {
            "completed_step_ids": [0],
            "plan_steps": [
                {"step_id": 0, "type": "execution"},
                {"step_id": 1, "type": "validation"},
            ],
        }
        assert route_next_step(state) == "validator"

    def test_route_next_step_to_judge_skip_judge(self):
        state = {
            "completed_step_ids": [0],
            "plan_steps": [
                {"step_id": 0, "type": "execution"},
                {"step_id": 1, "type": "judgment"},
            ],
            "skip_judge": True,
        }
        assert route_next_step(state) == "finalizer"


# ── Topology Integration Tests ────────────────────────────────────────

class TestTopologyIntegration:
    def test_single_graph_compiles(self):
        from agent.topologies.single import build_single_graph
        graph = build_single_graph().compile()
        assert graph is not None

    def test_pipeline_graph_compiles(self):
        from agent.topologies.pipeline import build_pipeline_graph
        graph = build_pipeline_graph().compile()
        assert graph is not None

    def test_supervisor_graph_compiles(self):
        from agent.topologies.supervisor import build_supervisor_graph
        graph = build_supervisor_graph().compile()
        assert graph is not None

    def test_fanout_graph_compiles(self):
        from agent.topologies.fanout import build_fanout_graph
        graph = build_fanout_graph().compile()
        assert graph is not None

    def test_ensemble_graph_compiles(self):
        from agent.topologies.ensemble import build_ensemble_graph
        graph = build_ensemble_graph().compile()
        assert graph is not None

    def test_builder_compiles_all_topologies(self):
        from agent.topologies.builder import compile_graph
        for topo in ["single", "pipeline", "supervisor", "fanout", "ensemble"]:
            graph = compile_graph(topo)
            assert graph is not None

    def test_builder_fallback_to_single(self):
        from agent.topologies.builder import compile_graph
        graph = compile_graph("nonexistent")
        assert graph is not None


class TestStateSchema:
    def test_merge_dicts(self):
        from agent.state import merge_dicts
        result = merge_dicts({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_merge_dicts_override(self):
        from agent.state import merge_dicts
        result = merge_dicts({"a": 1}, {"a": 2})
        assert result == {"a": 2}

    def test_merge_dicts_none(self):
        from agent.state import merge_dicts
        assert merge_dicts(None, {"a": 1}) == {"a": 1}
        assert merge_dicts({"a": 1}, None) == {"a": 1}

    def test_merge_logs(self):
        from agent.state import merge_logs
        result = merge_logs(["a"], ["b", "c"])
        assert result == ["a", "b", "c"]
