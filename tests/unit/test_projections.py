"""Unit tests for core/projections.py — state projection functions."""

import pytest

from core.projections import (
    get_valid_projection_edges,
    project_ensemble_to_fanout,
    project_ensemble_to_pipeline,
    project_ensemble_to_single,
    project_ensemble_to_supervisor,
    project_fanout_to_single,
    project_fanout_to_supervisor,
    project_pipeline_to_single,
    project_state,
    project_supervisor_to_pipeline,
    project_supervisor_to_single,
    validate_projected_state,
)

# ── Helpers ────────────────────────────────────────────────────────────

def _base_state(**overrides) -> dict:
    state = {
        "task": "test task",
        "topology": "ensemble",
        "step_results": {},
        "candidate_outputs": {},
        "plan_steps": [],
        "supervisor_remaining_tasks": None,
        "supervisor_completed_tasks": None,
        "_worker_assignments": None,
        "fanout_worker_results": None,
        "agent_a_result": None,
        "agent_b_result": None,
        "agent_c_result": None,
        "topology_history": [],
    }
    state.update(overrides)
    return state


# ── Ensemble projections ───────────────────────────────────────────────

class TestEnsembleToSingle:
    def test_picks_best_candidate(self):
        state = _base_state(
            candidate_outputs={
                "a": {"output": "ok", "confidence": 0.5, "tool_calls_count": 0, "tool_errors_count": 0},
                "b": {"output": "better", "confidence": 0.9, "tool_calls_count": 0, "tool_errors_count": 0},
                "c": {"output": "worst", "confidence": 0.3, "tool_calls_count": 0, "tool_errors_count": 0},
            }
        )
        result = project_ensemble_to_single(state)
        assert result["topology"] == "single"
        assert "better" in result["prior_context"]

    def test_no_candidates(self):
        state = _base_state(candidate_outputs={})
        result = project_ensemble_to_single(state)
        assert result["topology"] == "single"
        assert result["prior_context"] is None

    def test_partial_candidates(self):
        state = _base_state(
            candidate_outputs={
                "a": {"output": "only one", "confidence": 0.7, "tool_calls_count": 0, "tool_errors_count": 0},
            }
        )
        result = project_ensemble_to_single(state)
        assert result["topology"] == "single"
        assert "only one" in result["prior_context"]

    def test_clears_agent_results(self):
        state = _base_state()
        result = project_ensemble_to_single(state)
        assert result["agent_a_result"] is None
        assert result["agent_b_result"] is None
        assert result["agent_c_result"] is None


class TestEnsembleToFanout:
    def test_clears_agent_keys(self):
        state = _base_state()
        result = project_ensemble_to_fanout(state)
        assert result["topology"] == "fanout"
        assert result["_worker_assignments"] is None
        assert result["fanout_worker_results"] is None
        assert result["agent_a_result"] is None


class TestEnsembleToSupervisor:
    def test_chains_through_fanout(self):
        state = _base_state()
        result = project_ensemble_to_supervisor(state)
        assert result["topology"] == "supervisor"
        assert result["supervisor_remaining_tasks"] is None


class TestEnsembleToPipeline:
    def test_chains_through_fanout_supervisor(self):
        state = _base_state()
        result = project_ensemble_to_pipeline(state)
        assert result["topology"] == "pipeline"


# ── Fanout projections ────────────────────────────────────────────────

class TestFanoutToSupervisor:
    def test_queue_collapse(self):
        state = _base_state(
            step_results={1: "done", 2: None, 3: "also done"},
            plan_steps=[
                {"step_id": 1, "description": "Step 1"},
                {"step_id": 2, "description": "Step 2"},
                {"step_id": 3, "description": "Step 3"},
            ],
        )
        result = project_fanout_to_supervisor(state)
        assert result["topology"] == "supervisor"
        # Completed steps should be in supervisor_completed_tasks
        completed = result["supervisor_completed_tasks"] or []
        assert len(completed) == 2
        # Remaining tasks should be in supervisor_remaining_tasks
        remaining = result["supervisor_remaining_tasks"] or []
        assert len(remaining) == 1
        assert "Step 2" in remaining[0]

    def test_empty_workers(self):
        state = _base_state(step_results={}, plan_steps=[])
        result = project_fanout_to_supervisor(state)
        assert result["topology"] == "supervisor"
        assert result["supervisor_completed_tasks"] is None
        assert result["supervisor_remaining_tasks"] is None


class TestFanoutToSingle:
    def test_aggregates_completed(self):
        state = _base_state(step_results={1: "result1", 2: None})
        result = project_fanout_to_single(state)
        assert result["topology"] == "single"
        assert "result1" in result["prior_context"]


# ── Supervisor projections ────────────────────────────────────────────

class TestSupervisorToPipeline:
    def test_flatten(self):
        state = _base_state(
            supervisor_remaining_tasks=["task A", "task B"],
            supervisor_completed_tasks=[{"step_id": 1, "description": "done task"}],
        )
        result = project_supervisor_to_pipeline(state)
        assert result["topology"] == "pipeline"
        assert "task A" in result["prior_context"]
        assert "done task" in result["prior_context"]
        assert result["supervisor_remaining_tasks"] is None
        assert result["supervisor_completed_tasks"] is None


class TestSupervisorToSingle:
    def test_chains_through_pipeline(self):
        state = _base_state()
        result = project_supervisor_to_single(state)
        assert result["topology"] == "single"


# ── Pipeline projection ───────────────────────────────────────────────

class TestPipelineToSingle:
    def test_trivial(self):
        state = _base_state(topology="pipeline")
        result = project_pipeline_to_single(state)
        assert result["topology"] == "single"


# ── Dispatch table ────────────────────────────────────────────────────

class TestDispatchTable:
    def test_covers_all_edges(self):
        edges = get_valid_projection_edges()
        expected = [
            ("ensemble", "fanout"),
            ("ensemble", "supervisor"),
            ("ensemble", "pipeline"),
            ("ensemble", "single"),
            ("fanout", "supervisor"),
            ("fanout", "pipeline"),
            ("fanout", "single"),
            ("supervisor", "pipeline"),
            ("supervisor", "single"),
            ("pipeline", "single"),
            # Feedback topology (paper alignment)
            ("feedback", "single"),
            ("ensemble", "feedback"),
            ("fanout", "feedback"),
            ("supervisor", "feedback"),
            ("pipeline", "feedback"),
        ]
        assert sorted(edges) == sorted(expected)

    def test_unknown_edge_raises(self):
        state = _base_state()
        with pytest.raises(ValueError, match="Unknown projection edge"):
            project_state(state, "single", "ensemble")


# ── topology_history accumulation ──────────────────────────────────────

class TestHistoryAccumulation:
    def test_excludes_topology_history_from_projection(self):
        """topology_history is annotated with operator.add — preserved by checkpointer, not projection."""
        state = _base_state(
            topology_history=[{"from": "ensemble", "to": "fanout"}]
        )
        result = project_state(state, "fanout", "supervisor")
        assert "topology_history" not in result


class TestAnnotatedFieldGuard:
    """Verify that projections raise on any annotated field leak."""

    @pytest.mark.parametrize("from_t,to_t", [
        ("ensemble", "fanout"),
        ("ensemble", "supervisor"),
        ("ensemble", "pipeline"),
        ("ensemble", "single"),
        ("fanout", "supervisor"),
        ("fanout", "pipeline"),
        ("fanout", "single"),
        ("supervisor", "pipeline"),
        ("supervisor", "single"),
        ("pipeline", "single"),
    ])
    def test_projection_rejects_annotated_fields(self, from_t, to_t):
        from core.projections import _assert_no_annotated_fields
        # Pick any annotated field and inject it into a valid projection result
        fake_result = {"topology": to_t, "completed_step_ids": [1, 2]}
        with pytest.raises(ValueError, match="leaked annotated fields"):
            _assert_no_annotated_fields(fake_result, f"{from_t}→{to_t}")

    def test_all_annotated_fields_covered(self):
        from core.projections import _ANNOTATED_FIELDS
        expected = {"completed_step_ids", "step_results", "candidate_outputs",
                     "consumed_tokens", "consumed_cost", "topology_history",
                     "errors", "logs"}
        assert _ANNOTATED_FIELDS == expected


# ── State validation (Gap 2) ──────────────────────────────────────────

class TestValidateProjectedState:
    def test_valid_single_topology(self):
        projected = {"topology": "single"}
        is_valid, err = validate_projected_state(projected, "single")
        assert is_valid is True
        assert err == ""

    def test_valid_supervisor_topology(self):
        projected = {"topology": "supervisor", "supervisor_remaining_tasks": None}
        is_valid, err = validate_projected_state(projected, "supervisor")
        assert is_valid is True

    def test_valid_fanout_topology(self):
        projected = {"topology": "fanout", "_worker_assignments": None}
        is_valid, err = validate_projected_state(projected, "fanout")
        assert is_valid is True

    def test_missing_topology_field(self):
        projected = {}
        is_valid, err = validate_projected_state(projected, "single")
        assert is_valid is False
        assert "missing" in err.lower()

    def test_topology_mismatch(self):
        projected = {"topology": "fanout"}
        is_valid, err = validate_projected_state(projected, "single")
        assert is_valid is False
        assert "mismatch" in err.lower()

    def test_not_dict(self):
        is_valid, err = validate_projected_state("not a dict", "single")
        assert is_valid is False
        assert "not a dict" in err

    def test_projected_state_validated_in_safe_project(self):
        """validate_projected_state is called in _safe_project."""
        from agent.orchestrator import _safe_project
        state = _base_state()
        projected, topo = _safe_project(state, "pipeline", "single", "pipeline")
        assert projected is not None
        assert topo == "single"
