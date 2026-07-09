"""Unit tests for agent/nodes/entry_router.py — cold start, resume, routing."""

from agent.nodes.entry_router import entry_router_node, route_next_step


class TestEntryRouterNode:
    def test_cold_start(self):
        """When state is None, routes to planner."""
        result = entry_router_node(None)
        assert result == {"next_node": "planner"}

    def test_skips_completed(self):
        """When completed_step_ids populated, skips to next pending step."""
        state = {
            "completed_step_ids": [1, 2],
            "current_step_index": 0,
        }
        result = entry_router_node(state)
        assert result == {"current_step_index": 3}

    def test_no_completed_steps(self):
        """When no completed steps, returns empty dict (normal flow)."""
        state = {
            "completed_step_ids": [],
            "current_step_index": 0,
        }
        result = entry_router_node(state)
        assert result == {}

    def test_single_completed_step(self):
        """With one completed step, advances to index 2."""
        state = {
            "completed_step_ids": [1],
            "current_step_index": 0,
        }
        result = entry_router_node(state)
        assert result == {"current_step_index": 2}


class TestRouteNextStep:
    def test_no_completed_routes_to_planner(self):
        """With no completed steps, routes to planner."""
        state = {
            "completed_step_ids": [],
            "plan_steps": [{"step_id": 1}, {"step_id": 2}],
        }
        assert route_next_step(state) == "planner"

    def test_all_done_routes_to_judge(self):
        """When all steps complete, routes to judge."""
        state = {
            "completed_step_ids": [1, 2],
            "plan_steps": [{"step_id": 1}, {"step_id": 2}],
            "skip_judge": False,
        }
        assert route_next_step(state) == "judge"

    def test_all_done_skip_judge_routes_to_finalizer(self):
        """When all steps complete and skip_judge=True, routes to finalizer."""
        state = {
            "completed_step_ids": [1, 2],
            "plan_steps": [{"step_id": 1}, {"step_id": 2}],
            "skip_judge": True,
        }
        assert route_next_step(state) == "finalizer"

    def test_next_step_execution(self):
        """When next step is execution type, routes to executor."""
        state = {
            "completed_step_ids": [1],
            "plan_steps": [
                {"step_id": 1, "type": "execution"},
                {"step_id": 2, "type": "execution"},
                {"step_id": 3, "type": "execution"},
            ],
            "skip_judge": False,
        }
        assert route_next_step(state) == "executor"

    def test_next_step_validation(self):
        """When next step is validation type, routes to validator."""
        state = {
            "completed_step_ids": [1],
            "plan_steps": [
                {"step_id": 1, "type": "execution"},
                {"step_id": 2, "type": "execution"},
                {"step_id": 3, "type": "validation"},
            ],
            "skip_judge": False,
        }
        assert route_next_step(state) == "validator"

    def test_next_step_judgment(self):
        """When next step is judgment type, routes to judge."""
        state = {
            "completed_step_ids": [1],
            "plan_steps": [
                {"step_id": 1, "type": "execution"},
                {"step_id": 2, "type": "execution"},
                {"step_id": 3, "type": "judgment"},
            ],
            "skip_judge": False,
        }
        assert route_next_step(state) == "judge"

    def test_next_step_judgment_skip_judge(self):
        """When next step is judgment but skip_judge=True, routes to finalizer."""
        state = {
            "completed_step_ids": [1],
            "plan_steps": [
                {"step_id": 1, "type": "execution"},
                {"step_id": 2, "type": "execution"},
                {"step_id": 3, "type": "judgment"},
            ],
            "skip_judge": True,
        }
        assert route_next_step(state) == "finalizer"

    def test_unknown_type_defaults_to_executor(self):
        """Unknown step type defaults to executor."""
        state = {
            "completed_step_ids": [1],
            "plan_steps": [
                {"step_id": 1, "type": "execution"},
                {"step_id": 2, "type": "execution"},
                {"step_id": 3, "type": "unknown_type"},
            ],
            "skip_judge": False,
        }
        assert route_next_step(state) == "executor"
