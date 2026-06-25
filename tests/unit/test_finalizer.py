import pytest
from agent.nodes.finalizer import finalize_result, _is_subset
from agent.state import AgentState


def _make_state(**overrides) -> AgentState:
    defaults: dict = {
        "task": "",
        "step_results": {},
        "status": "executing",
        "logs": [],
        "errors": [],
        "final_result": "",
        "current_step_index": 0,
        "steps": [],
    }
    defaults.update(overrides)
    return defaults  # type: ignore[return-value]


class TestIsSubset:
    def test_identical_strings(self):
        assert _is_subset("hello", "hello") is True

    def test_subset_left_in_right(self):
        assert _is_subset("hello", "hello world") is True

    def test_subset_right_in_left(self):
        assert _is_subset("hello world", "hello") is True

    def test_no_subset(self):
        assert _is_subset("foo", "bar") is False

    def test_case_insensitive(self):
        assert _is_subset("Hello", "hello world") is True


class TestFinalizeResult:
    def test_single_result(self):
        state = _make_state(step_results={"1": "result one"})
        result = finalize_result(state)
        assert result["status"] == "completed"
        assert result["final_result"] == "result one"

    def test_multiple_unique_results(self):
        state = _make_state(step_results={"1": "first result", "2": "second result"})
        result = finalize_result(state)
        assert result["status"] == "completed"
        assert "first result" in result["final_result"]
        assert "second result" in result["final_result"]

    def test_dedup_subset_results(self):
        state = _make_state(step_results={"1": "short", "2": "short and longer text"})
        result = finalize_result(state)
        assert result["status"] == "completed"
        assert result["final_result"] == "short and longer text"

    def test_empty_step_results(self):
        state = _make_state()
        result = finalize_result(state)
        assert result["final_result"] == ""

    def test_errors_mark_failed(self):
        state = _make_state(
            step_results={"1": "some result"},
            errors=["something went wrong"],
        )
        result = finalize_result(state)
        assert result["status"] == "failed"
        assert result["final_result"] == "some result"

    def test_no_step_results_with_errors(self):
        state = _make_state(errors=["error"])
        result = finalize_result(state)
        assert result["status"] == "failed"
        assert result["final_result"] == ""
