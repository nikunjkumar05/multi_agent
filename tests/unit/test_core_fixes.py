import pytest
from agent.topologies.supervisor import build_supervisor_graph
from agent.topologies.pipeline import _should_continue
from agent.topologies.single import _route_after_validation
from agent.tools.registry import registry
from core.llm import estimate_tokens, estimate_cost
from agent.state import AgentState
from core.budget import BudgetTracker


def test_supervisor_topology_compiles():
    # Verify build_supervisor_graph builds a graph that compiles successfully
    builder = build_supervisor_graph()
    graph = builder.compile()
    assert graph is not None


def test_estimate_tokens_dict_aware():
    # Mock modern LangChain usage_metadata format
    class MockResponse:
        def __init__(self, usage_metadata):
            self.usage_metadata = usage_metadata
            self.content = "mock"

    res_dict = MockResponse({"total_tokens": 123})
    assert estimate_tokens(res_dict) == 123

    res_object = MockResponse(type('Usage', (object,), {"total_tokens": 456})())
    assert estimate_tokens(res_object) == 456


def test_pipeline_should_continue_terminal_failure():
    # If there are errors and retry_count >= MAX_RETRIES, and all steps done, should return "judge"
    state: AgentState = {
        "errors": ["Step failed"],
        "retry_count": 2, # MAX_RETRIES is 2
        "current_step_index": 0,
        "steps": [{"step_id": 1, "description": "Step 1", "status": "failed", "result": None, "error": "error"}],
        "step_results": {1: "some result"},  # step completed
    }
    assert _should_continue(state) == "judge"


def test_single_route_after_validation_terminal_failure():
    # If there are errors and retry_count >= MAX_RETRIES, and all steps done, should return "escalation"
    state: AgentState = {
        "errors": ["Step failed"],
        "retry_count": 2, # MAX_RETRIES is 2
        "current_step_index": 0,
        "steps": [{"step_id": 1, "description": "Step 1", "status": "failed", "result": None, "error": "error"}],
        "step_results": {1: "some result"},  # step completed
        "budget": BudgetTracker(max_cost_usd=1.0, consumed_cost=0.0),
    }
    result = _route_after_validation(state)
    assert result == "escalation"


def test_default_tool_registration():
    # Ensure default tools are present and registered in registry
    names = registry.list_names()
    assert "code_executor" in names
    assert "db_query" in names
    assert "file_read" in names
    assert "file_write" in names
    assert "file_list" in names
    assert "web_search" in names
