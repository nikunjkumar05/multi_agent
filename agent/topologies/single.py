from langgraph.graph import END, START, StateGraph

from agent.nodes.escalation import check_escalation
from agent.nodes.executor import execute_step
from agent.nodes.finalizer import finalize_result
from agent.nodes.judge import ensemble_judge
from agent.nodes.planner import plan_task
from agent.nodes.validator import validate_result
from agent.state import AgentState

MAX_RETRIES = 2


async def _route_after_validation(state: AgentState) -> str:
    retry_count = state.get("retry_count", 0)
    errors = state.get("errors", [])

    if errors:
        if retry_count < MAX_RETRIES:
            return "executor"
        else:
            return "finalizer"

    steps = state.get("steps", [])
    idx = state.get("current_step_index", 0)
    if idx < len(steps):
        return "executor"

    escalate = await check_escalation(state)
    if escalate == "judge":
        return "judge"
    return "finalizer"


def build_single_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("planner", plan_task)
    builder.add_node("executor", execute_step)
    builder.add_node("validator", validate_result)
    builder.add_node("judge", ensemble_judge)
    builder.add_node("finalizer", finalize_result)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "validator")
    builder.add_conditional_edges("validator", _route_after_validation, {
        "executor": "executor",
        "judge": "judge",
        "finalizer": "finalizer",
    })
    builder.add_edge("judge", "finalizer")
    builder.add_edge("finalizer", END)

    return builder
