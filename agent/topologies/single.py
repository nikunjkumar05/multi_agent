from agent.nodes.budget_gate import budget_gate_node
from agent.nodes.escalation import check_escalation
from agent.nodes.executor import execute_step
from agent.nodes.finalizer import finalize_result
from agent.nodes.judge import ensemble_judge
from agent.nodes.planner import plan_task
from agent.nodes.validator import validate_result
from agent.state import AgentState
from langgraph.graph import END, START, StateGraph

MAX_RETRIES = 2


def _route_after_validation(state: AgentState) -> str:
    # Budget gate said skip judge — go straight to finalizer
    if state.get("skip_judge"):
        return "finalizer"

    errors = state.get("errors", [])
    retry_count = state.get("retry_count", 0)
    step_results = state.get("step_results", {})
    steps = state.get("steps") or state.get("plan_steps") or []

    if errors and retry_count < MAX_RETRIES:
        return "executor"

    if steps:
        completed = len([r for r in step_results.values() if r is not None])
        if completed < len(steps):
            return "executor"

    return "escalation"


def build_single_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("planner", plan_task)
    builder.add_node("executor", execute_step)
    builder.add_node("budget_gate", budget_gate_node)
    builder.add_node("validator", validate_result)
    builder.add_node("escalation", check_escalation)
    builder.add_node("judge", ensemble_judge)
    builder.add_node("finalizer", finalize_result)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "budget_gate")
    builder.add_edge("budget_gate", "validator")
    builder.add_conditional_edges(
        "validator",
        _route_after_validation,
        {
            "executor": "executor",
            "finalizer": "finalizer",
            "escalation": "escalation",
        },
    )
    builder.add_conditional_edges(
        "escalation",
        lambda s: "judge" if s.get("escalation_triggered") else "finalizer",
        {
            "judge": "judge",
            "finalizer": "finalizer",
        },
    )
    builder.add_edge("judge", "finalizer")
    builder.add_edge("finalizer", END)

    return builder