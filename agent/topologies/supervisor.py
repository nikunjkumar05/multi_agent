from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agent.nodes.executor import execute_step
from agent.nodes.judge import ensemble_judge
from agent.nodes.planner import plan_task
from agent.nodes.validator import validate_result
from agent.state import AgentState
from core.llm import create_llm

SUPERVISOR_SYSTEM = """You are a supervisor managing worker agents.
Given a list of planned steps, determine which step to assign next.
Return ONLY the step_id (integer) to execute next, or -1 if all steps are done."""

MAX_RETRIES = 2


async def supervisor_node(state: AgentState) -> dict:
    steps = state.get("steps", [])
    idx = state.get("current_step_index", 0)

    if idx >= len(steps):
        return {"status": "completed"}

    retry_count = state.get("retry_count", 0)
    errors = state.get("errors", [])
    if errors and retry_count < MAX_RETRIES:
        return {"current_step_index": idx, "status": "executing"}

    pending = [s for s in steps if s["status"] == "pending"]
    if not pending:
        return {"status": "completed"}

    tier = state["decision"].model_tiers.get("planner", "standard")
    llm = create_llm(tier)

    step_descriptions = "\n".join(
        f"Step {s['step_id']}: {s['description']} [{s['status']}]"
        for s in steps
    )

    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM),
        HumanMessage(content=f"Steps:\n{step_descriptions}\n\nCurrent index: {idx}\nWhich step_id to assign next?"),
    ]

    response = await llm.ainvoke(messages)
    content = response.content.strip() if isinstance(response.content, str) else str(response.content).strip()
    try:
        next_id = int(content)
    except ValueError:
        next_id = pending[0]["step_id"]

    if next_id == -1 or not pending:
        return {"status": "completed"}

    match = next((i for i, s in enumerate(steps) if s["step_id"] == next_id), None)
    if match is None or steps[match]["status"] != "pending":
        target_idx = next((i for i, s in enumerate(steps) if s["status"] == "pending"), idx)
    else:
        target_idx = match
    return {"current_step_index": target_idx, "status": "executing"}


def _route_after_supervisor(state: AgentState) -> Literal["executor", "end"]:
    steps = state.get("steps", [])
    pending = [s for s in steps if s["status"] == "pending"]
    if not pending:
        return "end"
    return "executor"


def build_supervisor_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("planner", plan_task)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("executor", execute_step)
    builder.add_node("validator", validate_result)
    builder.add_node("judge", ensemble_judge)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {"executor": "executor", "end": END},
    )
    builder.add_edge("executor", "validator")
    builder.add_edge("validator", "supervisor")

    return builder
