from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agent.nodes.budget_gate import budget_gate_node
from agent.nodes.entry_router import entry_router_node
from agent.nodes.executor import execute_step
from agent.nodes.finalizer import finalize_result
from agent.nodes.judge import ensemble_judge
from agent.nodes.planner import plan_task
from agent.nodes.validator import validate_result
from agent.state import AgentState
from core.llm import create_llm, estimate_cost, estimate_tokens
from core.node_events import emit_event

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

    # Pre-LLM budget check — skip supervisor if budget exhausted
    from core.budget import should_skip_llm
    if should_skip_llm(state):
        budget = state.get("budget")
        spent_pct = round(state.get("consumed_cost", 0.0) / budget.max_cost_usd * 100, 1) if budget and budget.max_cost_usd > 0 else 0
        task_id = state.get("task_id", "")
        await emit_event(task_id, "supervisor_skipped", {
            "reason": "budget_exhausted",
            "spent_pct": spent_pct,
            "pending_steps": len(pending),
        })
        return {"status": "completed", "logs": [f"Supervisor skipped - budget exhausted ({spent_pct}% spent), {len(pending)} steps abandoned"]}

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

    sup_tokens = estimate_tokens(response)
    sup_cost = estimate_cost(response, tier)

    task_id = state.get("task_id", "")
    budget = state.get("budget")
    prev_tokens = state.get("consumed_tokens", 0)
    prev_cost = state.get("consumed_cost", 0.0)
    acc_tokens = prev_tokens + sup_tokens
    acc_cost = prev_cost + sup_cost
    await emit_event(task_id, "supervisor_decided", {
        "next_step_id": content,
        "tokens_used": acc_tokens,
        "cost_usd": round(acc_cost, 6),
        "budget_spent_pct": round(acc_cost / budget.max_cost_usd * 100, 1) if budget and budget.max_cost_usd > 0 else 0,
    })

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
    return {"current_step_index": target_idx, "status": "executing", "consumed_tokens": sup_tokens, "consumed_cost": sup_cost}


def _route_after_supervisor(state: AgentState) -> Literal["executor", "judge", "finalizer"]:
    if state.get("skip_judge"):
        return "finalizer"
    steps = state.get("steps", [])
    pending = [s for s in steps if s["status"] == "pending"]
    if not pending:
        return "judge"
    return "executor"


def build_supervisor_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("entry_router", entry_router_node)
    builder.add_node("planner", plan_task)
    builder.add_node("budget_gate_post_planner", budget_gate_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("budget_gate_post_supervisor", budget_gate_node)
    builder.add_node("executor", execute_step)
    builder.add_node("budget_gate", budget_gate_node)
    builder.add_node("validator", validate_result)
    builder.add_node("judge", ensemble_judge)
    builder.add_node("budget_gate_post_judge", budget_gate_node)
    builder.add_node("finalizer", finalize_result)

    builder.add_edge(START, "entry_router")
    builder.add_edge("entry_router", "planner")
    builder.add_edge("planner", "budget_gate_post_planner")
    builder.add_edge("budget_gate_post_planner", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {"executor": "budget_gate_post_supervisor", "judge": "judge", "finalizer": "finalizer"},
    )
    builder.add_edge("budget_gate_post_supervisor", "executor")
    builder.add_edge("executor", "budget_gate")
    builder.add_edge("budget_gate", "validator")
    builder.add_edge("validator", "supervisor")
    builder.add_edge("judge", "budget_gate_post_judge")
    builder.add_edge("budget_gate_post_judge", "finalizer")
    builder.add_edge("finalizer", END)

    return builder