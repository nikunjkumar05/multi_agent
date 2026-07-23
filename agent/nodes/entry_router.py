"""
Entry router node — handles cold start and resume after degradation.

Every topology graph's first node is entry_router. It handles two cases:
1. Cold start: state is None → route to planner
2. Resume after degradation: completed_step_ids populated → skip to next pending step
3. Budget exhausted: skip_judge set → route directly to finalizer
"""

from __future__ import annotations

from typing import Any


def entry_router_node(state: dict[str, Any] | None) -> dict[str, Any]:
    """
    LangGraph node: route to planner on cold start,
    or skip to next pending step on resume.
    If skip_judge is set (budget exhausted), skip planner entirely.
    """
    if state is None:
        return {"next_node": "planner"}

    # Budget exhausted — skip planner, go straight to finalizer
    if state.get("skip_judge", False):
        return {"next_node": "finalizer"}

    completed = state.get("completed_step_ids", [])
    if completed:
        next_idx = max(completed) + 1
        return {"current_step_index": next_idx}

    return {}


def route_next_step(state: dict[str, Any]) -> str:
    """
    Conditional edge: which node to visit next based on completed steps.
    Returns the name of the next node to visit.
    """
    completed = state.get("completed_step_ids", [])
    plan_steps = state.get("plan_steps", [])
    skip_judge = state.get("skip_judge", False)

    # Budget exhausted — skip everything, go to finalizer
    if skip_judge:
        return "finalizer"

    # Budget check: if spent_pct >= 90%, skip to finalizer
    budget = state.get("budget")
    acc_cost = state.get("consumed_cost", 0.0)
    if budget and budget.max_cost_usd > 0:
        spent_pct = acc_cost / budget.max_cost_usd
        if spent_pct >= 0.90:
            return "finalizer"

    if not completed:
        return "planner"

    next_idx = max(completed) + 1

    if next_idx >= len(plan_steps):
        return "finalizer" if skip_judge else "judge"

    next_step = plan_steps[next_idx]
    step_type = next_step.get("type", "execution")

    if step_type == "execution":
        return "executor"
    elif step_type == "validation":
        return "validator"
    elif step_type == "judgment":
        return "judge" if not skip_judge else "finalizer"
    else:
        return "executor"