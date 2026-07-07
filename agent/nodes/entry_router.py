"""
Entry router node — handles cold start and resume after degradation.

Every topology graph's first node is entry_router. It handles two cases:
1. Cold start: state is None → route to planner
2. Resume after degradation: completed_step_ids populated → skip to next pending step
"""

from __future__ import annotations

from typing import Any


def entry_router_node(state: dict[str, Any] | None) -> dict[str, Any]:
    """
    LangGraph node: route to planner on cold start,
    or skip to next pending step on resume.
    """
    if state is None:
        return {"next_node": "planner"}

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