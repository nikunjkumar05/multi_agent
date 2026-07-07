"""
Mid-execution budget gate with interrupt mechanism.

This node fires at synchronization barriers (after each LLM-invoking node).
It reads the live BudgetTracker and decides whether to:
- CONTINUE: budget OK
- PAUSE: interrupt → project → rebuild → resume
- SKIP_JUDGE: critical on single, skip judge
- EMERGENCY_SINGLE: critical on other, interrupt → collapse to single

Uses langgraph.types.interrupt() to pause the graph.
"""

from __future__ import annotations

from enum import Enum

from core.audit import get_audit_trail
from core.budget import BudgetBand, BudgetTracker
from core.node_events import emit_event


class BudgetGateAction(str, Enum):
    CONTINUE = "continue"
    PAUSE = "pause"
    SKIP_JUDGE = "skip_judge"
    EMERGENCY_SINGLE = "emergency_single"


def evaluate_gate(state: dict) -> BudgetGateAction:
    """
    Evaluate budget state and return the required action.
    Pure function, no side effects.
    """
    budget: BudgetTracker | None = state.get("budget")
    if budget is None:
        return BudgetGateAction.CONTINUE

    band = budget.get_band()
    topology = state.get("topology", "single")

    if band in (BudgetBand.HEALTHY, BudgetBand.TIER_DOWNGRADE):
        return BudgetGateAction.CONTINUE

    if band == BudgetBand.STRUCTURAL_DEGRADE:
        if topology == "single":
            return BudgetGateAction.CONTINUE
        return BudgetGateAction.PAUSE

    if band == BudgetBand.CRITICAL:
        if topology == "single":
            return BudgetGateAction.SKIP_JUDGE
        return BudgetGateAction.EMERGENCY_SINGLE

    return BudgetGateAction.CONTINUE


async def budget_gate_node(state: dict) -> dict:
    """
    LangGraph node: evaluate budget gate and interrupt if needed.
    Fires at synchronization barriers after LLM-invoking nodes.
    """
    from langgraph.types import interrupt

    action = evaluate_gate(state)

    if action == BudgetGateAction.CONTINUE:
        return {}

    task_id = state.get("task_id", "")
    topology = state.get("topology", "unknown")
    budget: BudgetTracker | None = state.get("budget")
    band = budget.get_band().value if budget else "unknown"

    if action == BudgetGateAction.PAUSE:
        await emit_event(
            task_id,
            "budget_gate_pause",
            {
                "band": band,
                "topology": topology,
                "message": f"Budget gate: {band} on {topology} → interrupt for degradation",
            },
        )
        audit = get_audit_trail()
        audit.record_budget_band(
            task_id=task_id,
            band=band,
            remaining_budget=budget.remaining_pct if budget else 0,
            action=f"Budget gate PAUSE: interrupting {topology} for degradation",
        )
        interrupt({"reason": "pause", "band": band, "topology": topology})
        return {}  # unreachable — interrupt() raises, but clarifies intent

    elif action == BudgetGateAction.EMERGENCY_SINGLE:
        await emit_event(
            task_id,
            "budget_gate_emergency",
            {
                "band": band,
                "topology": topology,
                "message": f"Budget gate: {band} on {topology} → emergency collapse to single",
            },
        )
        audit = get_audit_trail()
        audit.record_budget_band(
            task_id=task_id,
            band=band,
            remaining_budget=budget.remaining_pct if budget else 0,
            action=f"Budget gate EMERGENCY: collapsing {topology} to single",
        )
        interrupt({"reason": "emergency_single", "band": band, "topology": topology})
        return {}  # unreachable — interrupt() raises, but clarifies intent

    elif action == BudgetGateAction.SKIP_JUDGE:
        await emit_event(
            task_id,
            "budget_gate_skip_judge",
            {
                "band": band,
                "topology": topology,
                "message": f"Budget gate: {band} on single → skipping judge",
            },
        )
        return {"skip_judge": True}

    return {}