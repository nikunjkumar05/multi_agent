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

TOPOLOGY_DEGRADATION_CHAIN = ["ensemble", "fanout", "supervisor", "pipeline", "single"]


class BudgetGateAction(str, Enum):
    CONTINUE = "continue"
    PAUSE = "pause"
    SKIP_JUDGE = "skip_judge"
    EMERGENCY_SINGLE = "emergency_single"


def _next_topology(current: str) -> str:
    """Get the next topology in the degradation chain."""
    try:
        idx = TOPOLOGY_DEGRADATION_CHAIN.index(current)
        return TOPOLOGY_DEGRADATION_CHAIN[min(idx + 1, len(TOPOLOGY_DEGRADATION_CHAIN) - 1)]
    except ValueError:
        return "single"


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
    topology = state.get("topology", "single")
    budget: BudgetTracker | None = state.get("budget")
    band = budget.get_band().value if budget else "unknown"

    if action == BudgetGateAction.PAUSE:
        to_topology = _next_topology(topology)
        await emit_event(
            task_id,
            "budget_gate_pause",
            {
                "band": band,
                "from_topology": topology,
                "to_topology": to_topology,
                "message": f"Budget gate: {band} on {topology} → interrupt for degradation to {to_topology}",
                "consumed_tokens": budget.consumed_tokens if budget else 0,
                "consumed_cost": round(budget.consumed_cost, 6) if budget else 0,
                "spent_pct": round(budget.spent_pct, 1) if budget else 0,
            },
        )
        audit = get_audit_trail()
        audit.record_budget_band(
            task_id=task_id,
            band=band,
            remaining_budget=budget.remaining_pct if budget else 0,
            action=f"Budget gate PAUSE: interrupting {topology} for degradation to {to_topology}",
        )
        interrupt({
            "reason": "pause",
            "band": band,
            "from_topology": topology,
            "to_topology": to_topology,
        })
        return {}  # unreachable — interrupt() raises, but clarifies intent

    elif action == BudgetGateAction.EMERGENCY_SINGLE:
        await emit_event(
            task_id,
            "budget_gate_emergency",
            {
                "band": band,
                "from_topology": topology,
                "to_topology": "single",
                "message": f"Budget gate: {band} on {topology} → emergency collapse to single",
                "consumed_tokens": budget.consumed_tokens if budget else 0,
                "consumed_cost": round(budget.consumed_cost, 6) if budget else 0,
                "spent_pct": round(budget.spent_pct, 1) if budget else 0,
            },
        )
        audit = get_audit_trail()
        audit.record_budget_band(
            task_id=task_id,
            band=band,
            remaining_budget=budget.remaining_pct if budget else 0,
            action=f"Budget gate EMERGENCY: collapsing {topology} to single",
        )
        interrupt({
            "reason": "emergency_single",
            "band": band,
            "from_topology": topology,
            "to_topology": "single",
        })
        return {}  # unreachable — interrupt() raises, but clarifies intent

    elif action == BudgetGateAction.SKIP_JUDGE:
        await emit_event(
            task_id,
            "budget_gate_skip_judge",
            {
                "band": band,
                "topology": topology,
                "message": f"Budget gate: {band} on single → skipping judge",
                "consumed_tokens": budget.consumed_tokens if budget else 0,
                "consumed_cost": round(budget.consumed_cost, 6) if budget else 0,
                "spent_pct": round(budget.spent_pct, 1) if budget else 0,
            },
        )
        return {"skip_judge": True}

    return {}