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
    Uses accumulated consumed_cost from state (not BudgetTracker, which may be stale due to serialization).
    """
    budget: BudgetTracker | None = state.get("budget")
    if budget is None:
        return BudgetGateAction.CONTINUE

    # Use accumulated cost from state nodes (BudgetTracker object may be stale due to LangGraph serialization)
    acc_cost = state.get("consumed_cost", 0.0)
    if acc_cost == 0.0 and budget:
        acc_cost = budget.consumed_cost  # Fallback for tests and non-LangGraph callers
    spent_pct = (acc_cost / budget.max_cost_usd * 100) if budget.max_cost_usd > 0 else 0.0
    topology = state.get("topology", "single")

    if spent_pct < 90:
        return BudgetGateAction.CONTINUE

    if spent_pct < 100:
        if topology == "single":
            return BudgetGateAction.CONTINUE
        return BudgetGateAction.PAUSE

    # spent_pct >= 100
    if topology == "single":
        return BudgetGateAction.SKIP_JUDGE
    return BudgetGateAction.EMERGENCY_SINGLE


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
    acc_cost = state.get("consumed_cost", 0.0)
    acc_tokens = state.get("consumed_tokens", 0)
    spent_pct = (acc_cost / budget.max_cost_usd * 100) if budget and budget.max_cost_usd > 0 else 0.0
    band = "healthy" if spent_pct < 70 else "tier_downgrade" if spent_pct < 90 else "structural_degrade" if spent_pct < 100 else "critical"

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
                "consumed_tokens": acc_tokens,
                "consumed_cost": round(acc_cost, 6),
                "spent_pct": round(spent_pct, 1),
            },
        )
        audit = get_audit_trail()
        audit.record_budget_band(
            task_id=task_id,
            band=band,
            remaining_budget=max(0, 100 - spent_pct),
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
                "consumed_tokens": acc_tokens,
                "consumed_cost": round(acc_cost, 6),
                "spent_pct": round(spent_pct, 1),
            },
        )
        audit = get_audit_trail()
        audit.record_budget_band(
            task_id=task_id,
            band=band,
            remaining_budget=max(0, 100 - spent_pct),
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
                "consumed_tokens": acc_tokens,
                "consumed_cost": round(acc_cost, 6),
                "spent_pct": round(spent_pct, 1),
            },
        )
        return {"skip_judge": True}

    return {}