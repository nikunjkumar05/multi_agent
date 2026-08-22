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
from core.budget import BudgetTracker, next_topology
from core.node_events import emit_event

HARD_CAP_MULTIPLIER = 1.00  # Circuit breaker triggers at 100% of budget


class BudgetGateAction(str, Enum):
    CONTINUE = "continue"
    PAUSE = "pause"
    SKIP_JUDGE = "skip_judge"
    EMERGENCY_SINGLE = "emergency_single"


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
    Syncs BudgetTracker with live consumed_cost/consumed_tokens so that
    band detection works correctly across all consumers.
    Includes hard circuit breaker at 100% of budget.
    """
    from langgraph.types import interrupt

    # Sync BudgetTracker with live state values (fixes stale band detection)
    budget: BudgetTracker | None = state.get("budget")
    acc_cost = state.get("consumed_cost", 0.0)
    acc_tokens = state.get("consumed_tokens", 0)
    if budget is not None:
        budget.consumed_cost = acc_cost
        budget.consumed_tokens = acc_tokens

    # Hard circuit breaker: force stop if cost exceeds 100% of budget
    if budget and budget.max_cost_usd > 0 and acc_cost >= budget.max_cost_usd * HARD_CAP_MULTIPLIER:
        task_id = state.get("task_id", "")
        spent_pct = round(acc_cost / budget.max_cost_usd * 100, 1)
        await emit_event(task_id, "budget_circuit_breaker", {
            "consumed_cost": round(acc_cost, 6),
            "hard_cap": round(budget.max_cost_usd * HARD_CAP_MULTIPLIER, 6),
            "spent_pct": spent_pct,
            "message": f"Circuit breaker: {spent_pct}% spent exceeds {HARD_CAP_MULTIPLIER*100:.0f}% hard cap — forcing emergency stop",
        })
        audit = get_audit_trail()
        audit.record_budget_band(
            task_id=task_id,
            band="critical",
            remaining_budget=0,
            action=f"CIRCUIT BREAKER: cost ${acc_cost:.6f} exceeds hard cap ${budget.max_cost_usd * HARD_CAP_MULTIPLIER:.6f} — forcing stop",
        )
        interrupt({
            "reason": "circuit_breaker",
            "consumed_cost": acc_cost,
            "hard_cap": budget.max_cost_usd * HARD_CAP_MULTIPLIER,
        })
        return {"budget": budget}  # unreachable — interrupt() raises

    action = evaluate_gate(state)

    if action == BudgetGateAction.CONTINUE:
        return {"budget": budget}


    task_id = state.get("task_id", "")
    topology = state.get("topology", "single")
    spent_pct = (acc_cost / budget.max_cost_usd * 100) if budget and budget.max_cost_usd > 0 else 0.0
    band = budget.get_band().value

    if action == BudgetGateAction.PAUSE:
        to_topology = next_topology(topology)
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
        return {"skip_judge": True, "budget": budget}

    return {"budget": budget}