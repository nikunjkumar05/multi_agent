"""
Mid-execution budget checkpoint — a lightweight LangGraph node that is
injected between the Executor and Validator in every topology.

What it does
------------
* Reads the current BudgetBand from the BudgetTracker stored in state.
* Compares it against `last_budget_band` (the band recorded at the previous
  checkpoint).
* If the band has worsened, emits a "budget_degradation" event so the
  real-time WebSocket stream reflects the change, and records the crossing
  in the audit trail.
* Updates `last_budget_band` in state so the next checkpoint has a baseline.

What it does NOT do
-------------------
* It cannot change the compiled graph topology (graphs are static once
  compiled).  Structural degradation still happens pre-execution via
  core/degrader.py.
* It does NOT call any LLM.  Latency overhead is negligible (<1 ms).

The nodes downstream (validator, judge) already call BudgetTracker methods
(can_afford_tier, should_skip_judge) at runtime, so model-tier enforcement
is automatic once the BudgetTracker state changes.
"""

from __future__ import annotations

from core.audit import get_audit_trail
from core.budget import BudgetBand
from core.node_events import emit_event

# Numeric ordering so we can compare band severity
_BAND_SEVERITY: dict[str, int] = {
    BudgetBand.HEALTHY.value: 0,
    BudgetBand.TIER_DOWNGRADE.value: 1,
    BudgetBand.STRUCTURAL_DEGRADE.value: 2,
    BudgetBand.CRITICAL.value: 3,
}


async def budget_checkpoint(state: dict) -> dict:
    """
    LangGraph node: reads state, emits events if budget band has worsened,
    returns a partial state update with the new `last_budget_band`.
    """
    budget = state.get("budget")
    task_id = state.get("task_id", "")
    topology = state.get(
        "topology",
        state.get("decision", {}).topology if hasattr(state.get("decision", None), "topology") else "unknown",
    )

    if budget is None:
        return {}

    current_band: BudgetBand = budget.get_band()
    previous_band_value: str = state.get("last_budget_band") or BudgetBand.HEALTHY.value
    current_band_value: str = current_band.value

    previous_severity = _BAND_SEVERITY.get(previous_band_value, 0)
    current_severity = _BAND_SEVERITY.get(current_band_value, 0)

    if current_severity > previous_severity:
        # Band has worsened — emit event + audit
        await emit_event(
            task_id,
            "budget_band_crossed",
            {
                "from_band": previous_band_value,
                "to_band": current_band_value,
                "spent_pct": round(budget.spent_pct, 1),
                "topology": topology,
                "message": (
                    f"Budget band escalated from {previous_band_value} "
                    f"to {current_band_value} ({budget.spent_pct:.1f}% spent)"
                ),
            },
        )

        audit = get_audit_trail()
        audit.record_budget_band(
            task_id=task_id,
            band=current_band_value,
            remaining_budget=budget.remaining_pct,
            action=(
                f"Mid-execution band crossing: {previous_band_value} → {current_band_value}. "
                f"Downstream nodes will use allowed tiers: {budget.get_allowed_tiers()}."
            ),
        )

    return {"last_budget_band": current_band_value}
