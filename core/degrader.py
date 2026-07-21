from core.budget import BudgetTracker, BudgetBand, next_topology
from core.audit import get_audit_trail


def degrade_topology(budget: BudgetTracker, current_topology: str, task_id: str) -> str:
    band = budget.get_band()

    if band == BudgetBand.HEALTHY:
        return current_topology

    if band == BudgetBand.TIER_DOWNGRADE:
        audit = get_audit_trail()
        audit.record_budget_band(
            task_id=task_id,
            band="tier_downgrade",
            remaining_budget=budget.remaining_pct,
            action="Downgrading model tiers (frontier->standard, standard->cheap). Topology unchanged.",
        )
        return current_topology

    if band == BudgetBand.STRUCTURAL_DEGRADE:
        degraded = next_topology(current_topology)

        audit = get_audit_trail()
        audit.record_budget_band(
            task_id=task_id,
            band="structural_degrade",
            remaining_budget=budget.remaining_pct,
            action=f"Collapsing topology: {current_topology} -> {degraded}",
        )
        audit.record_degradation(
            task_id=task_id,
            from_topology=current_topology,
            to_topology=degraded,
            reason=f"Budget at {budget.spent_pct:.1f}% spent, structural degradation triggered",
        )
        return degraded

    if band == BudgetBand.CRITICAL:
        audit = get_audit_trail()
        audit.record_budget_band(
            task_id=task_id,
            band="critical",
            remaining_budget=budget.remaining_pct,
            action="Critical budget: skipping Judge, returning best available result",
        )
        return "single"

    return current_topology