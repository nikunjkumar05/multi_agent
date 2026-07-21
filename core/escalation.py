from core.budget import BudgetTracker

ESCALATION_THRESHOLD_CONFIDENCE = 0.85


def should_escalate(validator_confidence: float, reasoning_diverged: bool, budget: BudgetTracker) -> bool:
    if budget.should_skip_judge():
        return False

    if validator_confidence >= ESCALATION_THRESHOLD_CONFIDENCE:
        return False

    if not reasoning_diverged:
        return False

    return True