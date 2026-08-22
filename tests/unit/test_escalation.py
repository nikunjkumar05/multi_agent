from core.budget import BudgetTracker
from core.escalation import ESCALATION_THRESHOLD_CONFIDENCE, should_escalate


class TestShouldEscalate:
    def test_high_confidence_no_escalation(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.0)
        assert should_escalate(validator_confidence=0.9, reasoning_diverged=True, budget=bt) is False

    def test_low_confidence_no_divergence_no_escalation(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.0)
        assert should_escalate(validator_confidence=0.5, reasoning_diverged=False, budget=bt) is False

    def test_low_confidence_with_divergence_escalates(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.0)
        assert should_escalate(validator_confidence=0.5, reasoning_diverged=True, budget=bt) is True

    def test_budget_critical_skips_escalation(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=1.0)
        assert should_escalate(validator_confidence=0.3, reasoning_diverged=True, budget=bt) is False

    def test_budget_structural_degrade_skips_escalation(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.95)
        assert should_escalate(validator_confidence=0.3, reasoning_diverged=True, budget=bt) is False

    def test_exact_threshold_no_escalation(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.0)
        assert should_escalate(
            validator_confidence=ESCALATION_THRESHOLD_CONFIDENCE,
            reasoning_diverged=True,
            budget=bt,
        ) is False

    def test_just_below_threshold_escalates(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.0)
        assert should_escalate(
            validator_confidence=ESCALATION_THRESHOLD_CONFIDENCE - 0.01,
            reasoning_diverged=True,
            budget=bt,
        ) is True
