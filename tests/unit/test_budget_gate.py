"""Unit tests for agent/nodes/budget_gate.py — evaluate_gate() pure function."""


from agent.nodes.budget_gate import BudgetGateAction, evaluate_gate
from core.budget import BudgetTracker


def _make_budget(spent_pct: float, max_cost: float = 1.0) -> BudgetTracker:
    """Create a BudgetTracker with a specific spend percentage."""
    budget = BudgetTracker(max_cost_usd=max_cost)
    # Manually set consumed to hit the target percentage
    budget.consumed_cost = max_cost * spent_pct
    return budget


class TestEvaluateGate:
    def test_healthy_continues(self):
        budget = _make_budget(0.5)  # 50% spent
        state = {"budget": budget, "topology": "ensemble"}
        assert evaluate_gate(state) == BudgetGateAction.CONTINUE

    def test_tier_downgrade_continues(self):
        budget = _make_budget(0.8)  # 80% spent — TIER_DOWNGRADE band
        state = {"budget": budget, "topology": "ensemble"}
        assert evaluate_gate(state) == BudgetGateAction.CONTINUE

    def test_structural_degrade_triggers_pause(self):
        budget = _make_budget(0.95)  # 95% spent — STRUCTURAL_DEGRADE
        state = {"budget": budget, "topology": "pipeline"}
        assert evaluate_gate(state) == BudgetGateAction.PAUSE

    def test_structural_degrade_on_single_continues(self):
        budget = _make_budget(0.95)  # 95% spent — STRUCTURAL_DEGRADE
        state = {"budget": budget, "topology": "single"}
        assert evaluate_gate(state) == BudgetGateAction.CONTINUE

    def test_critical_on_single_skips_judge(self):
        budget = _make_budget(1.1)  # 110% spent — CRITICAL
        state = {"budget": budget, "topology": "single"}
        assert evaluate_gate(state) == BudgetGateAction.SKIP_JUDGE

    def test_critical_on_ensemble_emergency_single(self):
        budget = _make_budget(1.1)  # 110% spent — CRITICAL
        state = {"budget": budget, "topology": "ensemble"}
        assert evaluate_gate(state) == BudgetGateAction.EMERGENCY_SINGLE

    def test_critical_on_pipeline_emergency_single(self):
        budget = _make_budget(1.2)  # 120% spent — CRITICAL
        state = {"budget": budget, "topology": "pipeline"}
        assert evaluate_gate(state) == BudgetGateAction.EMERGENCY_SINGLE

    def test_no_budget_continues(self):
        state = {"budget": None, "topology": "ensemble"}
        assert evaluate_gate(state) == BudgetGateAction.CONTINUE

    def test_missing_budget_key_continues(self):
        state = {"topology": "ensemble"}
        assert evaluate_gate(state) == BudgetGateAction.CONTINUE
