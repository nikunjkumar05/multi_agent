import pytest
from unittest.mock import patch
from core.degrader import degrade_topology
from core.budget import BudgetTracker, BudgetBand, TOPOLOGY_DEGRADATION_CHAIN
from core.audit import AuditTrail


def _fresh_audit() -> AuditTrail:
    return AuditTrail()


class TestDegradeTopology:
    def test_healthy_band_returns_same(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.0)
        result = degrade_topology(bt, "ensemble", "task-1")
        assert result == "ensemble"

    def test_tier_downgrade_returns_same(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.8)
        result = degrade_topology(bt, "supervisor", "task-2")
        assert result == "supervisor"

    def test_structural_degrade_ensemble_to_fanout(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.95)
        result = degrade_topology(bt, "ensemble", "task-3")
        assert result == "fanout"

    def test_structural_degrade_fanout_to_feedback(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.95)
        result = degrade_topology(bt, "fanout", "task-4")
        assert result == "feedback"

    def test_structural_degrade_feedback_to_supervisor(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.95)
        result = degrade_topology(bt, "feedback", "task-4b")
        assert result == "supervisor"

    def test_structural_degrade_supervisor_to_pipeline(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.95)
        result = degrade_topology(bt, "supervisor", "task-5")
        assert result == "pipeline"

    def test_structural_degrade_pipeline_to_single(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.95)
        result = degrade_topology(bt, "pipeline", "task-6")
        assert result == "single"

    def test_structural_degrade_single_stays_single(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.95)
        result = degrade_topology(bt, "single", "task-7")
        assert result == "single"

    def test_structural_degrade_unknown_to_single(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.95)
        result = degrade_topology(bt, "unknown_topology", "task-8")
        assert result == "single"

    def test_critical_band_returns_single(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=1.0)
        result = degrade_topology(bt, "ensemble", "task-9")
        assert result == "single"

    def test_degradation_chain_order(self):
        assert TOPOLOGY_DEGRADATION_CHAIN == ["ensemble", "fanout", "feedback", "supervisor", "pipeline", "single"]

    @patch("core.degrader.get_audit_trail")
    def test_audit_records_on_structural_degrade(self, mock_get_audit):
        audit = _fresh_audit()
        mock_get_audit.return_value = audit
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.95)
        degrade_topology(bt, "ensemble", "task-audit-1")
        events = audit.get_task_audit("task-audit-1")
        event_types = [e["event_type"] for e in events]
        assert "budget_band_crossed" in event_types
        assert "structural_degradation" in event_types

    @patch("core.degrader.get_audit_trail")
    def test_audit_records_on_critical(self, mock_get_audit):
        audit = _fresh_audit()
        mock_get_audit.return_value = audit
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=1.0)
        degrade_topology(bt, "ensemble", "task-audit-2")
        events = audit.get_task_audit("task-audit-2")
        assert any(e["event_type"] == "budget_band_crossed" for e in events)
