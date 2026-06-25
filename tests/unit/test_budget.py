import pytest
from core.budget import BudgetTracker, BudgetBand


class TestBudgetBands:
    def test_healthy_band_below_70(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.5)
        assert bt.spent_pct == 50.0
        assert bt.get_band() == BudgetBand.HEALTHY

    def test_tier_downgrade_band_70_to_90(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.8)
        assert bt.get_band() == BudgetBand.TIER_DOWNGRADE

    def test_structural_degrade_band_90_to_100(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.95)
        assert bt.get_band() == BudgetBand.STRUCTURAL_DEGRADE

    def test_critical_band_at_100(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=1.0)
        assert bt.get_band() == BudgetBand.CRITICAL

    def test_critical_band_over_100(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=1.5)
        assert bt.get_band() == BudgetBand.CRITICAL


class TestCanAffordTier:
    def test_healthy_affords_all(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.0)
        assert bt.can_afford_tier("cheap") is True
        assert bt.can_afford_tier("standard") is True
        assert bt.can_afford_tier("frontier") is True

    def test_tier_downgrade_no_frontier(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.8)
        assert bt.can_afford_tier("cheap") is True
        assert bt.can_afford_tier("standard") is True
        assert bt.can_afford_tier("frontier") is False

    def test_structural_degrade_only_cheap(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.95)
        assert bt.can_afford_tier("cheap") is True
        assert bt.can_afford_tier("standard") is False
        assert bt.can_afford_tier("frontier") is False

    def test_critical_affords_nothing(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=1.0)
        assert bt.can_afford_tier("cheap") is False
        assert bt.can_afford_tier("standard") is False
        assert bt.can_afford_tier("frontier") is False


class TestGetAllowedTiers:
    def test_healthy_all_tiers(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.0)
        assert bt.get_allowed_tiers() == ["cheap", "standard", "frontier"]

    def test_tier_downgrade_two_tiers(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.8)
        assert bt.get_allowed_tiers() == ["cheap", "standard"]

    def test_structural_degrade_one_tier(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.95)
        assert bt.get_allowed_tiers() == ["cheap"]

    def test_critical_one_tier(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=1.0)
        assert bt.get_allowed_tiers() == ["cheap"]


class TestShouldSkipJudge:
    def test_healthy_no_skip(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.0)
        assert bt.should_skip_judge() is False

    def test_tier_downgrade_no_skip(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.8)
        assert bt.should_skip_judge() is False

    def test_structural_degrade_skip(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.95)
        assert bt.should_skip_judge() is True

    def test_critical_skip(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=1.0)
        assert bt.should_skip_judge() is True


class TestRecordUsage:
    def test_record_usage_increments(self):
        bt = BudgetTracker(max_cost_usd=1.0)
        bt.record_usage(tokens=100, cost=0.05)
        assert bt.consumed_tokens == 100
        assert bt.consumed_cost == 0.05

    def test_record_usage_accumulates(self):
        bt = BudgetTracker(max_cost_usd=1.0)
        bt.record_usage(tokens=100, cost=0.05)
        bt.record_usage(tokens=200, cost=0.10)
        assert bt.consumed_tokens == 300
        assert bt.consumed_cost == pytest.approx(0.15)


class TestGetDegradedTopology:
    def test_ensemble_to_fanout(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.0)
        assert bt.get_degraded_topology("ensemble") == "fanout"

    def test_fanout_to_supervisor(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.0)
        assert bt.get_degraded_topology("fanout") == "supervisor"

    def test_supervisor_to_pipeline(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.0)
        assert bt.get_degraded_topology("supervisor") == "pipeline"

    def test_pipeline_to_single(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.0)
        assert bt.get_degraded_topology("pipeline") == "single"

    def test_single_stays_single(self):
        bt = BudgetTracker(max_cost_usd=1.0, consumed_cost=0.0)
        assert bt.get_degraded_topology("single") == "single"
