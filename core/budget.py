from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from core.config import settings

ModelTier = Literal["cheap", "standard", "frontier"]

TOPOLOGY_DEGRADATION_CHAIN = ["ensemble", "fanout", "feedback", "supervisor", "pipeline", "single"]


class BudgetBand(str, Enum):
    HEALTHY = "healthy"
    TIER_DOWNGRADE = "tier_downgrade"
    STRUCTURAL_DEGRADE = "structural_degrade"
    CRITICAL = "critical"

@dataclass
class BudgetTracker:
    max_cost_usd: float = field(default_factory=lambda: settings.budget_max_cost_usd)
    max_tokens: int = field(default_factory=lambda: settings.budget_max_tokens)
    consumed_cost: float = 0.0
    consumed_tokens: int = 0

    @property
    def spent_pct(self) -> float:
        return (self.consumed_cost / self.max_cost_usd * 100) if self.max_cost_usd > 0 else 0.0
    
    @property
    def remaining_pct(self) -> float:
        return 100.0 - self.spent_pct
    
    def get_band(self) -> BudgetBand:
        spent = self.spent_pct
        if spent < 70:
            return BudgetBand.HEALTHY
        elif spent < 90:
            return BudgetBand.TIER_DOWNGRADE
        elif spent < 100:
            return BudgetBand.STRUCTURAL_DEGRADE
        else:
            return BudgetBand.CRITICAL
    

    def can_afford_tier(self, tier: ModelTier) -> bool:
        if self.get_band() == BudgetBand.CRITICAL:
            return False
        if self.get_band() == BudgetBand.STRUCTURAL_DEGRADE:
            return tier == "cheap"
        if self.get_band() == BudgetBand.TIER_DOWNGRADE:
            return tier in ("cheap", "standard")
        return True

    def get_allowed_tiers(self) -> list[ModelTier]:
        band = self.get_band()
        if band == BudgetBand.CRITICAL:
            return ["cheap"]
        if band == BudgetBand.STRUCTURAL_DEGRADE:
            return ["cheap"]
        if band == BudgetBand.TIER_DOWNGRADE:
            return ["cheap", "standard"]
        return ["cheap", "standard", "frontier"]

    def record_usage(self, tokens: int, cost: float) -> None:
        self.consumed_tokens += tokens
        self.consumed_cost += cost

    def should_skip_judge(self) -> bool:
        return self.get_band() in (BudgetBand.STRUCTURAL_DEGRADE, BudgetBand.CRITICAL)
    
    def get_degraded_topology(self, current_topology: str) -> str:
        return next_topology(current_topology)


def next_topology(current: str) -> str:
    """Get the next topology in the degradation chain."""
    try:
        idx = TOPOLOGY_DEGRADATION_CHAIN.index(current)
        return TOPOLOGY_DEGRADATION_CHAIN[min(idx + 1, len(TOPOLOGY_DEGRADATION_CHAIN) - 1)]
    except ValueError:
        return "single"


def should_skip_llm(state: dict, threshold: float = 0.9) -> bool:
    """Check if budget is too low to make an LLM call.

    Args:
        state: Current graph state with 'budget' and 'consumed_cost'.
        threshold: Skip if spent_pct >= threshold (default 0.9 = 90%).

    Returns:
        True if LLM call should be skipped (budget exhausted).
    """
    budget = state.get("budget")
    if not budget or budget.max_cost_usd <= 0:
        return False
    acc_cost = state.get("consumed_cost", 0.0)
    spent_pct = acc_cost / budget.max_cost_usd
    return spent_pct >= threshold


def get_band_from_state(state: dict) -> BudgetBand:
    """Compute the budget band from the state's consumed_cost.

    This is the correct way to get the band during execution, because
    BudgetTracker.consumed_cost may be stale (never updated mid-execution).
    The annotated `consumed_cost` field in state tracks the real cost.
    """
    budget = state.get("budget")
    if not budget or budget.max_cost_usd <= 0:
        return BudgetBand.HEALTHY
    acc_cost = state.get("consumed_cost", 0.0)
    spent_pct = acc_cost / budget.max_cost_usd * 100
    if spent_pct < 70:
        return BudgetBand.HEALTHY
    elif spent_pct < 90:
        return BudgetBand.TIER_DOWNGRADE
    elif spent_pct < 100:
        return BudgetBand.STRUCTURAL_DEGRADE
    else:
        return BudgetBand.CRITICAL
