from dataclasses import dataclass, field
from enum import Enum
from typing import Literal
from core.config import settings

ModelTier = Literal["cheap", "standard", "frontier"]


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
        degradation_map = {
            "ensemble": "fanout",
            "fanout": "supervisor",
            "supervisor": "pipeline",
            "pipeline": "single",
            "single": "single",
        }
        return degradation_map.get(current_topology, "single")