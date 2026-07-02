"""
Self-optimisation feedback loop.

After each task completes, call `record_task_result()` to:
  1. Update the RL policy with the combined quality + cost-efficiency reward.
  2. Record the outcome in the in-memory PerformanceStats tracker.

This replaces the inline rl.reward() call that was previously scattered
across agent/graph.py, centralising all post-task learning in one place.
"""

from __future__ import annotations

from core.config import settings
from core.rl_policy import RLPolicy
from core.stats import TaskOutcome, stats

# Simple keyword-based task-type detector — mirrors the feature extraction in rl_policy.py
_CODE_KW = {"function", "implement", "code", "class", "script"}
_RESEARCH_KW = {"explain", "research", "compare", "why", "how does", "describe", "summarize", "review"}
_DATA_KW = {"analyze", "data", "parallel", "bulk", "process"}
_VERIFY_KW = {"verify", "audit", "validate", "critical", "security", "proof"}


def detect_task_type(task: str) -> str:
    """
    Classify a plain-English task string into one of five task types.
    Returns the first matching type; falls back to "general".
    """
    lower = task.lower()
    if any(kw in lower for kw in _VERIFY_KW):
        return "verify"
    if any(kw in lower for kw in _CODE_KW):
        return "code"
    if any(kw in lower for kw in _RESEARCH_KW):
        return "research"
    if any(kw in lower for kw in _DATA_KW):
        return "data"
    return "general"


async def record_task_result(
    rl_policy: RLPolicy,
    topology: str,
    budget_band: str,
    task: str,
    quality_score: float,
    cost_usd: float,
    budget_total: float,
) -> None:
    """
    Central post-task hook.  Call once per completed task from agent/graph.py.

    Args:
        rl_policy:     The active RLPolicy instance (already loaded).
        topology:      The topology that was actually executed.
        budget_band:   The final BudgetBand.value string.
        task:          Original task text (used to detect task type).
        quality_score: 0.0–1.0 quality signal (1.0 = success, 0.0 = failure).
        cost_usd:      Actual USD consumed by this task.
        budget_total:  Original budget_usd for this task.
    """
    cost_efficiency = max(0.0, 1.0 - (cost_usd / budget_total)) if budget_total > 0 else 0.0
    reward = quality_score * settings.rl_quality_weight + cost_efficiency * settings.rl_cost_efficiency_weight

    # Update RL policy
    await rl_policy.reward(
        topology=topology,
        quality=quality_score,
        cost_efficiency=cost_efficiency,
    )

    # Record in performance stats
    stats.record(
        TaskOutcome(
            topology=topology,
            budget_band=budget_band,
            task_type=detect_task_type(task),
            quality_score=quality_score,
            cost_usd=cost_usd,
            cost_efficiency=cost_efficiency,
        )
    )
