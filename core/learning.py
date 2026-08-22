"""
Self-optimisation feedback loop.

After each task completes, call `record_task_result()` to:
  1. Update the RL policy with the combined quality + cost-efficiency reward.
  2. Record the outcome in the in-memory PerformanceStats tracker.

This replaces the inline rl.reward() call that was previously scattered
across agent/graph.py, centralising all post-task learning in one place.
"""

from __future__ import annotations

# Import the canonical detect_task_type from executor (single source of truth)
from agent.nodes.executor import detect_task_type
from core.rl_policy import RLPolicy
from core.stats import TaskOutcome, stats

# Topology-task type compatibility: which topology is "correct" for each task type
TOPOLOGY_TASK_MAP = {
    "code": "pipeline",
    "research": "supervisor",
    "data": "fanout",
    "verify": "ensemble",
    "math": "pipeline",
    "creative": "single",
    "general": "single",
}


def compute_topology_reward_multiplier(task_type: str, topology: str) -> float:
    """
    Compute a reward multiplier based on topology-task type compatibility.
    
    - 1.3x boost when topology matches the expected type for the task
    - 0.3x penalty when topology doesn't match (teaches RL to avoid wrong choices)
    - 1.0x neutral for general/unknown task types
    """
    expected = TOPOLOGY_TASK_MAP.get(task_type)
    if not expected:
        return 1.0  # Unknown task type — no adjustment
    
    if topology == expected:
        return 1.3  # Boost correct match
    else:
        return 0.3  # Penalize wrong match


async def record_task_result(
    rl_policy: RLPolicy,
    topology: str,
    budget_band: str,
    task: str,
    quality_score: float,
    cost_usd: float,
    budget_total: float,
    llm_topology: str | None = None,
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
        llm_topology:  The topology the LLM originally selected (if RL overrode it).
    """
    cost_efficiency = max(0.0, 1.0 - (cost_usd / budget_total)) if budget_total > 0 else 0.0

    # Detect task type
    task_type = detect_task_type(task)

    # Start with raw quality score — rl_policy.reward() will combine with cost_efficiency
    adjusted_quality = quality_score

    # Apply RL override penalty: if RL overrode the LLM and chose wrong topology, penalize
    if llm_topology and llm_topology != topology:
        expected = TOPOLOGY_TASK_MAP.get(task_type)
        if expected and topology != expected:
            adjusted_quality *= 0.5  # 50% penalty for wrong RL override

    # Apply topology-task type compatibility multiplier
    topo_multiplier = compute_topology_reward_multiplier(task_type, topology)
    adjusted_quality *= topo_multiplier

    # Update RL policy — pass adjusted quality (cost_efficiency is combined inside reward())
    await rl_policy.reward(
        topology=topology,
        quality=adjusted_quality,
        cost_efficiency=cost_efficiency,
    )

    # Record in performance stats
    stats.record(
        TaskOutcome(
            topology=topology,
            budget_band=budget_band,
            task_type=task_type,
            quality_score=quality_score,
            cost_usd=cost_usd,
            cost_efficiency=cost_efficiency,
        )
    )
