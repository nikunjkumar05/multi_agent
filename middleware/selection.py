"""Agent selection — budget-constrained provisioning (paper Component 1, adapted).

The BAMAS paper selects an LLM pool by maximizing quality subject to a budget
constraint (ILP). Here we adapt the same objective to whole agents: each task is
executed by ONE agent, so the problem reduces to:

    maximize   reliability(agent)          (quality proxy)
    subject to estimated_cost(agent) <= remaining_budget

With a handful of registered agents (n < 20) exhaustive evaluation IS the exact
optimum — no solver dependency required, fully deterministic.

Output is a preference ORDER used by the request path (first entry executes,
remaining entries are the fallback chain).
"""

from __future__ import annotations

import logging
from typing import Any

from middleware.adapters.base import AgentAdapter, AgentTask

log = logging.getLogger(__name__)

_MIN_RELIABILITY_DEFAULT = 0.0  # floor below which an agent is never chosen


def score_candidate(
    agent_id: str,
    adapter: AgentAdapter,
    task: AgentTask,
    remaining_budget: float | None,
) -> dict[str, Any]:
    """Score a single agent for a task. Never raises."""
    try:
        est_cost = float(adapter.estimate_cost(task))
    except Exception:
        est_cost = float("inf")

    caps = adapter.get_capabilities()
    quality = float(caps.get("reliability", 0.9))

    affordable = True
    if remaining_budget is not None:
        affordable = est_cost <= remaining_budget

    return {
        "agent_id": agent_id,
        "adapter": adapter,
        "estimated_cost": est_cost,
        "quality": quality,
        "affordable": affordable,
    }


def select_agents(
    candidates: dict[str, AgentAdapter],
    task: AgentTask,
    remaining_budget: float | None = None,
) -> list[tuple[str, AgentAdapter]]:
    """Return agents ordered best-first for execution + fallback.

    Ordering rules (paper-aligned, exact for small n):
      1. Affordable agents before unaffordable ones.
      2. Among affordable: higher reliability wins; ties broken by lower cost,
         then by agent_id for determinism.
      3. Unaffordable agents trail (cheapest-first) so the caller can still 402
         with a meaningful estimate, or degrade deliberately.

    Args:
        candidates: agent_id -> adapter map (already health/type-filtered).
        task: the concrete task, used for per-agent cost estimation.
        remaining_budget: USD available; None disables the affordability split.

    Returns:
        Ordered list of (agent_id, adapter).
    """
    scored = [
        score_candidate(aid, adapter, task, remaining_budget)
        for aid, adapter in candidates.items()
    ]

    def sort_key(s: dict[str, Any]):
        return (
            0 if s["affordable"] else 1,
            -s["quality"],
            s["estimated_cost"],
            s["agent_id"],
        )

    scored.sort(key=sort_key)

    if scored:
        top = scored[0]
        log.debug(
            "Selection order: %s (top=%s est=$%.6f q=%.2f affordable=%s)",
            [s["agent_id"] for s in scored],
            top["agent_id"], top["estimated_cost"], top["quality"], top["affordable"],
        )

    return [(s["agent_id"], s["adapter"]) for s in scored]


def filter_capable(
    registry,  # AgentRegistry (avoid import cycle)
    task_type: str,
    preferred: list[str] | None = None,
) -> dict[str, AgentAdapter]:
    """Health/type-filtered candidates, with preferred agents guaranteed present.

    Preferred agents that exist in the registry are included even if the
    generic capability table omits the task type (explicit user intent wins),
    but unhealthy/disabled agents are always excluded.
    """
    capable = dict(registry.get_for_task_type(task_type))
    for pid in preferred or []:
        info = registry.get_info(pid)
        if info is None or not info.enabled or not info.is_healthy:
            continue
        capable.setdefault(pid, info.adapter)
    return capable
