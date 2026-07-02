"""
Performance stats tracker — records task outcomes per topology / budget band / task type.
Feeds the RL reward function with richer historical context.
V1: in-memory singleton.  V2: persist to Redis.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean


@dataclass
class TaskOutcome:
    topology: str
    budget_band: str  # BudgetBand.value string
    task_type: str  # "code" | "research" | "data" | "verify" | "general"
    quality_score: float  # 0.0–1.0  (from Judge confidence or binary success)
    cost_usd: float
    cost_efficiency: float  # 1.0 − (spent / total)


class PerformanceStats:
    """Append-only store of TaskOutcome records with query helpers."""

    def __init__(self) -> None:
        self._outcomes: list[TaskOutcome] = []
        self._by_topology: dict[str, list[TaskOutcome]] = defaultdict(list)

    def record(self, outcome: TaskOutcome) -> None:
        self._outcomes.append(outcome)
        self._by_topology[outcome.topology].append(outcome)

    def best_topology_for(self, task_type: str, budget_band: str) -> str | None:
        """
        Returns the topology with the highest mean quality_score for the given
        task_type × budget_band context.  Returns None if fewer than 3 samples
        exist for any topology in that context.
        """
        scored: dict[str, float] = {}
        for topo, outcomes in self._by_topology.items():
            matching = [o.quality_score for o in outcomes if o.task_type == task_type and o.budget_band == budget_band]
            if len(matching) >= 3:
                scored[topo] = mean(matching)
        return max(scored, key=scored.__getitem__) if scored else None

    def summary(self) -> dict[str, object]:
        """Compact stats dict for the /health or observability endpoint."""
        total = len(self._outcomes)
        if total == 0:
            return {"total_tasks": 0}
        avg_quality = mean(o.quality_score for o in self._outcomes)
        avg_cost = mean(o.cost_usd for o in self._outcomes)
        topology_counts = {t: len(v) for t, v in self._by_topology.items()}
        return {
            "total_tasks": total,
            "avg_quality": round(avg_quality, 3),
            "avg_cost_usd": round(avg_cost, 5),
            "topology_counts": topology_counts,
        }


# Module-level singleton
stats = PerformanceStats()
