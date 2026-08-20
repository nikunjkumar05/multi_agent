"""Agent registry — stores all available agents and their capabilities.

The registry is the source of truth for which agents exist, what they can do,
how much they cost, and whether they're healthy.

Usage:
    registry = AgentRegistry()
    registry.register("opencode", OpenCodeAdapter())

    # Get cheapest agent for a task type
    agent = registry.get_cheapest("code_generation")

    # Get all healthy agents
    agents = registry.get_healthy()

    # Check if agent can handle task
    can = registry.can_handle("opencode", "debugging")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from middleware.adapters.base import AgentAdapter

log = logging.getLogger(__name__)


@dataclass
class AgentInfo:
    """Cached agent info with health and reliability metrics."""

    agent_id: str
    adapter: AgentAdapter
    capabilities: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    last_health_check: float = 0.0
    health_check_interval: float = 60.0  # seconds
    is_healthy: bool = True
    success_count: int = 0
    failure_count: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: int = 0
    total_tasks: int = 0

    @property
    def reliability(self) -> float:
        """Success rate (0.0 to 1.0)."""
        total = self.success_count + self.failure_count
        if total == 0:
            return self.capabilities.get("reliability", 0.95)
        return self.success_count / total

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if self.total_tasks == 0:
            return self.capabilities.get("latency_p50_ms", 5000)
        return self.total_latency_ms / self.total_tasks

    @property
    def avg_cost_per_task(self) -> float:
        """Average cost per task."""
        if self.total_tasks == 0:
            return 0.0
        return self.total_cost_usd / self.total_tasks

    def can_handle(self, task_type: str) -> bool:
        """Check if this agent supports the given task type."""
        supported = self.capabilities.get("task_types", [])
        return task_type in supported

    def record_success(self, cost_usd: float, latency_ms: int) -> None:
        """Record a successful task execution."""
        self.success_count += 1
        self.total_cost_usd += cost_usd
        self.total_latency_ms += latency_ms
        self.total_tasks += 1

    def record_failure(self) -> None:
        """Record a failed task execution."""
        self.failure_count += 1
        self.total_tasks += 1


class AgentRegistry:
    """Registry of all available agents.

    Central store for agent adapters, capabilities, and health status.
    Used by the ILP solver and task router to pick the best agent.
    """

    def __init__(self):
        self._agents: dict[str, AgentInfo] = {}

    def register(
        self,
        agent_id: str,
        adapter: AgentAdapter,
        enabled: bool = True,
    ) -> None:
        """Register an agent adapter.

        Args:
            agent_id: Unique identifier (e.g. "opencode", "aider").
            adapter: The agent adapter instance.
            enabled: Whether this agent is enabled by default.
        """
        capabilities = adapter.get_capabilities()
        self._agents[agent_id] = AgentInfo(
            agent_id=agent_id,
            adapter=adapter,
            capabilities=capabilities,
            enabled=enabled,
            is_healthy=adapter.health_check(),
            last_health_check=time.time(),
        )
        log.info(
            "Registered agent: %s (healthy=%s, types=%s)",
            agent_id,
            self._agents[agent_id].is_healthy,
            capabilities.get("task_types", []),
        )

    def unregister(self, agent_id: str) -> bool:
        """Remove an agent from the registry.

        Returns:
            True if agent was removed, False if not found.
        """
        if agent_id in self._agents:
            del self._agents[agent_id]
            log.info("Unregistered agent: %s", agent_id)
            return True
        return False

    def get(self, agent_id: str) -> AgentAdapter | None:
        """Get an agent adapter by ID.

        Returns:
            The adapter, or None if not found/disabled/unhealthy.
        """
        info = self._agents.get(agent_id)
        if info is None:
            return None
        if not info.enabled or not info.is_healthy:
            return None
        return info.adapter

    def get_info(self, agent_id: str) -> AgentInfo | None:
        """Get full agent info by ID."""
        return self._agents.get(agent_id)

    def get_all(self) -> dict[str, AgentAdapter]:
        """Get all healthy, enabled agents."""
        return {
            aid: info.adapter
            for aid, info in self._agents.items()
            if info.enabled and info.is_healthy
        }

    def get_healthy(self) -> dict[str, AgentAdapter]:
        """Get all healthy agents (regardless of enabled status)."""
        return {
            aid: info.adapter
            for aid, info in self._agents.items()
            if info.is_healthy
        }

    def get_for_task_type(self, task_type: str) -> dict[str, AgentAdapter]:
        """Get all agents that can handle a specific task type.

        Returns:
            Dict of agent_id -> adapter, filtered by task type support
            and health status.
        """
        return {
            aid: info.adapter
            for aid, info in self._agents.items()
            if info.enabled and info.is_healthy and info.can_handle(task_type)
        }

    def get_cheapest(self, task_type: str) -> tuple[str, AgentAdapter] | None:
        """Get the cheapest agent for a task type.

        Returns:
            Tuple of (agent_id, adapter) or None if no agent available.
        """
        candidates = self.get_for_task_type(task_type)
        if not candidates:
            return None

        # Sort by input pricing (cheapest first)
        def pricing_key(item: tuple[str, AgentAdapter]) -> float:
            aid, adapter = item
            info = self._agents[aid]
            pricing = info.capabilities.get("pricing", {})
            return pricing.get("input_per_1k", 999)

        sorted_agents = sorted(candidates.items(), key=pricing_key)
        return sorted_agents[0]

    def get_reliable(self, min_reliability: float = 0.9) -> dict[str, AgentAdapter]:
        """Get agents with reliability above threshold."""
        return {
            aid: info.adapter
            for aid, info in self._agents.items()
            if info.enabled and info.is_healthy and info.reliability >= min_reliability
        }

    def can_handle(self, agent_id: str, task_type: str) -> bool:
        """Check if an agent can handle a task type."""
        info = self._agents.get(agent_id)
        if info is None:
            return False
        return info.can_handle(task_type)

    def record_success(self, agent_id: str, cost_usd: float, latency_ms: int) -> None:
        """Record a successful execution for an agent."""
        info = self._agents.get(agent_id)
        if info:
            info.record_success(cost_usd, latency_ms)

    def record_failure(self, agent_id: str) -> None:
        """Record a failed execution for an agent."""
        info = self._agents.get(agent_id)
        if info:
            info.record_failure()

    def refresh_health(self) -> dict[str, bool]:
        """Re-check health for all agents.

        Returns:
            Dict of agent_id -> new_health_status.
        """
        results = {}
        for aid, info in self._agents.items():
            try:
                old_health = info.is_healthy
                info.is_healthy = info.adapter.health_check()
                info.last_health_check = time.time()
                results[aid] = info.is_healthy

                if old_health and not info.is_healthy:
                    log.warning("Agent %s became unhealthy", aid)
                elif not old_health and info.is_healthy:
                    log.info("Agent %s recovered to healthy", aid)

            except Exception as e:
                info.is_healthy = False
                results[aid] = False
                log.error("Health check failed for %s: %s", aid, e)

        return results

    def set_enabled(self, agent_id: str, enabled: bool) -> bool:
        """Enable or disable an agent.

        Returns:
            True if agent was found and updated, False otherwise.
        """
        info = self._agents.get(agent_id)
        if info:
            info.enabled = enabled
            log.info("Agent %s enabled=%s", agent_id, enabled)
            return True
        return False

    def list_agents(self) -> list[dict[str, Any]]:
        """List all agents with their info.

        Returns:
            List of dicts with agent details.
        """
        result = []
        for aid, info in self._agents.items():
            result.append({
                "agent_id": aid,
                "enabled": info.enabled,
                "healthy": info.is_healthy,
                "reliability": round(info.reliability, 2),
                "avg_latency_ms": round(info.avg_latency_ms),
                "avg_cost_per_task": round(info.avg_cost_per_task, 6),
                "total_tasks": info.total_tasks,
                "capabilities": info.capabilities,
            })
        return result

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the registry."""
        total = len(self._agents)
        healthy = sum(1 for i in self._agents.values() if i.is_healthy)
        enabled = sum(1 for i in self._agents.values() if i.enabled)
        total_tasks = sum(i.total_tasks for i in self._agents.values())
        total_cost = sum(i.total_cost_usd for i in self._agents.values())

        return {
            "total_agents": total,
            "healthy_agents": healthy,
            "enabled_agents": enabled,
            "total_tasks_executed": total_tasks,
            "total_cost_usd": round(total_cost, 6),
        }
