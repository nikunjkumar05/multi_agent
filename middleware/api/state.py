"""Shared API state — singletons for registry, budget manager, task store.

Lives outside route modules so /budgets and /tasks can share instances
without circular imports.
"""

from __future__ import annotations

import os

from middleware.budget.budget_manager import BudgetManager
from middleware.registry.agent_registry import AgentRegistry


def _build_registry() -> AgentRegistry:
    """Build the agent registry.

    Test mode (BAMAS_MIDDLEWARE_TEST_MODE=1) registers only MockAdapters
    so tests never spawn real CLIs or call live LLM APIs.
    """
    registry = AgentRegistry()
    if os.getenv("BAMAS_MIDDLEWARE_TEST_MODE") == "1":
        from middleware.adapters.mock import MockAdapter

        registry.register("mock", MockAdapter("mock"))
        return registry

    from middleware.adapters.aider import AiderAdapter
    from middleware.adapters.opencode import OpenCodeAdapter

    registry.register("opencode", OpenCodeAdapter())
    registry.register("aider", AiderAdapter())
    return registry


registry: AgentRegistry = _build_registry()
budget_manager: BudgetManager = BudgetManager()

# In-memory task store (replaced by SQLite persistence post-demo).
tasks_db: dict = {}
