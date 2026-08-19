"""Base adapter interface for all coding AI agents.

Every agent (OpenCode, Aider, Cursor, Codex) must implement this interface.
The middleware uses this to talk to any agent without knowing its internals.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentTask:
    """A task sent to an agent for execution.

    This is the standardized input format. Every agent receives the same
    structure regardless of its native API format.
    """

    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    task_type: str = "code_generation"  # code_generation, code_review, refactoring, debugging, documentation, testing
    prompt: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    # context contains: language, project_root, files, etc.
    budget_usd: float = 0.50
    timeout_seconds: int = 120
    preferred_agents: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class AgentResult:
    """Standardized output from an agent.

    Every agent returns the same structure regardless of its native output format.
    The middleware uses this to compare costs, quality, and latency across agents.
    """

    task_id: str = ""
    agent: str = ""
    output: str = ""
    cost_usd: float = 0.0
    tokens_used: int = 0
    latency_ms: int = 0
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentAdapter(ABC):
    """Abstract base class for all agent adapters.

    To add a new agent (e.g., Cursor), create a new file in middleware/adapters/
    and implement these 4 methods. The middleware will automatically use it.

    Example:
        class CursorAdapter(AgentAdapter):
            async def execute(self, task: AgentTask) -> AgentResult:
                # Call Cursor's API
                pass
            def estimate_cost(self, task: AgentTask) -> float:
                return 0.01
            def health_check(self) -> bool:
                return True
            def get_capabilities(self) -> dict:
                return {"task_types": ["code_generation"], "max_tokens": 128000}
    """

    @abstractmethod
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute a task using this agent.

        This is the main method. It should:
        1. Convert AgentTask to the agent's native format
        2. Call the agent (CLI, API, etc.)
        3. Convert the agent's response to AgentResult
        4. Track cost and latency

        Args:
            task: The standardized task to execute.

        Returns:
            AgentResult with output, cost, and status.
        """
        pass

    @abstractmethod
    def estimate_cost(self, task: AgentTask) -> float:
        """Estimate cost before execution.

        Used by the ILP solver to pick the cheapest agent.
        Should be fast (no API calls).

        Args:
            task: The task to estimate cost for.

        Returns:
            Estimated cost in USD.
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if this agent is available.

        Used by the agent registry to filter out dead agents.
        Should be fast (no heavy operations).

        Returns:
            True if agent is available, False otherwise.
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> dict[str, Any]:
        """Return this agent's capabilities.

        Used by the task classifier to match tasks to agents.
        Should return:
            - task_types: list of supported task types
            - max_tokens: maximum context window
            - pricing: dict with input_per_1k and output_per_1k
            - latency_p50_ms: typical latency
            - reliability: success rate (0.0 to 1.0)

        Returns:
            Dict of capabilities.
        """
        pass

    def get_name(self) -> str:
        """Return the agent's name. Override if needed."""
        return self.__class__.__name__.replace("Adapter", "").lower()
