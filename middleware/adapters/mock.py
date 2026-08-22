"""Mock adapter — deterministic agent for testing.

Never spawns processes or makes network calls. Returns instantly
with a canned result. Used when BAMAS_MIDDLEWARE_TEST_MODE=1 so unit
tests don't hit real CLIs / LLM APIs.
"""

from __future__ import annotations

import asyncio
from typing import Any

from middleware.adapters.base import AgentAdapter, AgentResult, AgentTask


class MockAdapter(AgentAdapter):
    """Instant fake agent for tests and dry runs."""

    def __init__(self, name: str = "mock", fail: bool = False):
        self._name = name
        self._fail = fail

    async def execute(self, task: AgentTask) -> AgentResult:
        await asyncio.sleep(0)  # yield to event loop
        latency_ms = 1
        if self._fail:
            return AgentResult(
                task_id=task.task_id,
                agent=self._name,
                output="",
                cost_usd=0.0,
                tokens_used=0,
                latency_ms=latency_ms,
                success=False,
                error="Mock failure (intentional)",
            )
        return AgentResult(
            task_id=task.task_id,
            agent=self._name,
            output=f"[mock:{self._name}] {task.prompt}",
            cost_usd=0.0001,
            tokens_used=max(1, len(task.prompt) // 4),
            latency_ms=latency_ms,
            success=True,
            metadata={"mock": True},
        )

    def estimate_cost(self, task: AgentTask) -> float:
        return 0.0001

    def health_check(self) -> bool:
        return True

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "task_types": [
                "code_generation",
                "code_review",
                "refactoring",
                "debugging",
                "documentation",
                "testing",
                "explanation",
            ],
            "max_tokens": 8_000,
            "pricing": {"input_per_1k": 0.0, "output_per_1k": 0.0},
            "latency_p50_ms": 1,
            "latency_p99_ms": 5,
            "reliability": 1.0,
            "supports_files": False,
            "supports_model_selection": False,
            "mock": True,
        }

    def get_name(self) -> str:
        return self._name
