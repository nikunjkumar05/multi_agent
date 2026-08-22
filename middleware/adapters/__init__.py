from middleware.adapters.base import AgentAdapter, AgentTask, AgentResult
from middleware.adapters.opencode import OpenCodeAdapter
from middleware.adapters.aider import AiderAdapter
from middleware.adapters.mock import MockAdapter

ADAPTERS: dict[str, type[AgentAdapter]] = {
    "opencode": OpenCodeAdapter,
    "aider": AiderAdapter,
    "mock": MockAdapter,
}

__all__ = [
    "AgentAdapter",
    "AgentTask",
    "AgentResult",
    "OpenCodeAdapter",
    "AiderAdapter",
    "MockAdapter",
    "ADAPTERS",
]
