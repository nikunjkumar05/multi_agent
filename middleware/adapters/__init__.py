from middleware.adapters.aider import AiderAdapter
from middleware.adapters.base import AgentAdapter, AgentResult, AgentTask
from middleware.adapters.mock import MockAdapter
from middleware.adapters.opencode import OpenCodeAdapter

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
