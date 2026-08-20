from middleware.adapters.base import AgentAdapter, AgentTask, AgentResult
from middleware.adapters.opencode import OpenCodeAdapter
from middleware.adapters.aider import AiderAdapter

ADAPTERS: dict[str, type[AgentAdapter]] = {
    "opencode": OpenCodeAdapter,
    "aider": AiderAdapter,
}

__all__ = ["AgentAdapter", "AgentTask", "AgentResult", "OpenCodeAdapter", "AiderAdapter", "ADAPTERS"]
