from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ToolResult:
    success: bool
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property    
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        pass
    
    def to_langchain_tool(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }