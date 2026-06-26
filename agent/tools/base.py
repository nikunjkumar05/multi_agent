from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import StructuredTool


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

    def to_langchain_tool(self) -> StructuredTool:
        def _run(**kwargs: Any) -> str:
            result = self.execute(**kwargs)
            if result.success:
                if isinstance(result.output, (dict, list)):
                    import json
                    return json.dumps(result.output, default=str)
                return str(result.output) if result.output is not None else "Done"
            return f"Error: {result.error}"

        return StructuredTool(
            name=self.name,
            description=self.description,
            func=_run,
        )
