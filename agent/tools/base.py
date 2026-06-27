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

        # Dynamically create args_schema Pydantic model from parameters dict
        from pydantic import Field, create_model
        fields = {}
        properties = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])
        for name, prop in properties.items():
            desc = prop.get("description", "")
            if name in required:
                fields[name] = (str, Field(..., description=desc))
            else:
                fields[name] = (str | None, Field(prop.get("default", None), description=desc))

        args_schema = create_model(f"{self.name}_args_schema", **fields)

        return StructuredTool(
            name=self.name,
            description=self.description,
            func=_run,
            args_schema=args_schema,
        )
