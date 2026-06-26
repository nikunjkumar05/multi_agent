from typing import Any

from langchain_core.tools import StructuredTool

from agent.tools.base import BaseTool, ToolResult


class ToolRegistry:
    _instance: "ToolRegistry | None" = None

    def __new__(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools: dict[str, BaseTool] = {}
        return cls._instance

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def execute(self, name: str, **kwargs: Any) -> ToolResult:
        tool = self.get_tool(name)
        if tool is None:
            return ToolResult(success=False, output=None, error=f"Tool '{name}' not found.")
        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    def list_tools(self) -> list[dict[str, Any]]:
        return [tool.to_langchain_tool() for tool in self._tools.values()]

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def get_langchain_tools(self) -> list[StructuredTool]:
        return [tool.to_langchain_tool() for tool in self._tools.values()]


registry = ToolRegistry()
