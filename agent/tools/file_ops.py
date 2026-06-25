import os
from pathlib import Path
from typing import Any, Literal

from agent.tools.base import BaseTool, ToolResult

WORKSPACE = Path("./workspace").resolve()


def _safe_path(rel_path: str) -> Path | None:
    try:
        resolved = (WORKSPACE / rel_path).resolve()
        if not str(resolved).startswith(str(WORKSPACE)):
            return None
        return resolved
    except Exception:
        return None


class FileReadTool(BaseTool):
    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return "Read a file from the workspace directory."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path within workspace"},
            },
            "required": ["path"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        path = _safe_path(kwargs.get("path", ""))
        if path is None:
            return ToolResult(success=False, output=None, error="Invalid path")
        if not path.exists():
            return ToolResult(success=False, output=None, error="File not found")
        try:
            content = path.read_text(encoding="utf-8")
            return ToolResult(success=True, output=content)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class FileWriteTool(BaseTool):
    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        return "Write content to a file in the workspace directory."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path within workspace"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        path = _safe_path(kwargs.get("path", ""))
        content = kwargs.get("content", "")
        if path is None:
            return ToolResult(success=False, output=None, error="Invalid path")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return ToolResult(success=True, output=f"Written to {path.relative_to(WORKSPACE)}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class FileListTool(BaseTool):
    @property
    def name(self) -> str:
        return "file_list"

    @property
    def description(self) -> str:
        return "List files in the workspace directory."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative subdirectory (default: root)"},
            },
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        rel_path = kwargs.get("path", "")
        path = _safe_path(rel_path) if rel_path else WORKSPACE
        if path is None:
            return ToolResult(success=False, output=None, error="Invalid path")
        if not path.exists():
            return ToolResult(success=False, output=None, error="Directory not found")
        try:
            entries = []
            for entry in sorted(path.iterdir()):
                kind = "dir" if entry.is_dir() else "file"
                entries.append({"name": entry.name, "type": kind})
            return ToolResult(success=True, output=entries)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))