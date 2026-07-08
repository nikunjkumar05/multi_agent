import re
import sqlite3
from typing import Any

from agent.tools.base import BaseTool, ToolResult

BLOCKED_KEYWORDS = {"DELETE", "DROP", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE"}


class DBQueryTool(BaseTool):
    @property
    def name(self) -> str:
        return "db_query"

    @property
    def description(self) -> str:
        return "Execute a read-only SQL query against a SQLite database. SELECT only."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SELECT SQL query"},
                "database": {
                    "type": "string",
                    "description": "Path to SQLite database file",
                    "default": "./workspace/data.db",
                },
            },
            "required": ["query"],
        }
    
    def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query", "").strip()
        db_path = kwargs.get("database", "./workspace/data.db")

        if not query:
            return ToolResult(success=False, output=None, error="No query provided")

        upper = query.upper()
        for kw in BLOCKED_KEYWORDS:
            if re.search(r'\b' + kw + r'\b', upper):
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Blocked: {kw} statements not allowed. SELECT only.",
                )

        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA query_only = ON")
                cursor = conn.execute(query)
                rows = [dict(row) for row in cursor.fetchall()]
            return ToolResult(
                success=True,
                output=rows,
                metadata={"row_count": len(rows)},
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
