import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from agent.tools.base import BaseTool, ToolResult

class CodeExecutor(BaseTool):
    @property
    def name(self) -> str:
        return "code_executor"

    @property
    def description(self) -> str:
        return "Execute Python code in a sandboxed subprocess. Timeout: 10s. Does NOT support input() or interactive prompts — use hardcoded test values. Pass CLI args via the 'args' parameter."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute."
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command-line arguments to pass to the script (e.g. [\"-l\", \"12\"]).",
                },
            },
            "required": ["code"]
        }
    
    def execute(self, **kwargs: Any) -> ToolResult:
        code = kwargs.get("code", "")
        args = kwargs.get("args", [])
        if not code:
            return ToolResult(success=False, output=None, error="No code provided.")
        
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                tmp_path = f.name
            
            result = subprocess.run(
                [sys.executable, tmp_path] + (args or []),
                capture_output=True,
                text=True,
                timeout=10,
                stdin=subprocess.DEVNULL,
            )

            if result.returncode != 0:
                return ToolResult(success=False, output=result.stdout, error=result.stderr)

            if not result.stdout.strip():
                return ToolResult(
                    success=True,
                    output="Code ran successfully but produced no output. Did you forget print()?",
                    metadata={"returncode": result.returncode, "empty_output": True},
                )

            return ToolResult(
                success=True,
                output=result.stdout,
                metadata={"returncode": result.returncode},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output=None, error="Execution timed out (10s)")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
