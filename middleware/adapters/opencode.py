"""OpenCode adapter — talks to OpenCode CLI via subprocess.

Runs: opencode run --format json --auto --dir <project_root> <prompt>

The --format json flag gives us structured JSON events we can parse
for output, cost, and token usage.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from middleware.adapters.base import AgentAdapter, AgentResult, AgentTask

log = logging.getLogger(__name__)


class OpenCodeAdapter(AgentAdapter):
    """Adapter for OpenCode CLI agent.

    Usage:
        adapter = OpenCodeAdapter()
        result = await adapter.execute(task)
    """

    def __init__(
        self,
        binary: str | None = None,
        default_model: str | None = None,
    ):
        # Find the opencode binary
        self.binary = binary or shutil.which("opencode") or "opencode"
        self.default_model = default_model  # e.g. "anthropic/claude-sonnet-4-20250514"

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute a task using OpenCode CLI.

        Spawns: opencode run --format json --auto --dir <cwd> <prompt>

        Parses JSON event stream for:
          - assistant messages → output
          - usage events → token counts
          - cost events → cost_usd
        """
        start = time.monotonic()
        project_root = task.context.get("project_root", os.getcwd())

        # Build command
        cmd = [
            self.binary,
            "run",
            "--format", "json",
            "--auto",
            "--dir", project_root,
        ]

        if self.default_model:
            cmd.extend(["--model", self.default_model])

        # Attach files if provided
        files = task.context.get("files", [])
        for f in files:
            cmd.extend(["--file", str(f)])

        # The prompt is the message
        cmd.append(task.prompt)

        log.info("OpenCode: running command (cwd=%s)", project_root)
        log.debug("OpenCode: cmd=%s", cmd)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=project_root,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=task.timeout_seconds,
            )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            latency_ms = int((time.monotonic() - start) * 1000)

            # Parse JSON event stream
            output, tokens_used, cost_usd = self._parse_json_output(stdout)

            success = proc.returncode == 0
            error = None if success else stderr[:500] or f"exit code {proc.returncode}"

            return AgentResult(
                task_id=task.task_id,
                agent="opencode",
                output=output,
                cost_usd=cost_usd,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                success=success,
                error=error,
                metadata={
                    "model": self.default_model,
                    "return_code": proc.returncode,
                    "stderr": stderr[:200] if stderr else None,
                },
            )

        except asyncio.TimeoutError:
            latency_ms = int((time.monotonic() - start) * 1000)
            # Kill the process if still running
            if proc and proc.returncode is None:
                proc.kill()
                await proc.wait()
            return AgentResult(
                task_id=task.task_id,
                agent="opencode",
                output="",
                cost_usd=0.0,
                tokens_used=0,
                latency_ms=latency_ms,
                success=False,
                error=f"Timeout after {task.timeout_seconds}s",
            )
        except FileNotFoundError:
            latency_ms = int((time.monotonic() - start) * 1000)
            return AgentResult(
                task_id=task.task_id,
                agent="opencode",
                output="",
                cost_usd=0.0,
                tokens_used=0,
                latency_ms=latency_ms,
                success=False,
                error=f"OpenCode binary not found: {self.binary}",
            )
        except Exception as e:
            latency_ms = int((time.monotonic() - start) * 1000)
            return AgentResult(
                task_id=task.task_id,
                agent="opencode",
                output="",
                cost_usd=0.0,
                tokens_used=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            )

    def estimate_cost(self, task: AgentTask) -> float:
        """Estimate cost based on prompt length and known model pricing.

        Uses rough heuristic: ~$0.003/1K input, ~$0.015/1K output for standard models.
        The ILP solver will use this to compare agents.
        """
        # Estimate input tokens: chars / 4
        input_tokens = max(1, len(task.prompt) // 4)

        # Add context tokens
        for f in task.context.get("files", []):
            try:
                file_size = Path(f).stat().st_size
                input_tokens += file_size // 4
            except OSError:
                pass

        # Estimate output tokens: ~20% of input
        output_tokens = max(100, input_tokens // 5)

        # Pricing per 1K tokens (standard model)
        input_cost = (input_tokens / 1000.0) * 0.003
        output_cost = (output_tokens / 1000.0) * 0.015

        return round(input_cost + output_cost, 6)

    def health_check(self) -> bool:
        """Check if opencode binary exists and is executable."""
        return shutil.which(self.binary) is not None

    def get_capabilities(self) -> dict[str, Any]:
        """Return OpenCode capabilities."""
        return {
            "task_types": [
                "code_generation",
                "code_review",
                "refactoring",
                "debugging",
                "documentation",
                "testing",
            ],
            "max_tokens": 200_000,
            "pricing": {
                "input_per_1k": 0.003,
                "output_per_1k": 0.015,
            },
            "latency_p50_ms": 5000,
            "latency_p99_ms": 30000,
            "reliability": 0.95,
            "supports_files": True,
            "supports_model_selection": True,
        }

    def _parse_json_output(self, stdout: str) -> tuple[str, int, float]:
        """Parse OpenCode's JSON event stream.

        OpenCode outputs one JSON object per line. Events include:
          {"type":"assistant","content":"...","model":"..."}
          {"type":"usage","input_tokens":N,"output_tokens":N}
          {"type":"cost","cost_usd":0.00123}

        Returns: (output_text, total_tokens, total_cost_usd)
        """
        output_parts: list[str] = []
        total_tokens = 0
        total_cost = 0.0

        for line in stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Not JSON — might be plain text output
                output_parts.append(line)
                continue

            event_type = event.get("type", "")

            # Assistant message — this is the actual output
            if event_type in ("assistant", "message", "text", "content"):
                content = event.get("content") or event.get("text") or event.get("message") or ""
                if content:
                    output_parts.append(str(content))

            # Usage event — token counts
            if event_type in ("usage", "tokens"):
                input_tok = event.get("input_tokens", 0) or 0
                output_tok = event.get("output_tokens", 0) or 0
                total_tokens += input_tok + output_tok

            # Cost event — USD cost
            if event_type in ("cost", "expense"):
                cost = event.get("cost_usd", 0) or event.get("cost", 0) or 0
                total_cost += float(cost)

            # Some formats embed usage in the message event itself
            if "usage" in event:
                usage = event["usage"]
                total_tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            if "cost_usd" in event:
                total_cost += float(event["cost_usd"])

        # If no JSON events parsed, treat entire stdout as output
        output = "\n".join(output_parts) if output_parts else stdout.strip()

        # If still no tokens, estimate from output length
        if total_tokens == 0 and output:
            total_tokens = max(1, len(output) // 4)

        # If still no cost, estimate from tokens
        if total_cost == 0 and total_tokens > 0:
            total_cost = (total_tokens / 1000.0) * 0.003  # rough estimate

        return output, total_tokens, round(total_cost, 6)
