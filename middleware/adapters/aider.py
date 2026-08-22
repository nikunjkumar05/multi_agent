"""Aider adapter — talks to Aider CLI via subprocess.

Runs: aider --message <prompt> --model <model> --no-auto-commits --no-git --yes-always <files>

Aider outputs plain text (no JSON), so we parse stdout for:
  - Token usage from the cost summary line
  - The actual code changes from the diff output
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from middleware.adapters.base import AgentAdapter, AgentResult, AgentTask

log = logging.getLogger(__name__)


class AiderAdapter(AgentAdapter):
    """Adapter for Aider CLI agent.

    Usage:
        adapter = AiderAdapter()
        result = await adapter.execute(task)
    """

    def __init__(
        self,
        binary: str | None = None,
        default_model: str | None = None,
    ):
        self.binary = binary or shutil.which("aider") or "aider"
        self.default_model = default_model  # e.g. "anthropic/claude-sonnet-4-20250514"

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute a task using Aider CLI.

        Spawns: aider --message <prompt> --model <model> --no-auto-commits
                --no-git --yes-always --no-show-release-notes <files>

        Parses stdout for:
          - Cost summary line: "Cost: $0.001234 (approx)"
          - Token counts: "tok usage: 1234 in + 567 out = 1801 total"
          - Diff output as the actual result
        """
        start = time.monotonic()
        project_root = task.context.get("project_root", os.getcwd())

        # Build command
        cmd = [
            self.binary,
            "--message", task.prompt,
            "--no-auto-commits",
            "--no-git",
            "--yes-always",
            "--no-show-release-notes",
            "--no-stream",
        ]

        if self.default_model:
            cmd.extend(["--model", self.default_model])

        # Add files to edit
        files = task.context.get("files", [])
        cmd.extend(files)

        log.info("Aider: running command (cwd=%s)", project_root)
        log.debug("Aider: cmd=%s", cmd)

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

            # Parse output
            output, tokens_used, cost_usd = self._parse_output(stdout)

            success = proc.returncode == 0
            error = None if success else stderr[:500] or f"exit code {proc.returncode}"

            return AgentResult(
                task_id=task.task_id,
                agent="aider",
                output=output,
                cost_usd=cost_usd,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                success=success,
                error=error,
                metadata={
                    "model": self.default_model,
                    "return_code": proc.returncode,
                    "files_edited": files,
                    "stderr": stderr[:200] if stderr else None,
                },
            )

        except asyncio.TimeoutError:
            latency_ms = int((time.monotonic() - start) * 1000)
            if proc and proc.returncode is None:
                proc.kill()
                await proc.wait()
            return AgentResult(
                task_id=task.task_id,
                agent="aider",
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
                agent="aider",
                output="",
                cost_usd=0.0,
                tokens_used=0,
                latency_ms=latency_ms,
                success=False,
                error=f"Aider binary not found: {self.binary}",
            )
        except Exception as e:
            latency_ms = int((time.monotonic() - start) * 1000)
            return AgentResult(
                task_id=task.task_id,
                agent="aider",
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
        """
        input_tokens = max(1, len(task.prompt) // 4)

        for f in task.context.get("files", []):
            try:
                file_size = Path(f).stat().st_size
                input_tokens += file_size // 4
            except OSError:
                pass

        output_tokens = max(100, input_tokens // 5)

        input_cost = (input_tokens / 1000.0) * 0.003
        output_cost = (output_tokens / 1000.0) * 0.015

        return round(input_cost + output_cost, 6)

    def health_check(self) -> bool:
        """Check if aider binary exists and is executable."""
        return shutil.which(self.binary) is not None

    def get_capabilities(self) -> dict[str, Any]:
        """Return Aider capabilities."""
        return {
            "task_types": [
                "code_generation",
                "code_review",
                "refactoring",
                "debugging",
                "documentation",
                "testing",
            ],
            "max_tokens": 128_000,
            "pricing": {
                "input_per_1k": 0.003,
                "output_per_1k": 0.015,
            },
            "latency_p50_ms": 8000,
            "latency_p99_ms": 45000,
            "reliability": 0.92,
            "supports_files": True,
            "supports_model_selection": True,
        }

    def _parse_output(self, stdout: str) -> tuple[str, int, float]:
        """Parse Aider's text output for tokens and cost.

        Aider prints lines like:
          "Cost: $0.001234 (approx)"
          "tok usage: 1234 in + 567 out = 1801 total"

        Also captures the diff/code changes as the actual output.
        """
        total_tokens = 0
        total_cost = 0.0
        output_lines: list[str] = []

        for line in stdout.split("\n"):
            stripped = line.strip()

            # Cost line: "Cost: $0.001234 (approx)"
            cost_match = re.search(r"Cost:\s*\$([0-9.]+)", stripped, re.IGNORECASE)
            if cost_match:
                total_cost = float(cost_match.group(1))
                continue

            # Token line: "tok usage: 1234 in + 567 out = 1801 total"
            tok_match = re.search(
                r"(\d+)\s*(?:in|input).*?(\d+)\s*(?:out|output).*?(\d+)\s*total",
                stripped,
                re.IGNORECASE,
            )
            if tok_match:
                total_tokens = int(tok_match.group(3))
                continue

            # Simpler token line: "1234 tokens"
            simple_tok = re.search(r"(\d+)\s*tokens?", stripped, re.IGNORECASE)
            if simple_tok and total_tokens == 0:
                total_tokens = int(simple_tok.group(1))
                continue

            # Skip aider's own UI lines
            if stripped.startswith("Aider") or stripped.startswith("aider"):
                continue
            if stripped.startswith("Model:") or stripped.startswith("Git"):
                continue
            if stripped == "" and output_lines:
                # Empty line might be separator, keep going
                continue

            # Everything else is potential output
            output_lines.append(line)

        # Join output, trimming leading/trailing empty lines
        output = "\n".join(output_lines).strip()

        # If no tokens found, estimate from output length
        if total_tokens == 0 and output:
            total_tokens = max(1, len(output) // 4)

        # If no cost found, estimate from tokens
        if total_cost == 0 and total_tokens > 0:
            total_cost = (total_tokens / 1000.0) * 0.003

        return output, total_tokens, round(total_cost, 6)
