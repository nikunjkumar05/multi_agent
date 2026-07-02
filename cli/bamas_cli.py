#!/usr/bin/env python3
"""
bamas-cli — Budget Burn Risk Analyser for BAMAS-powered agents.

Usage
-----
    # Requires the BAMAS server to be running
    python -m cli.bamas_cli --task "Write a Python function to compute Fibonacci numbers" --budget 0.50

    # Point at a non-default server
    python -m cli.bamas_cli --task "Audit this codebase for security issues" --budget 2.00 --server http://prod.example.com

    # Output as JSON (useful for scripting / CI pipelines)
    python -m cli.bamas_cli --task "..." --budget 1.00 --json

Exit codes
----------
    0  LOW risk   (>30% budget headroom)
    1  MEDIUM risk (10–30% headroom)
    2  HIGH risk  (<10% headroom)
    3  Error contacting server
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

try:
    import httpx
except ImportError:
    print("ERROR: httpx is required. Run: pip install httpx", file=sys.stderr)
    sys.exit(3)


_RISK_COLORS = {
    "LOW": "\033[92m",  # green
    "MEDIUM": "\033[93m",  # yellow
    "HIGH": "\033[91m",  # red
}
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _colorise(text: str, color: str) -> str:
    """Apply ANSI color if the terminal supports it."""
    if sys.stdout.isatty():
        return f"{color}{text}{_RESET}"
    return text


async def _call_estimate(task: str, budget: float, server: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{server.rstrip('/')}/estimate",
            json={"task": task, "budget_usd": budget},
        )
        resp.raise_for_status()
        return resp.json()


def _print_report(data: dict[str, Any], task: str, as_json: bool) -> int:
    if as_json:
        print(json.dumps(data, indent=2))
    else:
        risk: str = data.get("risk_level", "UNKNOWN")
        color = _RISK_COLORS.get(risk, "")
        print()
        print(_colorise("=== BAMAS Budget Burn Risk Report ===", _BOLD))
        print(f"  Task            : {task[:80]}{'...' if len(task) > 80 else ''}")
        print(f"  Budget          : ${data['budget_usd']:.4f}")
        print(f"  Topology        : {data['topology']}")
        print(f"  Model tiers     : {data['model_tiers']}")
        print(f"  Est. cost       : ${data['estimated_cost_usd']:.6f}")
        print(f"  Headroom        : {data['budget_headroom_pct']:.1f}%")
        print(f"  Risk level      : {_colorise(risk, color)}")
        print()
        print(f"  Rationale       : {data['rationale']}")

        alts = data.get("alternatives_considered", [])
        if alts:
            print()
            print("  Alternatives considered:")
            for alt in alts:
                topo = alt.get("topology", alt.get("raw", "?"))
                reason = alt.get("reason", "")
                print(f"    • {topo}: {reason}")

        print(_colorise("=====================================", _BOLD))
        print()

    # Return exit code based on risk
    return {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(data.get("risk_level", "HIGH"), 2)


async def _run(task: str, budget: float, server: str, as_json: bool) -> int:
    try:
        data = await _call_estimate(task, budget, server)
    except httpx.ConnectError:
        print(f"ERROR: Cannot connect to BAMAS server at {server}", file=sys.stderr)
        print("       Is the server running?  Try: uvicorn api.main:app --reload", file=sys.stderr)
        return 3
    except httpx.HTTPStatusError as exc:
        print(f"ERROR: Server returned {exc.response.status_code}", file=sys.stderr)
        try:
            print(f"       {exc.response.json()}", file=sys.stderr)
        except Exception:
            pass
        return 3
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    return _print_report(data, task, as_json)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bamas-cli",
        description="BAMAS Budget Burn Risk Analyser — dry-run cost estimation",
    )
    parser.add_argument("--task", required=True, help="Plain-English task description")
    parser.add_argument("--budget", type=float, required=True, help="Budget in USD (e.g. 0.50)")
    parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="BAMAS server base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output raw JSON instead of the formatted report",
    )
    args = parser.parse_args()

    exit_code = asyncio.run(_run(args.task, args.budget, args.server, args.as_json))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
