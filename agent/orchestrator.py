"""
Orchestrator loop for mid-execution topology degradation.

Replaces the single `ainvoke()` call in graph.py with a loop that:
1. Invokes the current topology graph
2. Catches interrupt() returns from budget_gate_node
3. Projects state to the degraded topology
4. Builds a new graph on the same checkpointer
5. Resumes execution on the new topology

CRITICAL-on-single is handled as a terminal policy — returns best
available output with status=degraded_completion.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from core.audit import get_audit_trail
from core.budget import BudgetBand
from core.node_events import emit_event
from core.projections import project_state

log = logging.getLogger(__name__)

_GRAPH_TIMEOUT = 300  # hard timeout per graph invocation (seconds)


async def run_task_with_degradation(
    graph: Any,
    initial_state: dict[str, Any],
    task_id: str,
    topology: str,
    checkpointer: MemorySaver,
) -> dict[str, Any]:
    """
    Run a topology graph with mid-execution degradation support.

    Args:
        graph: The compiled LangGraph for the initial topology.
        initial_state: The initial state dict for the graph.
        task_id: Unique task identifier.
        topology: The initial topology name.
        checkpointer: Shared MemorySaver for cross-graph state.

    Returns:
        The final result dict with status, final_output, topology, etc.
    """
    config = {"configurable": {"thread_id": task_id}}
    current_topology = topology
    current_graph = graph

    # Set initial state
    current_graph.update_state(config, initial_state)

    max_degradations = 5  # Safety limit to prevent infinite loops
    degradation_count = 0
    visited_topologies: set[str] = {topology}  # Track to detect projection cycles

    while degradation_count < max_degradations:
        try:
            result = await asyncio.wait_for(
                current_graph.ainvoke(None, config),
                timeout=_GRAPH_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.error("Graph timed out after %ds on topology %s", _GRAPH_TIMEOUT, current_topology)
            return _build_result(
                {"status": "failed", "final_output": None, "logs": [f"Graph timed out on {current_topology}"]},
                current_topology,
                degradation_count,
            )
        except Exception as e:
            log.error("Graph failed on topology %s: %s", current_topology, e)
            return _build_result(
                {"status": "failed", "final_output": None, "logs": [f"Graph failed on {current_topology}: {e}"]},
                current_topology,
                degradation_count,
            )

        # Check for interrupt
        if "__interrupt__" not in result:
            # Normal completion
            return _build_result(result, current_topology, degradation_count)

        # Interrupt detected — extract payload
        interrupts = result["__interrupt__"]
        interrupt_value = interrupts[0].value

        reason = interrupt_value.get("reason", "")
        from_topology = interrupt_value.get("from_topology", current_topology)
        to_topology = interrupt_value.get("to_topology")

        # EMERGENCY_SINGLE: forced collapse to single (budget_gate CRITICAL on non-single topology)
        if reason == "emergency_single":
            to_topology = "single"

        # SKIP_JUDGE: task is complete but budget exhausted — complete degraded
        if reason == "skip_judge":
            return _build_result(
                {"status": "degraded_completion", "final_output": result.get("final_output") or result.get("final_result")},
                current_topology,
                degradation_count,
            )

        # If no target topology specified, compute from degradation chain
        if not to_topology:
            to_topology = _next_topology(from_topology)

        # Idempotency: if already at target, we've exhausted the chain
        if to_topology == current_topology:
            break

        # Cycle detection: if we've already projected to this topology, break
        if to_topology in visited_topologies:
            break
        visited_topologies.add(to_topology)

        # Get current state for projection
        state = current_graph.get_state(config).values

        # Project state to new topology
        try:
            projected = project_state(state, from_topology, to_topology)
        except ValueError:
            # Unknown projection edge — fall back to single
            to_topology = "single"
            projected = project_state(state, from_topology, to_topology)

        # Emit degradation event
        await emit_event(
            task_id,
            "topology_degraded",
            {
                "from_topology": from_topology,
                "to_topology": to_topology,
                "band": interrupt_value.get("band", "unknown"),
                "reason": reason,
                "degradation_number": degradation_count + 1,
            },
        )

        # Audit the degradation
        audit = get_audit_trail()
        audit.record_degradation(
            task_id=task_id,
            from_topology=from_topology,
            to_topology=to_topology,
            reason=f"Mid-execution: {reason} on {from_topology}",
        )

        # Build new graph for degraded topology
        from agent.topologies.builder import compile_graph

        current_topology = to_topology
        current_graph = compile_graph(current_topology)
        current_graph.update_state(config, projected)

        degradation_count += 1

    # Safety limit reached — return failure
    return _build_result(
        {"status": "failed", "final_output": None}, current_topology, degradation_count
    )


def _build_result(
    result: dict[str, Any], topology: str, degradation_count: int
) -> dict[str, Any]:
    """Build the final result dict from graph output."""
    final_output = result.get("final_output") or result.get("final_result")
    status = result.get("status", "completed")

    # If we degraded and status is still running, mark as degraded completion
    if degradation_count > 0 and status not in ("completed", "failed"):
        status = "degraded_completion"

    return {
        "status": status,
        "final_result": final_output,
        "final_output": final_output,
        "judge_output": result.get("judge_output"),
        "topology": topology,
        "degradation_count": degradation_count,
        "consumed_tokens": result.get("consumed_tokens", 0),
        "consumed_cost": result.get("consumed_cost", 0.0),
        "logs": result.get("logs", []),
    }


TOPOLOGY_DEGRADATION_CHAIN = ["ensemble", "fanout", "supervisor", "pipeline", "single"]


def _next_topology(current: str) -> str:
    """Get the next topology in the degradation chain."""
    try:
        idx = TOPOLOGY_DEGRADATION_CHAIN.index(current)
        return TOPOLOGY_DEGRADATION_CHAIN[min(idx + 1, len(TOPOLOGY_DEGRADATION_CHAIN) - 1)]
    except ValueError:
        return "single"
