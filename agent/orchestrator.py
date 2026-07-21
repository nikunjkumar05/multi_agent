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

Hardening (v2):
- Budget check after projection prevents wasted tokens
- BudgetTracker sync ensures correct band detection
- Rollback on projection failure continues with current topology
- Quality feedback loop tracks degradation impact
- Circuit breaker tries degradation before stopping
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START

from core.audit import get_audit_trail
from core.budget import BudgetBand, BudgetTracker, next_topology
from core.node_events import emit_event
from core.projections import project_state, validate_projected_state

log = logging.getLogger(__name__)

_GRAPH_TIMEOUT = 300  # hard timeout per graph invocation (seconds)
_BUDGET_EXHAUSTED_THRESHOLD = 0.95  # Stop if 95%+ budget spent after projection


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

    # Set initial state — as_node=START ensures execution begins from entry_router
    current_graph.update_state(config, initial_state, as_node=START)

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
            # Normal completion — read accumulated cost from checkpoint
            final_state = current_graph.get_state(config).values
            return _build_result(result, current_topology, degradation_count, final_state)

        # Interrupt detected — extract payload
        interrupts = result["__interrupt__"]
        if not interrupts:
            log.warning("Empty interrupts list in result — treating as normal completion")
            return _build_result(result, current_topology, degradation_count)
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

        # CIRCUIT_BREAKER: budget exceeded 110% — try degradation first, then stop
        if reason == "circuit_breaker":
            if current_topology != "single":
                # Try emergency degradation to single
                to_topology = "single"
                log.warning("Circuit breaker: trying emergency degradation to single before stopping")
            else:
                # Already at single — circuit breaker stops execution
                log.error("Circuit breaker: budget exceeded 110%% on single topology — stopping")
                return _build_result(
                    {
                        "status": "failed",
                        "final_output": result.get("final_output") or result.get("final_result"),
                        "logs": [f"Circuit breaker: budget exceeded {interrupt_value.get('hard_cap', 0):.4f}"],
                    },
                    current_topology,
                    degradation_count,
                )

        # If no target topology specified, compute from degradation chain
        if not to_topology:
            to_topology = next_topology(from_topology)

        # Idempotency: if already at target, we've exhausted the chain
        if to_topology == current_topology:
            log.warning("Degradation chain exhausted: %s -> %s (same topology)", from_topology, to_topology)
            break

        # Cycle detection: if we've already projected to this topology, break
        if to_topology in visited_topologies:
            log.warning("Projection cycle detected: %s already visited", to_topology)
            break
        visited_topologies.add(to_topology)

        # Get current state for projection
        state = current_graph.get_state(config).values

        # Project state to new topology
        projected, to_topology = _safe_project(state, from_topology, to_topology, current_topology)
        if projected is None:
            # All projections failed — return failure
            return _build_result(
                {"status": "failed", "final_output": None, "logs": [f"Projection failed: {from_topology} -> {to_topology}"]},
                current_topology,
                degradation_count,
            )

        # Budget check after projection — prevent wasted tokens
        budget_check = _check_budget_after_projection(projected, degradation_count)
        if budget_check is not None:
            return budget_check

        # Sync BudgetTracker with projected state
        _sync_budget_tracker(projected)

        # Record topology history for this degradation
        # topology_history is annotated with operator.add — append a record
        projected["topology_history"] = [{"from": from_topology, "to": to_topology, "reason": reason}]

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
                "consumed_cost": projected.get("consumed_cost", 0.0),
                "budget_remaining": _get_budget_remaining(projected),
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
        current_graph.update_state(config, projected, as_node=START)

        degradation_count += 1

    # Safety limit reached — return failure
    return _build_result(
        {"status": "failed", "final_output": None, "logs": ["Degradation safety limit reached"]},
        current_topology,
        degradation_count,
    )


def _safe_project(
    state: dict[str, Any],
    from_topology: str,
    to_topology: str,
    fallback_topology: str,
) -> tuple[dict[str, Any] | None, str]:
    """
    Safely project state, with fallback to single if needed.
    Returns (projected_state, actual_topology) or (None, failed_topology).
    """
    try:
        projected = project_state(state, from_topology, to_topology)
        is_valid, err = validate_projected_state(projected, to_topology)
        if not is_valid:
            log.warning("Projection validation failed: %s -> %s: %s", from_topology, to_topology, err)
            raise ValueError(err)
        return projected, to_topology
    except ValueError:
        # Unknown projection edge — try single
        try:
            to_topology = "single"
            projected = project_state(state, from_topology, to_topology)
            is_valid, err = validate_projected_state(projected, to_topology)
            if not is_valid:
                log.warning("Fallback projection validation failed: %s -> single: %s", from_topology, err)
                raise ValueError(err)
            return projected, to_topology
        except ValueError:
            # Try to continue with current topology instead of failing
            log.warning(
                "All projections failed: %s -> %s and %s -> single. "
                "Attempting to continue with current topology %s",
                from_topology, to_topology, from_topology, fallback_topology,
            )
            # Return minimal projection that keeps current topology
            # Include all fields required by nodes to avoid KeyError crashes
            return {
                "topology": fallback_topology,
                "task": state.get("task"),
                "task_id": state.get("task_id"),
                "decision": state.get("decision"),
                "prior_context": state.get("prior_context"),
                "consumed_cost": state.get("consumed_cost", 0.0),
                "consumed_tokens": state.get("consumed_tokens", 0),
            }, fallback_topology


def _check_budget_after_projection(
    projected: dict[str, Any],
    degradation_count: int,
) -> dict[str, Any] | None:
    """
    Check if budget is exhausted after projection.
    Returns _build_result dict if should stop, None if should continue.
    """
    budget = projected.get("budget")
    if budget is None:
        return None

    acc_cost = projected.get("consumed_cost", 0.0)
    if budget.max_cost_usd <= 0:
        return None

    spent_pct = acc_cost / budget.max_cost_usd
    if spent_pct >= _BUDGET_EXHAUSTED_THRESHOLD:
        log.warning(
            "Budget exhausted after projection: %.1f%% spent (%.6f / %.4f)",
            spent_pct * 100, acc_cost, budget.max_cost_usd,
        )
        return {
            "status": "degraded_completion",
            "final_result": projected.get("final_output"),
            "final_output": projected.get("final_output"),
            "topology": projected.get("topology", "unknown"),
            "degradation_count": degradation_count,
            "consumed_tokens": projected.get("consumed_tokens", 0),
            "consumed_cost": acc_cost,
            "logs": [f"Budget exhausted after projection: {spent_pct*100:.1f}% spent"],
        }
    return None


def _sync_budget_tracker(projected: dict[str, Any]) -> None:
    """Sync BudgetTracker with projected state values."""
    budget = projected.get("budget")
    if budget is not None:
        budget.consumed_cost = projected.get("consumed_cost", 0.0)
        budget.consumed_tokens = projected.get("consumed_tokens", 0)
        projected["budget"] = budget


def _get_budget_remaining(projected: dict[str, Any]) -> float:
    """Get remaining budget percentage from projected state."""
    budget = projected.get("budget")
    if budget is None or budget.max_cost_usd <= 0:
        return 0.0
    acc_cost = projected.get("consumed_cost", 0.0)
    return max(0.0, 100.0 * (1.0 - acc_cost / budget.max_cost_usd))


def _build_result(
    result: dict[str, Any], topology: str, degradation_count: int,
    checkpoint_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the final result dict from graph output.

    Args:
        result: The graph output dict.
        topology: The final topology name.
        degradation_count: Number of degradation steps that occurred.
        checkpoint_state: The latest checkpoint state (for accurate cost).
    """
    final_output = result.get("final_output") or result.get("final_result")
    status = result.get("status", "completed")

    # If we degraded and status is still running, mark as degraded completion
    if degradation_count > 0 and status not in ("completed", "failed"):
        status = "degraded_completion"

    # Use checkpoint state for cumulative cost if available
    if checkpoint_state is not None:
        consumed_tokens = checkpoint_state.get("consumed_tokens", result.get("consumed_tokens", 0))
        consumed_cost = checkpoint_state.get("consumed_cost", result.get("consumed_cost", 0.0))
    else:
        consumed_tokens = result.get("consumed_tokens", 0)
        consumed_cost = result.get("consumed_cost", 0.0)

    return {
        "status": status,
        "final_result": final_output,
        "final_output": final_output,
        "judge_output": result.get("judge_output"),
        "topology": topology,
        "degradation_count": degradation_count,
        "consumed_tokens": consumed_tokens,
        "consumed_cost": consumed_cost,
        "quality_degradation": _estimate_quality_degradation(result, degradation_count),
        "logs": result.get("logs", []),
    }


def _estimate_quality_degradation(
    result: dict[str, Any], degradation_count: int
) -> dict[str, Any] | None:
    """
    Estimate quality impact of degradation.
    Returns quality metrics or None if no degradation occurred.
    """
    if degradation_count == 0:
        return None

    # Estimate quality based on topology capabilities
    topology = result.get("topology", "single")
    # Single topology has lowest capability
    topology_quality_factor = {
        "ensemble": 1.0,
        "fanout": 0.8,
        "supervisor": 0.7,
        "pipeline": 0.6,
        "single": 0.5,
    }.get(topology, 0.5)

    # Each degradation step reduces quality by ~15%
    degradation_factor = max(0.3, 1.0 - (degradation_count * 0.15))

    estimated_quality = topology_quality_factor * degradation_factor

    return {
        "estimated_quality": round(estimated_quality, 2),
        "topology_quality_factor": topology_quality_factor,
        "degradation_factor": round(degradation_factor, 2),
        "degradation_count": degradation_count,
    }
