"""
State projection functions for mid-execution topology degradation.

All projections are pure Python, $0 LLM cost.
Projection functions must NOT include annotated fields (step_results,
completed_step_ids, candidate_outputs, errors, logs, topology_history)
— these are preserved automatically by the checkpointer.
"""

from __future__ import annotations

from typing import Any, Callable

# ── Projection Functions ──────────────────────────────────────────────

def project_ensemble_to_fanout(state: dict[str, Any]) -> dict[str, Any]:
    """ensemble → fanout: clear agent-specific keys. step_results preserved by checkpointer."""
    return {
        "topology": "fanout",
        "_worker_assignments": None,
        "fanout_worker_results": None,
        "agent_a_result": None,
        "agent_b_result": None,
        "agent_c_result": None,
    }


def project_ensemble_to_supervisor(state: dict[str, Any]) -> dict[str, Any]:
    """ensemble → supervisor: chain through fanout."""
    fanout_state = project_ensemble_to_fanout(state)
    return {
        **fanout_state,
        "topology": "supervisor",
        "supervisor_remaining_tasks": None,
        "supervisor_completed_tasks": None,
    }


def project_ensemble_to_pipeline(state: dict[str, Any]) -> dict[str, Any]:
    """ensemble → pipeline: chain through fanout→supervisor."""
    supervisor_state = project_ensemble_to_supervisor(state)
    return {
        **supervisor_state,
        "topology": "pipeline",
    }


def project_ensemble_to_single(state: dict[str, Any]) -> dict[str, Any]:
    """ensemble → single: deterministic scoring + refinement handoff."""
    candidates = state.get("candidate_outputs", {})
    best_candidate = _select_best_candidate(candidates)

    prior_context = None
    if best_candidate:
        prior_context = f"Prior ensemble output:\n{best_candidate.get('output', '')}"

    return {
        "topology": "single",
        "prior_context": prior_context,
        "agent_a_result": None,
        "agent_b_result": None,
        "agent_c_result": None,
    }


def project_fanout_to_supervisor(state: dict[str, Any]) -> dict[str, Any]:
    """fanout → supervisor: queue collapse engine."""
    step_results = state.get("step_results", {})
    plan_steps = state.get("plan_steps", [])

    completed_steps = []
    remaining_tasks = []

    for i, step in enumerate(plan_steps):
        step_id = step.get("step_id", i)
        if step_id in step_results and step_results[step_id] is not None:
            completed_steps.append(step)
        else:
            remaining_tasks.append(step.get("description", f"Step {step_id}"))

    prior_context = None
    if completed_steps:
        summaries = [f"Step {s.get('step_id')}: {s.get('result', 'done')}" for s in completed_steps]
        prior_context = f"Previously completed steps:\n" + "\n".join(summaries)

    return {
        "topology": "supervisor",
        "prior_context": prior_context,
        "supervisor_remaining_tasks": remaining_tasks if remaining_tasks else None,
        "supervisor_completed_tasks": completed_steps if completed_steps else None,
        "_worker_assignments": None,
        "fanout_worker_results": None,
    }


def project_fanout_to_pipeline(state: dict[str, Any]) -> dict[str, Any]:
    """fanout → pipeline: chain through supervisor."""
    supervisor_state = project_fanout_to_supervisor(state)
    return {
        **supervisor_state,
        "topology": "pipeline",
    }


def project_fanout_to_single(state: dict[str, Any]) -> dict[str, Any]:
    """fanout → single: aggregate completed, discard rest."""
    step_results = state.get("step_results", {})
    completed = {k: v for k, v in step_results.items() if v is not None}

    prior_context = None
    if completed:
        summaries = [f"Step {k}: {v}" for k, v in completed.items()]
        prior_context = f"Partially completed work:\n" + "\n".join(summaries)

    return {
        "topology": "single",
        "prior_context": prior_context,
        "_worker_assignments": None,
        "fanout_worker_results": None,
    }


def project_supervisor_to_pipeline(state: dict[str, Any]) -> dict[str, Any]:
    """supervisor → pipeline: flatten supervisor queue."""
    remaining = state.get("supervisor_remaining_tasks") or []
    completed = state.get("supervisor_completed_tasks") or []

    prior_context = None
    if completed or remaining:
        parts = []
        if completed:
            parts.append("Completed: " + ", ".join([t.get("description", "") for t in completed]))
        if remaining:
            parts.append("Remaining: " + ", ".join(remaining))
        prior_context = "\n".join(parts)

    return {
        "topology": "pipeline",
        "prior_context": prior_context,
        "supervisor_remaining_tasks": None,
        "supervisor_completed_tasks": None,
    }


def project_supervisor_to_single(state: dict[str, Any]) -> dict[str, Any]:
    """supervisor → single: chain through pipeline."""
    pipeline_state = project_supervisor_to_pipeline(state)
    return {
        **pipeline_state,
        "topology": "single",
    }


def project_pipeline_to_single(state: dict[str, Any]) -> dict[str, Any]:
    """pipeline → single: trivial — topology_history update only."""
    return {
        "topology": "single",
    }


# ── Dispatch Table ────────────────────────────────────────────────────

_PROJECTIONS: dict[tuple[str, str], Callable[[dict[str, Any]], dict[str, Any]]] = {
    ("ensemble", "fanout"): project_ensemble_to_fanout,
    ("ensemble", "supervisor"): project_ensemble_to_supervisor,
    ("ensemble", "pipeline"): project_ensemble_to_pipeline,
    ("ensemble", "single"): project_ensemble_to_single,
    ("fanout", "supervisor"): project_fanout_to_supervisor,
    ("fanout", "pipeline"): project_fanout_to_pipeline,
    ("fanout", "single"): project_fanout_to_single,
    ("supervisor", "pipeline"): project_supervisor_to_pipeline,
    ("supervisor", "single"): project_supervisor_to_single,
    ("pipeline", "single"): project_pipeline_to_single,
}


def project_state(state: dict[str, Any], from_topology: str, to_topology: str) -> dict[str, Any]:
    """
    Project state from one topology to another.
    Returns a partial state update (no annotated fields).
    """
    key = (from_topology, to_topology)
    proj_fn = _PROJECTIONS.get(key)
    if proj_fn is None:
        raise ValueError(f"Unknown projection edge: {from_topology} → {to_topology}")

    projected = proj_fn(state)

    # Add topology_history entry
    projected["topology_history"] = [
        {"from": from_topology, "to": to_topology}
    ]

    return projected


def get_valid_projection_edges() -> list[tuple[str, str]]:
    """Return all valid (from, to) pairs."""
    return list(_PROJECTIONS.keys())


# ── Helpers ───────────────────────────────────────────────────────────

def _select_best_candidate(candidates: dict[str, Any]) -> dict[str, Any] | None:
    """
    Deterministic scoring for ensemble→single:
    Score = 0.5×Confidence - 0.3×ErrorRate + 0.2×StructureCompleteness
    """
    if not candidates:
        return None

    best = None
    best_score = -1.0

    for key, candidate in candidates.items():
        if not isinstance(candidate, dict):
            continue

        output = candidate.get("output", "")
        confidence = candidate.get("confidence", 0.5)
        tool_calls = candidate.get("tool_calls_count", 0)
        tool_errors = candidate.get("tool_errors_count", 0)

        error_rate = tool_errors / tool_calls if tool_calls > 0 else 0.0
        structure = _check_structure(output)

        score = 0.5 * confidence - 0.3 * error_rate + 0.2 * structure

        if score > best_score:
            best_score = score
            best = candidate

    return best


def _check_structure(text: str) -> float:
    """Check for structural completeness: code blocks, headers, JSON, lists."""
    if not text:
        return 0.0

    score = 0.0
    if "```" in text:
        score += 0.3
    if any(text.startswith(p) for p in ["#", "##"]):
        score += 0.2
    if text.strip().startswith("{") or text.strip().startswith("["):
        score += 0.2
    if any(f"{i}." in text for i in range(1, 6)):
        score += 0.1
    if len(text) > 100:
        score += 0.2

    return min(score, 1.0)