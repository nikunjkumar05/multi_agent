"""
State projection functions for mid-execution topology degradation.

All projections are pure Python, $0 LLM cost.
Projection functions must NOT include annotated fields (step_results,
completed_step_ids, candidate_outputs, errors, logs, topology_history)
— these are preserved automatically by the checkpointer.
"""

from __future__ import annotations

from typing import Any, Callable

# Fields annotated with Annotated[type, reducer] in AgentState.
# The checkpointer preserves these automatically — projections must NEVER set them.
_ANNOTATED_FIELDS = frozenset({
    "completed_step_ids",
    "step_results",
    "candidate_outputs",
    "consumed_tokens",
    "consumed_cost",
    "topology_history",
    "errors",
    "logs",
})


def _assert_no_annotated_fields(result: dict[str, Any], edge: str) -> dict[str, Any]:
    """Assert that a projection result contains no annotated fields."""
    leaked = _ANNOTATED_FIELDS & result.keys()
    if leaked:
        raise ValueError(
            f"Projection {edge} leaked annotated fields: {sorted(leaked)}. "
            f"These are preserved by the checkpointer and must not be set in projection output."
        )
    return result

# ── Projection Functions ──────────────────────────────────────────────

def project_ensemble_to_fanout(state: dict[str, Any]) -> dict[str, Any]:
    """ensemble → fanout: carry agent results as prior_context, clear agent-specific keys."""
    agent_results = []
    for key in ("agent_a_result", "agent_b_result", "agent_c_result"):
        result = state.get(key)
        if result and isinstance(result, dict):
            output = result.get("output", "")
            if output:
                agent_results.append(output)

    prior_context = None
    if agent_results:
        prior_context = "Ensemble agent outputs from prior execution:\n" + "\n\n---\n\n".join(agent_results)

    return _assert_no_annotated_fields({
        "topology": "fanout",
        "prior_context": prior_context,
        "_worker_assignments": None,
        "fanout_worker_results": None,
        "agent_a_result": None,
        "agent_b_result": None,
        "agent_c_result": None,
    }, "ensemble→fanout")


def project_ensemble_to_supervisor(state: dict[str, Any]) -> dict[str, Any]:
    """ensemble → supervisor: chain through fanout, preserving agent results."""
    fanout_state = project_ensemble_to_fanout(state)
    return _assert_no_annotated_fields({
        **fanout_state,
        "topology": "supervisor",
        "supervisor_remaining_tasks": None,
        "supervisor_completed_tasks": None,
    }, "ensemble→supervisor")


def project_ensemble_to_pipeline(state: dict[str, Any]) -> dict[str, Any]:
    """ensemble → pipeline: chain through fanout→supervisor, preserving agent results."""
    supervisor_state = project_ensemble_to_supervisor(state)
    return _assert_no_annotated_fields({
        **supervisor_state,
        "topology": "pipeline",
    }, "ensemble→pipeline")


def project_ensemble_to_single(state: dict[str, Any]) -> dict[str, Any]:
    """ensemble → single: deterministic scoring + refinement handoff."""
    candidates = state.get("candidate_outputs", {})
    best_candidate = _select_best_candidate(candidates)

    prior_context = None
    if best_candidate:
        prior_context = f"Prior ensemble output:\n{best_candidate.get('output', '')}"

    return _assert_no_annotated_fields({
        "topology": "single",
        "prior_context": prior_context,
        "agent_a_result": None,
        "agent_b_result": None,
        "agent_c_result": None,
    }, "ensemble→single")


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

    return _assert_no_annotated_fields({
        "topology": "supervisor",
        "prior_context": prior_context,
        "supervisor_remaining_tasks": remaining_tasks if remaining_tasks else None,
        "supervisor_completed_tasks": completed_steps if completed_steps else None,
        "_worker_assignments": None,
        "fanout_worker_results": None,
    }, "fanout→supervisor")


def project_fanout_to_pipeline(state: dict[str, Any]) -> dict[str, Any]:
    """fanout → pipeline: chain through supervisor."""
    supervisor_state = project_fanout_to_supervisor(state)
    return _assert_no_annotated_fields({
        **supervisor_state,
        "topology": "pipeline",
    }, "fanout→pipeline")


def project_fanout_to_single(state: dict[str, Any]) -> dict[str, Any]:
    """fanout → single: aggregate completed, discard rest."""
    step_results = state.get("step_results", {})
    completed = {k: v for k, v in step_results.items() if v is not None}

    prior_context = None
    if completed:
        summaries = [f"Step {k}: {v}" for k, v in completed.items()]
        prior_context = f"Partially completed work:\n" + "\n".join(summaries)

    return _assert_no_annotated_fields({
        "topology": "single",
        "prior_context": prior_context,
        "_worker_assignments": None,
        "fanout_worker_results": None,
    }, "fanout→single")


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

    return _assert_no_annotated_fields({
        "topology": "pipeline",
        "prior_context": prior_context,
        "supervisor_remaining_tasks": None,
        "supervisor_completed_tasks": None,
    }, "supervisor→pipeline")


def project_supervisor_to_single(state: dict[str, Any]) -> dict[str, Any]:
    """supervisor → single: chain through pipeline."""
    pipeline_state = project_supervisor_to_pipeline(state)
    return _assert_no_annotated_fields({
        **pipeline_state,
        "topology": "single",
    }, "supervisor→single")


def project_pipeline_to_single(state: dict[str, Any]) -> dict[str, Any]:
    """pipeline → single: carry forward completed step results as prior_context."""
    step_results = state.get("step_results", {})
    completed = {k: v for k, v in step_results.items() if v is not None}

    prior_context = None
    if completed:
        summaries = [f"Step {k}: {v}" for k, v in completed.items()]
        prior_context = "Pipeline work completed before degradation:\n" + "\n".join(summaries)

    return _assert_no_annotated_fields({
        "topology": "single",
        "prior_context": prior_context,
    }, "pipeline→single")


def project_feedback_to_single(state: dict[str, Any]) -> dict[str, Any]:
    """feedback → single: carry forward latest output as prior_context."""
    step_results = state.get("step_results", {})
    steps = state.get("steps", [])

    prior_context = None
    if steps:
        last_step_id = steps[-1]["step_id"]
        last_output = step_results.get(last_step_id, "")
        if last_output:
            prior_context = f"Feedback iteration output:\n{last_output}"

    return _assert_no_annotated_fields({
        "topology": "single",
        "prior_context": prior_context,
        "feedback_iteration": 0,
        "critic_accepted": False,
        "critic_feedback": None,
    }, "feedback→single")


def project_any_to_feedback(state: dict[str, Any], from_topology: str) -> dict[str, Any]:
    """any → feedback: carry forward completed work as prior_context."""
    step_results = state.get("step_results", {})
    completed = {k: v for k, v in step_results.items() if v is not None}

    prior_context = None
    if completed:
        summaries = [f"Step {k}: {v}" for k, v in completed.items()]
        prior_context = f"Prior work from {from_topology}:\n" + "\n".join(summaries)

    return _assert_no_annotated_fields({
        "topology": "feedback",
        "prior_context": prior_context,
        "feedback_iteration": 0,
        "critic_accepted": False,
        "critic_feedback": None,
    }, f"{from_topology}→feedback")


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
    ("feedback", "single"): project_feedback_to_single,
    # Feedback can be reached from any topology
    ("ensemble", "feedback"): lambda s: project_any_to_feedback(s, "ensemble"),
    ("fanout", "feedback"): lambda s: project_any_to_feedback(s, "fanout"),
    ("supervisor", "feedback"): lambda s: project_any_to_feedback(s, "supervisor"),
    ("pipeline", "feedback"): lambda s: project_any_to_feedback(s, "pipeline"),
}


def project_state(state: dict[str, Any], from_topology: str, to_topology: str) -> dict[str, Any]:
    """
    Project state from one topology to another.
    Returns a partial state update (no annotated fields).
    topology_history is annotated with operator.add and preserved automatically
    by the checkpointer — do NOT include it in projection output.
    """
    key = (from_topology, to_topology)
    proj_fn = _PROJECTIONS.get(key)
    if proj_fn is None:
        raise ValueError(f"Unknown projection edge: {from_topology} → {to_topology}")

    projected = proj_fn(state)

    # topology_history is annotated — preserved by checkpointer, do NOT set it here

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


# ── State Validation ─────────────────────────────────────────────────

_TOPOLOGY_REQUIRED_FIELDS: dict[str, list[str]] = {
    "single": ["topology"],
    "pipeline": ["topology"],
    "supervisor": ["topology"],
    "fanout": ["topology"],
    "ensemble": ["topology"],
    "feedback": ["topology"],
}

_TOPOLOGY_OPTIONAL_FIELDS: dict[str, list[str]] = {
    "supervisor": ["supervisor_remaining_tasks", "supervisor_completed_tasks"],
    "fanout": ["_worker_assignments", "fanout_worker_results"],
    "ensemble": ["agent_a_result", "agent_b_result", "agent_c_result"],
    "feedback": ["feedback_iteration", "critic_accepted", "critic_feedback"],
}


def validate_projected_state(projected: dict[str, Any], to_topology: str) -> tuple[bool, str]:
    """
    Validate that a projected state is valid for the target topology.
    Returns (is_valid, error_message).
    Catches incomplete projections that would crash the new graph.
    """
    if not isinstance(projected, dict):
        return False, "Projected state is not a dict"

    topology = projected.get("topology")
    if topology is None:
        return False, "Projected state missing 'topology' field"

    if topology != to_topology:
        return False, f"Projected topology mismatch: expected {to_topology}, got {topology}"

    required = _TOPOLOGY_REQUIRED_FIELDS.get(to_topology, ["topology"])
    for field in required:
        if field not in projected:
            return False, f"Projected state missing required field '{field}' for {to_topology}"

    return True, ""