from agent.state import AgentState


def _is_subset(a: str, b: str) -> bool:
    a_norm = a.lower().strip()
    b_norm = b.lower().strip()
    return a_norm in b_norm or b_norm in a_norm


def _build_degradation_summary(state: AgentState) -> dict | None:
    """Build a transparency summary of what was skipped or degraded."""
    issues = []

    # Check for skipped steps
    step_results = state.get("step_results", {})
    skipped_steps = [k for k, v in step_results.items() if isinstance(v, str) and "[Step skipped" in v]
    if skipped_steps:
        issues.append({
            "type": "steps_skipped",
            "detail": f"{len(skipped_steps)} step(s) skipped due to budget exhaustion",
            "step_ids": skipped_steps,
            "impact": "Incomplete execution — some planned work was not performed",
        })

    # Check for truncated results
    steps = state.get("steps") or state.get("plan_steps") or []
    truncated_steps = [s.get("step_id") for s in steps if s.get("truncated")]
    if truncated_steps:
        issues.append({
            "type": "steps_truncated",
            "detail": f"{len(truncated_steps)} step(s) truncated due to budget cap",
            "step_ids": truncated_steps,
            "impact": "Output may be incomplete — results were cut short",
        })

    # Check for skipped validation
    val_skipped = state.get("validation_skipped", False)
    if val_skipped:
        reason = state.get("validation_skip_reason", "unknown")
        issues.append({
            "type": "validation_skipped",
            "detail": f"Quality validation was skipped ({reason})",
            "impact": "No quality review was performed — result may contain errors",
        })

    # Check for topology degradation
    topology_history = state.get("topology_history", [])
    if topology_history:
        degradations = [h for h in topology_history if isinstance(h, dict) and "from" in h]
        if degradations:
            chain = " → ".join([f"{d.get('from', '?')}→{d.get('to', '?')}" for d in degradations])
            issues.append({
                "type": "topology_degraded",
                "detail": f"Topology was degraded: {chain}",
                "impact": "Execution used a simpler agent topology than originally planned",
            })

    # Check for skipped judge
    skip_judge = state.get("skip_judge", False)
    if skip_judge:
        issues.append({
            "type": "judge_skipped",
            "detail": "Final quality review (Judge) was skipped due to budget exhaustion",
            "impact": "Result was not reviewed or curated by the Judge agent",
        })

    if not issues:
        return None

    return {
        "degraded": True,
        "issue_count": len(issues),
        "issues": issues,
        "warning": "This result was produced under budget degradation. Quality may be reduced.",
    }


def finalize_result(state: AgentState) -> dict:
    errors = state.get("errors", [])
    step_results = state.get("step_results", {})
    steps = state.get("steps") or state.get("plan_steps") or []

    if not step_results:
        return {
            "status": "failed" if errors else "completed",
            "final_output": state.get("final_output") or state.get("final_result") or "",
            "final_result": state.get("final_output") or state.get("final_result") or "",
        }

    # Check if all planned steps have results — completed tasks beat retry errors
    all_done = len(step_results) >= len(steps) if steps else len(step_results) > 0
    status = "completed" if all_done else "failed"

    results_list = list(step_results.values())

    if len(results_list) == 1:
        combined = results_list[0]
    else:
        best = max(results_list, key=len)
        others = [r for r in results_list if r != best]
        unique_others = [r for r in others if not _is_subset(r, best)]

        if not unique_others:
            combined = best
        else:
            combined = best + "\n\n---\n\n" + "\n\n".join(unique_others)

    # Build degradation summary for transparency
    degradation_summary = _build_degradation_summary(state)

    # Prepend warning to final result if degraded
    if degradation_summary:
        warning = (
            "⚠️ DEGRADED RESULT — This output was produced under budget constraints.\n"
            f"{degradation_summary['warning']}\n"
            f"Issues: {degradation_summary['issue_count']}\n"
            "---\n\n"
        )
        combined = warning + combined

    result = {
        "status": status,
        "final_output": combined,
        "final_result": combined,
    }

    # Attach degradation summary as structured metadata
    if degradation_summary:
        result["degradation_summary"] = degradation_summary

    return result
