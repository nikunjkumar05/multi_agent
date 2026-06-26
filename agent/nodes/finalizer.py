from agent.state import AgentState


def _is_subset(a: str, b: str) -> bool:
    a_norm = a.lower().strip()
    b_norm = b.lower().strip()
    return a_norm in b_norm or b_norm in a_norm


def finalize_result(state: AgentState) -> dict:
    errors = state.get("errors", [])
    step_results = state.get("step_results", {})

    if not step_results:
        return {
            "status": "failed" if errors else "completed",
            "final_result": state.get("final_result") or "",
        }

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

    return {
        "status": "failed" if errors else "completed",
        "final_result": combined,
    }
