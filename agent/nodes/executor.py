from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AgentState
from agent.tools.registry import registry
from core.llm import create_llm
from core.node_events import emit_event

EXECUTOR_SYSTEM = """You are a step executor. Complete the given step using available tools.
After using a tool, provide the final result as plain text.
Always produce a result — never return empty output.

If previous steps have already produced output, build on it — do NOT repeat the same work.
Only include new/changed content in your result."""


async def execute_step(state: AgentState) -> dict:
    idx = state.get("current_step_index", 0)
    steps = state.get("steps", [])
    if idx >= len(steps):
        return {"status": "completed", "errors": state.get("errors", []) + ["No steps to execute"]}

    step = dict(steps[idx])
    step["status"] = "running"

    task_id = state.get("task_id", "")
    await emit_event(task_id, "step_started", {
        "step_id": step["step_id"],
        "description": step["description"],
    })

    tier = state["decision"].model_tiers.get("executor", "standard")
    llm = create_llm(tier)

    previous_results = state.get("step_results", {})
    context_block = ""
    if previous_results:
        parts = []
        for sid, res in previous_results.items():
            preview = str(res)[:500]
            parts.append(f"Step {sid} result: {preview}")
        context_block = "\n\nPrevious step results:\n" + "\n".join(parts)

    messages = [
        SystemMessage(content=EXECUTOR_SYSTEM),
        HumanMessage(content=(
            f"Step {step['step_id']}: {step['description']}\n\n"
            f"Task: {state['task']}\n"
            f"Tools available: {registry.list_names()}"
            f"{context_block}"
        )),
    ]

    result = llm.invoke(messages)
    output = result.content if isinstance(result.content, str) else str(result.content)

    step["status"] = "completed"
    step["result"] = output

    updated_steps = list(steps)
    updated_steps[idx] = step

    step_results = dict(state.get("step_results", {}))
    step_results[step["step_id"]] = output

    await emit_event(task_id, "step_completed", {
        "step_id": step["step_id"],
        "result_preview": str(output)[:200],
    })

    return {
        "steps": updated_steps,
        "step_results": step_results,
        "current_step_index": idx + 1,
        "status": "executing",
        "logs": state.get("logs", []) + [f"Completed step {step['step_id']}"],
    }
