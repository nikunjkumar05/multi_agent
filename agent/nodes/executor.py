from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AgentState
from agent.tools.registry import registry
from core.llm import create_llm

EXECUTOR_SYSTEM = """You are a step executor. Complete the given step using available tools.
After using a tool, provide the final result as plain text.
Always produce a result — never return empty output."""


def execute_step(state: AgentState) -> dict:
    idx = state.get("current_step_index", 0)
    steps = state.get("steps", [])
    if idx >= len(steps):
        return {"status": "completed", "errors": state.get("errors", []) + ["No steps to execute"]}

    step = dict(steps[idx])
    step["status"] = "running"

    tier = state["decision"].model_tiers.get("executor", "standard")
    llm = create_llm(tier)

    messages = [
        SystemMessage(content=EXECUTOR_SYSTEM),
        HumanMessage(content=f"Step {step['step_id']}: {step['description']}\n\nTask: {state['task']}\n\nTools available: {registry.list_names()}"),
    ]

    result = llm.invoke(messages)
    output = result.content if isinstance(result.content, str) else str(result.content)

    step["status"] = "completed"
    step["result"] = output

    updated_steps = list(steps)
    updated_steps[idx] = step

    step_results = dict(state.get("step_results", {}))
    step_results[step["step_id"]] = output

    return {
        "steps": updated_steps,
        "step_results": step_results,
        "current_step_index": idx + 1,
        "status": "executing",
        "logs": state.get("logs", []) + [f"Completed step {step['step_id']}"],
    }