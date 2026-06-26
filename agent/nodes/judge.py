from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AgentState
from core.llm import create_llm, estimate_cost, estimate_tokens
from core.node_events import emit_event

JUDGE_SYSTEM = """You are a judge agent. You receive a task, executor output, and validator assessment.
Your job is to produce the BEST possible final result.

You can:
1. Accept the executor's output if it's good enough
2. Improve it based on the validator's concerns
3. Rewrite from scratch if both are wrong

Return the BEST final result as plain text. Be concise and accurate."""


async def ensemble_judge(state: AgentState) -> dict:
    step_results = state.get("step_results", {})
    last_step = state.get("steps", [])
    last_result = ""
    if last_step:
        last_step_id = last_step[-1]["step_id"]
        last_result = step_results.get(last_step_id, "")

    validator_conf = state.get("validator_confidence", 1.0)
    diverged = state.get("reasoning_diverged", False)

    tier = state["decision"].model_tiers.get("judge", "frontier")
    llm = create_llm(tier, temperature=0.3)

    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=(
            f"Task: {state['task']}\n\n"
            f"Executor output:\n{last_result}\n\n"
            f"Validator confidence: {validator_conf:.2f}\n"
            f"Reasoning diverged: {diverged}\n\n"
            f"Produce the best final result."
        )),
    ]

    response = llm.invoke(messages)

    budget = state.get("budget")
    if budget:
        budget.record_usage(
            tokens=estimate_tokens(response),
            cost=estimate_cost(response, tier),
        )

    judge_output = response.content if isinstance(response.content, str) else str(response.content)

    task_id = state.get("task_id", "")
    await emit_event(task_id, "judge_completed", {
        "result_preview": str(judge_output)[:200],
    })

    return {
        "judge_output": judge_output,
        "final_result": judge_output,
        "status": "completed",
        "logs": state.get("logs", []) + ["Judge produced final result"],
    }
