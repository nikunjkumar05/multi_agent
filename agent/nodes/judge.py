import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent.nodes.executor import _extract_text
from agent.state import AgentState
from core.llm import create_llm, estimate_cost, estimate_tokens
from core.node_events import emit_event

log = logging.getLogger(__name__)

JUDGE_SYSTEM = """You are a judge agent. You receive a task, executor output, and validator assessment.
Your job is to produce the BEST possible final result.

You can:
1. Accept the executor's output if it's good enough
2. Improve it based on the validator's concerns
3. Rewrite from scratch if both are wrong

Return the BEST final result as plain text. Be concise and accurate."""

_JUDGE_TIMEOUT = 90


async def ensemble_judge(state: AgentState) -> dict:
    step_results = state.get("step_results", {})

    # Extract parallel agent outputs (e.g. "a", "b", "c" from ensemble, or "agent_*" keys)
    agent_outputs = []
    for k, v in step_results.items():
        if isinstance(k, str) and (k.startswith("agent_") or len(k) == 1):
            agent_name = f"Agent {k.upper()}" if len(k) == 1 else k.replace("agent_", "Agent ").upper()
            agent_outputs.append(f"{agent_name} output:\n{v}")

    if agent_outputs:
        executor_outputs_str = "\n\n".join(agent_outputs)
    else:
        last_step = state.get("steps", [])
        last_result = ""
        if last_step:
            last_step_id = last_step[-1]["step_id"]
            last_result = step_results.get(last_step_id, "")
        executor_outputs_str = f"Executor output:\n{last_result}"

    # If all agent outputs are error messages, don't call LLM — pick the best one
    all_failed = all(
        isinstance(v, str) and v.startswith("[Agent") and v.endswith("]")
        for v in step_results.values()
        if isinstance(v, str)
    )

    task_id = state.get("task_id", "")

    if all_failed or not executor_outputs_str.strip():
        # All agents failed — pick the least-bad result as final output
        best = max(step_results.values(), key=lambda x: len(str(x))) if step_results else "No output produced"
        await emit_event(task_id, "judge_completed", {
            "result_preview": str(best)[:200],
            "tokens_used": 0,
            "cost_usd": 0,
            "budget_spent_pct": 0,
        })
        return {
            "judge_output": str(best),
            "final_output": str(best),
            "final_result": str(best),
            "status": "completed",
            "logs": ["Judge: all agents failed, using best available output"],
        }

    validator_conf = state.get("validator_confidence") or 1.0
    diverged = state.get("reasoning_diverged", False)

    tier = state["decision"].model_tiers.get("judge", "frontier")
    llm = create_llm(tier, temperature=0.3)

    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=(
            f"Task: {state['task']}\n\n"
            f"{executor_outputs_str}\n\n"
            f"Validator confidence: {validator_conf:.2f}\n"
            f"Reasoning diverged: {diverged}\n\n"
            f"Produce the best final result."
        )),
    ]

    try:
        response = await asyncio.wait_for(
            llm.ainvoke(messages),
            timeout=_JUDGE_TIMEOUT,
        )
        judge_output = _extract_text(response.content)

        if not judge_output or not judge_output.strip():
            judge_output = executor_outputs_str
    except asyncio.TimeoutError:
        log.warning("Judge timed out after %ds", _JUDGE_TIMEOUT)
        judge_output = executor_outputs_str
        response = None
    except Exception as e:
        log.warning("Judge failed: %s", e)
        judge_output = executor_outputs_str
        response = None

    budget = state.get("budget")
    if budget and response is not None:
        try:
            budget.record_usage(
                tokens=estimate_tokens(response),
                cost=estimate_cost(response, tier),
            )
        except Exception:
            pass

    await emit_event(task_id, "judge_completed", {
        "result_preview": str(judge_output)[:200],
        "tokens_used": budget.consumed_tokens if budget else 0,
        "cost_usd": round(budget.consumed_cost, 6) if budget else 0,
        "budget_spent_pct": round(budget.spent_pct, 1) if budget else 0,
    })

    return {
        "judge_output": judge_output,
        "final_output": judge_output,
        "final_result": judge_output,
        "status": "completed",
        "logs": ["Judge produced final result"],
    }
