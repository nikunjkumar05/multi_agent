from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from agent.state import AgentState
from agent.tools.registry import registry
from core.llm import create_llm, estimate_cost, estimate_tokens
from core.node_events import emit_event

MAX_TOOL_ITERATIONS = 10
MAX_RETRIES = 2


def _extract_text(content) -> str:
    """Extract plain text from LangChain content blocks.

    Mistral API returns content as a list of blocks like:
        [{'type': 'text', 'text': '...'}, {'type': 'reference', ...}]
    This extracts just the text portions.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts) if texts else str(content)
    return str(content)

TASK_TYPE_PROMPTS = {
    "code": (
        "You are an expert software engineer. Write clean, correct, well-structured code.\n"
        "Include error handling where appropriate. Follow best practices for the language.\n"
        "Use the code_executor tool to run and verify your code before returning the final result.\n"
        "If the code fails, fix the errors and try again.\n\n"
        "CRITICAL RULES FOR CODE EXECUTION:\n"
        "- Do NOT use input() — it will timeout. Use hardcoded test values instead.\n"
        "- For CLI scripts with argparse, pass test arguments using the args parameter, e.g. args=[\"-l\", \"12\"].\n"
        "- Do NOT call code_executor with the same code if it already failed.\n"
        "- After code runs successfully ONCE, stop calling tools and return the final code as text.\n"
        "- Always include print() to show output — bare expressions produce no output."
    ),
    "math": (
        "You are a mathematical expert. Show clear, step-by-step reasoning.\n"
        "Use the code_executor tool for any numerical computations to avoid arithmetic errors.\n"
        "State your answer clearly at the end."
    ),
    "research": (
        "You are a thorough researcher. Provide comprehensive, well-organized analysis.\n"
        "Use the web_search tool to find current, accurate information.\n"
        "Cite your sources and present findings clearly."
    ),
    "creative": (
        "You are a creative writer. Focus on quality, originality, and engagement.\n"
        "Write with vivid language, strong structure, and compelling content.\n"
        "Your output should be polished and ready to use."
    ),
    "data": (
        "You are a data analyst. Analyze data systematically and thoroughly.\n"
        "Use the code_executor tool for computations, statistical analysis, and visualizations.\n"
        "Present findings with clear conclusions and recommendations."
    ),
    "general": (
        "You are a helpful assistant. Complete the given step accurately and thoroughly.\n"
        "Use available tools when they would help: code_executor for computations,\n"
        "web_search for current information, file_read/file_write for file operations.\n"
        "Always produce a result — never return empty output."
    ),
}

CODE_KEYWORDS = {"write", "code", "function", "implement", "create", "generate", "build", "develop", "script", "class", "program"}
MATH_KEYWORDS = {"calculate", "solve", "equation", "compute", "sum", "add", "subtract", "multiply", "divide", "what is", "factorial", "square root"}
RESEARCH_KEYWORDS = {"explain", "research", "compare", "why", "how does", "describe", "summarize", "review", "analyze"}
CREATIVE_KEYWORDS = {"story", "poem", "creative", "essay", "narrative", "article", "blog post", "fiction", "novel"}
DATA_KEYWORDS = {"analyze", "data", "dataset", "statistics", "chart", "graph", "visualization"}


def detect_task_type(task: str) -> str:
    task_lower = task.lower()
    if any(kw in task_lower for kw in CREATIVE_KEYWORDS):
        return "creative"
    if any(kw in task_lower for kw in DATA_KEYWORDS):
        return "data"
    if any(kw in task_lower for kw in CODE_KEYWORDS):
        return "code"
    if any(kw in task_lower for kw in MATH_KEYWORDS):
        return "math"
    if any(kw in task_lower for kw in RESEARCH_KEYWORDS):
        return "research"
    return "general"


def _build_executor_prompt(task_type: str) -> str:
    base = TASK_TYPE_PROMPTS.get(task_type, TASK_TYPE_PROMPTS["general"])
    tool_names = registry.list_names()
    return f"{base}\n\nAvailable tools: {', '.join(tool_names)}"


async def execute_step(state: AgentState) -> dict:
    idx = state.get("current_step_index", 0)
    steps = state.get("steps", [])
    if idx >= len(steps):
        return {
            "status": "completed",
            "current_step_index": idx,
            "completed_step_ids": state.get("completed_step_ids", []),
            "errors": state.get("errors", []) + ["No steps to execute"],
        }

    step = dict(steps[idx])
    step["status"] = "running"

    # Check step budget cap before starting
    step_budget_caps = state.get("step_budget_caps", {})
    step_cap = step_budget_caps.get(str(step["step_id"]))
    prev_cost = state.get("consumed_cost", 0.0)
    if step_cap is not None and prev_cost >= step_cap:
        step["status"] = "skipped"
        step["result"] = "[Step skipped - budget cap exceeded]"
        updated_steps = list(steps)
        updated_steps[idx] = step
        task_id = state.get("task_id", "")
        await emit_event(task_id, "step_skipped", {
            "step_id": step["step_id"],
            "reason": "budget_cap_exceeded",
            "spent_pct": round(prev_cost / state["budget"].max_cost_usd * 100, 1) if state.get("budget") and state["budget"].max_cost_usd > 0 else 0,
        })
        return {
            "steps": updated_steps,
            "current_step_index": idx + 1,
            "status": "executing",
            "retry_count": 0,
            "completed_step_ids": [step["step_id"]],
            "logs": [f"Step {step['step_id']} skipped - budget cap exceeded"],
        }

    task_id = state.get("task_id", "")
    step_results = dict(state.get("step_results", {}))

    await emit_event(task_id, "step_started", {
        "step_id": step["step_id"],
        "description": step["description"],
    })

    # Pre-LLM budget check — skip step if budget exhausted
    from core.budget import should_skip_llm
    if should_skip_llm(state):
        budget = state.get("budget")
        spent_pct = round(state.get("consumed_cost", 0.0) / budget.max_cost_usd * 100, 1) if budget and budget.max_cost_usd > 0 else 0
        step["status"] = "skipped"
        step["result"] = "[Step skipped - budget exhausted]"
        updated_steps = list(steps)
        updated_steps[idx] = step
        step_results[step["step_id"]] = step["result"]
        await emit_event(task_id, "step_skipped", {
            "step_id": step["step_id"],
            "reason": "budget_exhausted",
            "spent_pct": spent_pct,
        })
        return {
            "steps": updated_steps,
            "step_results": step_results,
            "current_step_index": idx + 1,
            "status": "executing",
            "retry_count": 0,
            "completed_step_ids": [step["step_id"]],
            "errors": [],
            "logs": [f"Step {step['step_id']} skipped - budget exhausted"],
        }

    tier = state["decision"].model_tiers.get("executor", "standard")
    llm = create_llm(tier)

    task_type = detect_task_type(state["task"])
    system_prompt = _build_executor_prompt(task_type)

    previous_results = state.get("step_results", {})
    context_block = ""
    if previous_results:
        parts = []
        for sid, res in previous_results.items():
            preview = str(res)[:500]
            parts.append(f"Step {sid} result: {preview}")
        context_block = "\n\nPrevious step results:\n" + "\n".join(parts)

    retry_count = state.get("retry_count", 0)
    retry_block = ""
    if retry_count > 0:
        last_error = state.get("errors", ["Unknown error"])[-1] if state.get("errors") else "Unknown error"
        retry_block = f"\n\nNOTE: Previous attempt failed with error: {last_error}\nTry a different approach."

    prior_context = state.get("prior_context")
    prior_block = ""
    if prior_context:
        prior_block = f"\n\nPrior context from previous topology:\n{prior_context}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"Step {step['step_id']}: {step['description']}\n\n"
            f"Task: {state['task']}"
            f"{context_block}"
            f"{retry_block}"
            f"{prior_block}"
        )),
    ]

    budget = state.get("budget")

    try:
        langchain_tools = registry.get_langchain_tools()
        if langchain_tools:
            llm_with_tools = llm.bind_tools(langchain_tools)
        else:
            llm_with_tools = llm

        output, exec_tokens, exec_cost = await _react_loop(llm_with_tools, messages, tier, state)

        prev_tokens = state.get("consumed_tokens", 0)
        prev_cost = state.get("consumed_cost", 0.0)
        total_tokens = prev_tokens + exec_tokens
        total_cost = prev_cost + exec_cost
        budget = state.get("budget")

        # Check if step exceeded its budget cap
        step_budget_caps = state.get("step_budget_caps", {})
        step_cap = step_budget_caps.get(str(step["step_id"]))
        if step_cap is not None and total_cost >= step_cap:
            step["status"] = "completed"
            step["result"] = output[:500] + "\n[Truncated - step budget cap exceeded]" if len(str(output)) > 500 else output
            step["truncated"] = True
            updated_steps = list(steps)
            updated_steps[idx] = step
            step_results[step["step_id"]] = step["result"]
            await emit_event(task_id, "step_truncated", {
                "step_id": step["step_id"],
                "reason": "budget_cap_exceeded",
                "spent_pct": round(total_cost / budget.max_cost_usd * 100, 1) if budget and budget.max_cost_usd > 0 else 0,
            })
            return {
                "steps": updated_steps,
                "step_results": step_results,
                "current_step_index": idx + 1,
                "status": "executing",
                "retry_count": 0,
                "completed_step_ids": [step["step_id"]],
                "errors": [],
                "consumed_tokens": exec_tokens,
                "consumed_cost": exec_cost,
                "logs": [f"Step {step['step_id']} truncated - step budget cap exceeded"],
            }

        step["status"] = "completed"
        step["result"] = output

        updated_steps = list(steps)
        updated_steps[idx] = step

        step_results[step["step_id"]] = output

        await emit_event(task_id, "step_completed", {
            "step_id": step["step_id"],
            "result_preview": str(output)[:200],
            "tokens_used": total_tokens,
            "cost_usd": round(total_cost, 6),
            "budget_spent_pct": round(total_cost / budget.max_cost_usd * 100, 1) if budget and budget.max_cost_usd > 0 else 0,
        })

        return {
            "steps": updated_steps,
            "step_results": step_results,
            "current_step_index": idx + 1,
            "status": "executing",
            "retry_count": 0,
            "completed_step_ids": [step["step_id"]],
            "errors": [],
            "consumed_tokens": exec_tokens,
            "consumed_cost": exec_cost,
            "logs": [f"Completed step {step['step_id']}"],
        }

    except Exception as e:
        step["status"] = "failed"
        step["error"] = str(e)
        updated_steps = list(steps)
        updated_steps[idx] = step

        new_retry = retry_count + 1
        # After MAX_RETRIES, advance past this step to prevent infinite loop
        if new_retry >= MAX_RETRIES:
            return {
                "steps": updated_steps,
                "step_results": step_results,
                "current_step_index": idx + 1,
                "status": "executing",
                "retry_count": 0,
                "completed_step_ids": [step["step_id"]],
                "errors": state.get("errors", []) + [f"Step {step['step_id']} failed after {MAX_RETRIES} retries: {e}"],
                "logs": [f"Step {step['step_id']} failed after {MAX_RETRIES} retries — skipping"],
            }

        return {
            "steps": updated_steps,
            "status": "executing",
            "retry_count": new_retry,
            "errors": state.get("errors", []) + [f"Step {step['step_id']} failed: {e}"],
            "logs": [f"Step {step['step_id']} failed: {e}"],
        }


async def _react_loop(llm: Any, messages: list, tier: str, state: AgentState) -> tuple[str, int, float]:
    import asyncio

    tool_messages: list = []
    executed_code_hashes: set[int] = set()
    loop_tokens = 0
    loop_cost = 0.0
    _LLM_TIMEOUT = 60  # per-call timeout (seconds)

    for _iteration in range(MAX_TOOL_ITERATIONS):
        # Budget check inside loop — break early if exhausted
        from core.budget import should_skip_llm
        if should_skip_llm(state, threshold=0.75):
            last_content = ""
            for msg in reversed(tool_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    last_content = _extract_text(msg.content)
                    break
            if not last_content:
                last_content = "\n".join(
                    str(m.content)[:500] for m in tool_messages if isinstance(m, ToolMessage)
                )
            return str(last_content)[:1000] + "\n\n[Budget exhausted — tool loop stopped]", loop_tokens, loop_cost

        try:
            response = await asyncio.wait_for(
                llm.ainvoke(messages + tool_messages),
                timeout=_LLM_TIMEOUT,
            )
        except asyncio.TimeoutError:
            return f"[LLM call timed out after {_LLM_TIMEOUT}s]", loop_tokens, loop_cost

        loop_tokens += estimate_tokens(response)
        loop_cost += estimate_cost(response, tier)

        if not isinstance(response, AIMessage):
            return _extract_text(response.content), loop_tokens, loop_cost

        if not response.tool_calls:
            return _extract_text(response.content), loop_tokens, loop_cost

        tool_messages.append(response)
        for tool_call in response.tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", "")

            # Dedup: skip re-execution of identical code that already ran
            if tool_name == "code_executor":
                code_str = tool_args.get("code", "")
                code_hash = hash(code_str)
                if code_hash in executed_code_hashes:
                    tool_messages.append(ToolMessage(
                        content="Same code was already executed and failed. Do NOT retry identical code. Return your final answer as text now.",
                        tool_call_id=tool_id,
                    ))
                    continue
                executed_code_hashes.add(code_hash)

            await emit_event(state.get("task_id", ""), "tool_call", {
                "tool": tool_name,
                "args": tool_args,
            })

            import asyncio
            result = await asyncio.to_thread(registry.execute, tool_name, **tool_args)
            result_text = result.output if result.success else f"Error: {result.error}"

            await emit_event(state.get("task_id", ""), "tool_result", {
                "tool": tool_name,
                "success": result.success,
                "output_preview": str(result_text)[:200],
            })

            tool_messages.append(ToolMessage(
                content=str(result_text),
                tool_call_id=tool_id,
            ))

    # Return the last AI response content, or accumulated tool results
    last_ai_content = ""
    for msg in reversed(tool_messages):
        if isinstance(msg, AIMessage) and msg.content:
            last_ai_content = _extract_text(msg.content)
            break
    if not last_ai_content:
        # Fallback: collect all tool outputs
        last_ai_content = "\n".join(
            str(m.content)[:500] for m in tool_messages if isinstance(m, ToolMessage)
        )
    return str(last_ai_content)[:1000] + "\n\n[Tool loop limit reached]", loop_tokens, loop_cost
