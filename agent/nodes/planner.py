import asyncio
import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.nodes.executor import _extract_text
from agent.state import AgentState, PlanStep
from core.llm import create_llm, estimate_cost, estimate_tokens
from core.node_events import emit_event

log = logging.getLogger(__name__)

PLANNER_TIMEOUT = 60

def _budget_adjusted_step_count(desired: int, budget, tier: str) -> int:
    """Reduce step count if budget can't support the desired number of steps.

    Cost model per step:
    - Executor LLM call: ~500 tokens output + ~200 tokens input = ~700 tokens
    - Validator LLM call: ~300 tokens output
    - Judge (amortized): ~200 tokens output
    Total per step: ~1200 tokens

    We need budget for: planner (1 call) + N steps × executor + N steps × validator + judge (1 call)
    """
    if budget is None or budget.max_cost_usd <= 0:
        return desired

    from core.llm import estimate_cost_from_tokens

    # Paper Eq. 1: c_i = T_in * P_in + T_out * P_out
    # Estimate cost per step: executor (~700 tok) + validator (~300 tok)
    executor_input = 560  # ~80% of 700
    executor_output = 140  # ~20% of 700
    validator_input = 240  # ~80% of 300
    validator_output = 60  # ~20% of 300
    cost_per_step = (
        estimate_cost_from_tokens(executor_input, executor_output, tier)
        + estimate_cost_from_tokens(validator_input, validator_output, tier)
    )
    # Fixed overhead: planner (~500 tok input, ~200 tok output) + judge (~200 tok input, ~500 tok output)
    fixed_cost = (
        estimate_cost_from_tokens(500, 200, tier)
        + estimate_cost_from_tokens(200, 500, tier)
    )

    remaining = budget.max_cost_usd - budget.consumed_cost
    if remaining <= fixed_cost:
        return 1  # Only enough for 1 step

    available_for_steps = remaining - fixed_cost
    max_steps_by_budget = max(1, int(available_for_steps / cost_per_step)) if cost_per_step > 0 else desired

    adjusted = min(desired, max_steps_by_budget)

    if adjusted < desired:
        log.info(
            "Budget-adjusted steps: %d → %d (budget=$%.4f, remaining=$%.4f, cost_per_step=$%.6f)",
            desired, adjusted, budget.max_cost_usd, remaining, cost_per_step,
        )

    return adjusted


TRIVIAL_KEYWORDS = {
    "hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye",
    "yes", "no", "ok", "okay", "sure", "help",
}

MATH_KEYWORDS = {
    "calculate", "solve", "equation", "compute", "sum", "add", "subtract",
    "multiply", "divide", "what is", "how much", "factorial", "square root",
}

ACTION_VERBS = {
    "write", "create", "build", "implement", "develop", "generate",
    "design", "code", "program", "script", "function", "class",
    "explain", "describe", "compare", "analyze", "research", "review",
    "summarize", "evaluate", "optimize", "refactor", "test", "debug",
    "fix", "update", "modify", "delete", "remove", "add", "install",
    "deploy", "configure", "setup", "initialize", "migrate", "convert",
    "verify", "audit", "validate", "proof", "check", "inspect",
    "search", "find", "list", "show", "display", "print",
}


def analyze_task_complexity(task: str) -> int:
    task_lower = task.lower().strip()
    words = task_lower.split()
    word_count = len(words)

    if task_lower in TRIVIAL_KEYWORDS or (word_count <= 2 and not any(v in task_lower for v in ACTION_VERBS)):
        return 1

    if any(kw in task_lower for kw in MATH_KEYWORDS) and word_count < 15:
        return 1

    action_count = sum(1 for v in ACTION_VERBS if v in task_lower)

    has_multi_phase = any(phrase in task_lower for phrase in [
        " and then", " first ", " finally ", "step 1", "step 2",
        "phase 1", "phase 2", "after that", "once done",
    ])

    if has_multi_phase or action_count >= 3 or word_count > 40:
        return 3

    if action_count >= 1 or word_count > 8:
        return 2

    return 1


def _strip_code_fences(text: str) -> str:
    """Strip markdown code block fences (```json ... ``` or ``` ... ```)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


PLANNER_SYSTEM = """You are a task planner. Break the given task into clear, numbered steps.
Return ONLY a JSON array of steps. Each step has:
- step_id (int, starting at 1)
- description (string, clear action to perform)

Rules:
- Produce EXACTLY the number of steps specified in the step_count parameter.
- Each step must produce a distinct, non-overlapping piece of work.
- Never repeat the same action in different steps.
- Steps should be sequential and dependent — each step builds on the previous.
- Be specific: "Write a fibonacci function with memoization" not "Write code".
"""


async def plan_task(state: AgentState) -> dict:
    task_id = state.get("task_id", "")
    await emit_event(task_id, "planner_started", {"task": state["task"]})

    # Resume case: steps already exist from previous topology.
    # Skip re-planning — just advance past completed steps.
    existing_steps = state.get("steps") or state.get("plan_steps") or []
    completed_ids = state.get("completed_step_ids", [])
    if existing_steps and completed_ids:
        next_idx = max(completed_ids) + 1
        if next_idx >= len(existing_steps):
            # All steps completed — go to finalizer
            return {
                "current_step_index": next_idx,
                "status": "executing",
                "logs": [f"Resume: all {len(existing_steps)} steps already completed"],
            }
        return {
            "current_step_index": next_idx,
            "status": "executing",
            "logs": [f"Resume: skipping to step {next_idx} ({len(completed_ids)} already done)"],
        }

    # Pre-LLM budget check — skip planning if budget exhausted
    from core.budget import should_skip_llm
    if should_skip_llm(state):
        budget = state.get("budget")
        spent_pct = round(state.get("consumed_cost", 0.0) / budget.max_cost_usd * 100, 1) if budget and budget.max_cost_usd > 0 else 0
        fallback_step = PlanStep(step_id=1, description=state["task"], status="pending", result=None, error=None)
        await emit_event(task_id, "planner_skipped", {
            "reason": "budget_exhausted",
            "spent_pct": spent_pct,
        })
        return {
            "steps": [fallback_step],
            "plan_steps": [fallback_step],
            "current_step_index": 0,
            "status": "executing",
            "retry_count": 0,
            "consumed_tokens": 0,
            "consumed_cost": 0.0,
            "logs": [f"Planner skipped - budget exhausted ({spent_pct}% spent), using single-step fallback"],
        }

    tier = state["decision"].model_tiers.get("planner", "standard")
    llm = create_llm(tier)

    # Budget-aware step count: scale steps to what the budget can support
    budget = state.get("budget")
    desired_steps = analyze_task_complexity(state["task"])
    step_count = _budget_adjusted_step_count(desired_steps, budget, tier)

    local_tokens = 0
    local_cost = 0.0

    try:
        response = await asyncio.wait_for(
            llm.ainvoke([
                SystemMessage(content=PLANNER_SYSTEM),
                HumanMessage(content=(
                    f"Task: {state['task']}\n\n"
                    f"Produce exactly {step_count} step(s)."
                )),
            ]),
            timeout=PLANNER_TIMEOUT,
        )

        local_tokens = estimate_tokens(response)
        local_cost = estimate_cost(response, tier)

        content = _extract_text(response.content)
        cleaned_content = _strip_code_fences(content)
        try:
            steps_raw = json.loads(cleaned_content)
        except json.JSONDecodeError:
            lines = cleaned_content.strip().split("\n")
            steps_raw = []
            for i, line in enumerate(lines, 1):
                cleaned = re.sub(r"^\d+[\.\)\:\-]\s*", "", line.strip())
                if cleaned:
                    steps_raw.append({"step_id": i, "description": cleaned})

        steps_raw = steps_raw[:step_count]

        while len(steps_raw) < step_count:
            steps_raw.append({
                "step_id": len(steps_raw) + 1,
                "description": f"Continue and complete the task (part {len(steps_raw) + 1})",
            })

    except Exception as e:
        log.warning("Planner LLM failed: %s — using single-step fallback", e)
        steps_raw = [{"step_id": 1, "description": state["task"]}]

    steps: list[PlanStep] = [
        PlanStep(
            step_id=s["step_id"],
            description=s["description"],
            status="pending",
            result=None,
            error=None,
        )
        for s in steps_raw
    ]

    budget = state.get("budget")
    prev_tokens = state.get("consumed_tokens", 0) if state else 0
    prev_cost = state.get("consumed_cost", 0.0) if state else 0.0
    acc_tokens = prev_tokens + local_tokens
    acc_cost = prev_cost + local_cost

    # Calculate per-step budget caps (accumulated cost allowed after each step)
    remaining_budget = (budget.max_cost_usd - acc_cost) if budget and budget.max_cost_usd > 0 else float("inf")
    per_step = remaining_budget / max(len(steps), 1)
    step_budget_caps = {}
    for i, step in enumerate(steps):
        step_budget_caps[str(step["step_id"])] = round(acc_cost + per_step * (i + 1), 6)

    await emit_event(task_id, "planner_completed", {
        "step_count": len(steps),
        "tokens_used": acc_tokens,
        "cost_usd": round(acc_cost, 6),
        "budget_spent_pct": round(acc_cost / budget.max_cost_usd * 100, 1) if budget and budget.max_cost_usd > 0 else 0,
    })

    return {
        "steps": steps,
        "plan_steps": steps,
        "current_step_index": 0,
        "status": "planning",
        "retry_count": 0,
        "consumed_tokens": local_tokens,
        "consumed_cost": local_cost,
        "step_budget_caps": step_budget_caps,
        "logs": [f"Planned {len(steps)} steps (complexity={step_count})"],
    }
