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

    step_count = analyze_task_complexity(state["task"])

    tier = state["decision"].model_tiers.get("planner", "standard")
    llm = create_llm(tier)

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
        "current_step_index": 0,
        "status": "planning",
        "retry_count": 0,
        "consumed_tokens": acc_tokens,
        "consumed_cost": acc_cost,
        "step_budget_caps": step_budget_caps,
        "logs": [f"Planned {len(steps)} steps (complexity={step_count})"],
    }
