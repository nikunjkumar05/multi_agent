import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AgentState, PlanStep
from core.llm import create_llm, estimate_cost, estimate_tokens
from core.node_events import emit_event

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
    response = llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=(
            f"Task: {state['task']}\n\n"
            f"Produce exactly {step_count} step(s)."
        )),
    ])

    budget = state.get("budget")
    if budget:
        budget.record_usage(
            tokens=estimate_tokens(response),
            cost=estimate_cost(response, tier),
        )

    content = response.content if isinstance(response.content, str) else str(response.content)
    try:
        steps_raw = json.loads(content)
    except json.JSONDecodeError:
        lines = content.strip().split("\n")
        steps_raw = []
        for i, line in enumerate(lines, 1):
            cleaned = line.strip().lstrip("0123456789.:-) ")
            if cleaned:
                steps_raw.append({"step_id": i, "description": cleaned})

    steps_raw = steps_raw[:step_count]

    while len(steps_raw) < step_count:
        steps_raw.append({
            "step_id": len(steps_raw) + 1,
            "description": f"Continue and complete the task (part {len(steps_raw) + 1})",
        })

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

    await emit_event(task_id, "planner_completed", {"step_count": len(steps)})

    return {
        "steps": steps,
        "current_step_index": 0,
        "status": "planning",
        "retry_count": 0,
        "logs": state.get("logs", []) + [f"Planned {len(steps)} steps (complexity={step_count})"],
    }
