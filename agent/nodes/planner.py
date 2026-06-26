from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AgentState, PlanStep
from core.llm import create_llm
from core.node_events import emit_event

PLANNER_SYSTEM = """You are a task planner. Break the given task into clear, numbered steps.
Return ONLY a JSON array of steps. Each step has:
- step_id (int, starting at 1)
- description (string, clear action to perform)

Example:
[{"step_id": 1, "description": "Write the coin change DP function"}, {"step_id": 2, "description": "Add test cases and verify output"}]

Rules:
- 1-3 steps maximum. Most tasks need only 1-2 steps.
- For simple tasks (code generation, Q&A, math, writing), use exactly 1 step.
- For medium tasks (code + tests, research + summary), use 2 steps.
- Never exceed 3 steps unless the task is explicitly multi-phase.
- Each step must produce a distinct, non-overlapping piece of work.
- Never repeat the same action in different steps.
- Steps should be sequential and dependent — each step builds on the previous.
"""

async def plan_task(state: AgentState) -> dict:
    task_id = state.get("task_id", "")
    await emit_event(task_id, "planner_started", {"task": state["task"]})

    tier = state["decision"].model_tiers.get("planner", "standard")
    llm = create_llm(tier)
    decision = llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=f"Task: {state['task']}"),
    ])
    import json
    content = decision.content if isinstance(decision.content, str) else str(decision.content)
    try:
        steps_raw = json.loads(content)
    except json.JSONDecodeError:
        lines = content.strip().split("\n")
        steps_raw = []
        for i, line in enumerate(lines, 1):
            cleaned = line.strip().lstrip("0123456789.:-) ")
            if cleaned:
                steps_raw.append({"step_id": i, "description": cleaned})

    steps_raw = steps_raw[:3]

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
        "logs": state.get("logs", []) + [f"Planned {len(steps)} steps"],
    }
