from langchain_core.messages import HumanMessage, SystemMessage
from agent.state import AgentState, PlanStep
from core.llm import create_llm

PLANNER_SYSTEM = """You are a task planner. Break the given task into clear, numbered steps.
Return ONLY a JSON array of steps. Each step has:
- step_id (int, starting at 1)
- description (string, clear action to perform)

Example:
[{"step_id": 1, "description": "Research quantum computing basics"}, {"step_id": 2, "description": "Draft summary of key concepts"}]

Rules:
- 2-8 steps maximum
- Each step must be actionable by a single agent
- Steps should be sequential and dependent
"""

def plan_task(state: AgentState) -> dict:
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

    return {
        "steps": steps,
        "current_step_index": 0,
        "status": "planning",
        "logs": state.get("logs", []) + [f"Planned {len(steps)} steps"],
    }