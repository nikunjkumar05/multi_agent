from typing import Literal

from agent.state import AgentState
from core.escalation import should_escalate
from core.node_events import emit_event


async def check_escalation(state: AgentState) -> Literal["judge", "continue"]:
    escalate = should_escalate(
        validator_confidence=state.get("validator_confidence", 1.0),
        reasoning_diverged=state.get("reasoning_diverged", False),
        budget=state["budget"],
    )

    task_id = state.get("task_id", "")
    await emit_event(task_id, "escalation_check", {
        "confidence": state.get("validator_confidence", 1.0),
        "diverged": state.get("reasoning_diverged", False),
        "escalated": escalate,
    })

    if escalate:
        return "judge"
    return "continue"
