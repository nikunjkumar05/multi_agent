from typing import Literal

from agent.state import AgentState
from core.escalation import should_escalate


def check_escalation(state: AgentState) -> Literal["judge", "continue"]:
    escalate = should_escalate(
        validator_confidence=state.get("validator_confidence", 1.0),
        reasoning_diverged=state.get("reasoning_diverged", False),
        budget=state["budget"],
    )
    if escalate:
        return "judge"
    return "continue"