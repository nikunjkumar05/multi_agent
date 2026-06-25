import json

from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AgentState
from core.llm import create_llm

VALIDATOR_SYSTEM = """You are a result validator. Evaluate whether the executed step result is correct and complete.

Return ONLY a JSON object:
{
  "confidence": 0.0-1.0,
  "reasoning_diverged": true/false,
  "issues": ["list of issues if any"],
  "assessment": "brief explanation"
}

Confidence rules:
- 1.0 = perfect result
- 0.8+ = good, minor issues
- 0.5-0.8 = needs review
- below 0.5 = likely wrong

reasoning_diverged = true if the executor's approach seems fundamentally wrong or uses different reasoning than expected.
"""


def validate_result(state: AgentState) -> dict:
    idx = state.get("current_step_index", 0)
    steps = state.get("steps", [])
    step_results = state.get("step_results", {})

    if idx < 0 or idx >= len(steps):
        return {"validator_confidence": 1.0, "reasoning_diverged": False, "status": "validating"}

    step = steps[idx]
    result_text = step_results.get(step["step_id"], step.get("result", ""))

    tier = state["decision"].model_tiers.get("validator", "cheap")
    llm = create_llm(tier)

    messages = [
        SystemMessage(content=VALIDATOR_SYSTEM),
        HumanMessage(content=(
            f"Task: {state['task']}\n"
            f"Step: {step['description']}\n"
            f"Result: {result_text}"
        )),
    ]

    response = llm.invoke(messages)
    content = response.content if isinstance(response.content, str) else str(response.content)

    try:
        parsed = json.loads(content)
        confidence = float(parsed.get("confidence", 0.5))
        diverged = bool(parsed.get("reasoning_diverged", False))
    except (json.JSONDecodeError, ValueError, TypeError):
        confidence = 0.5
        diverged = False

    return {
        "validator_confidence": confidence,
        "reasoning_diverged": diverged,
        "status": "validating",
        "logs": state.get("logs", []) + [
            f"Validation: confidence={confidence:.2f}, diverged={diverged}"
        ],
    }
