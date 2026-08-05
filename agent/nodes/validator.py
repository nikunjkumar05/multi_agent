import json

from langchain_core.messages import HumanMessage, SystemMessage

from agent.nodes.executor import _extract_text
from agent.state import AgentState
from agent.nodes.executor import detect_task_type
from core.llm import create_llm, estimate_cost, estimate_tokens
from core.node_events import emit_event

VALIDATOR_PROMPTS = {
    "code": (
        "You are a code reviewer. Evaluate the code for:\n"
        "- Correctness and logic errors\n"
        "- Edge cases and error handling\n"
        "- Code style and readability\n"
        "- Security concerns\n\n"
        "Return ONLY a JSON object:\n"
        '{"confidence": 0.0-1.0, "reasoning_diverged": true/false, '
        '"issues": ["list of issues if any"], "assessment": "brief explanation"}\n\n'
        "Confidence: 1.0=perfect, 0.8+=good, 0.5-0.8=needs review, <0.5=likely wrong\n"
        "reasoning_diverged = true if the approach is fundamentally wrong."
    ),
    "math": (
        "You are a math verifier. Check the mathematical reasoning for:\n"
        "- Calculation accuracy\n"
        "- Logical flow of steps\n"
        "- Correct application of formulas\n"
        "- Final answer correctness\n\n"
        "Return ONLY a JSON object:\n"
        '{"confidence": 0.0-1.0, "reasoning_diverged": true/false, '
        '"issues": ["list of issues if any"], "assessment": "brief explanation"}'
    ),
    "research": (
        "You are a research quality checker. Evaluate for:\n"
        "- Accuracy of claims\n"
        "- Completeness of analysis\n"
        "- Quality of sources referenced\n"
        "- Logical coherence\n\n"
        "Return ONLY a JSON object:\n"
        '{"confidence": 0.0-1.0, "reasoning_diverged": true/false, '
        '"issues": ["list of issues if any"], "assessment": "brief explanation"}'
    ),
    "general": (
        "You are a result validator. Evaluate whether the executed step result is correct and complete.\n\n"
        "Return ONLY a JSON object:\n"
        '{"confidence": 0.0-1.0, "reasoning_diverged": true/false, '
        '"issues": ["list of issues if any"], "assessment": "brief explanation"}\n\n'
        "Confidence: 1.0=perfect, 0.8+=good, 0.5-0.8=needs review, <0.5=likely wrong\n"
        "reasoning_diverged = true if the approach is fundamentally wrong."
    ),
}

import re


def _heuristic_validate(result_text: str, step_description: str) -> tuple[float, bool, list[str]]:
    """Zero-cost heuristic validation when budget is too low for LLM validation.

    Returns (confidence, reasoning_diverged, issues).
    """
    issues = []

    if not result_text or not result_text.strip():
        return 0.1, True, ["Empty output"]

    text = result_text.strip()

    # Check for error markers
    error_markers = ["[Step skipped", "[Error:", "[Failed", "[LLM call timed out", "Traceback (most recent call last)"]
    for marker in error_markers:
        if marker in text:
            issues.append(f"Contains error marker: {marker}")

    # Check minimum length (a meaningful result should be >50 chars)
    if len(text) < 50:
        issues.append(f"Very short output ({len(text)} chars)")

    # Check for code-specific markers when task is code-related
    code_keywords = ["def ", "class ", "function", "return", "import"]
    step_lower = step_description.lower()
    is_code_task = any(kw in step_lower for kw in ["code", "function", "implement", "write", "script", "class"])
    if is_code_task:
        has_code = any(kw in text for kw in code_keywords)
        if not has_code:
            issues.append("Code task but no code constructs found in output")
        # Check for syntax error indicators
        if "SyntaxError" in text or "IndentationError" in text:
            issues.append("Contains Python syntax errors")

    # Check for math-specific markers
    is_math_task = any(kw in step_lower for kw in ["calculate", "compute", "solve", "math", "what is"])
    if is_math_task:
        # Check if there's a numeric answer
        has_number = bool(re.search(r'\d+', text))
        if not has_number:
            issues.append("Math task but no numeric result found")

    # Compute confidence
    if issues:
        confidence = max(0.2, 0.7 - len(issues) * 0.15)
    else:
        confidence = 0.75  # Heuristic can't be higher than 0.75

    diverged = len(issues) >= 2 or any("error" in i.lower() for i in issues)

    return confidence, diverged, issues


async def validate_result(state: AgentState) -> dict:
    idx = state.get("current_step_index", 0)
    steps = state.get("steps", [])
    step_results = state.get("step_results", {})

    last_idx = idx - 1
    if last_idx < 0 or last_idx >= len(steps):
        return {
            "validator_confidence": 1.0,
            "reasoning_diverged": False,
            "validation_skipped": False,
            "status": "validating",
            "consumed_tokens": 0,
            "consumed_cost": 0.0,
            "logs": ["Validation: no step to validate (index out of bounds)"],
        }

    step = steps[last_idx]
    result_text = step_results.get(step["step_id"], step.get("result", ""))

    budget = state.get("budget")
    acc_cost = state.get("consumed_cost", 0.0)
    if budget and budget.max_cost_usd > 0:
        spent_pct = (acc_cost / budget.max_cost_usd) * 100
        if spent_pct >= 90:
            # Instead of skipping: use zero-cost heuristic validation
            task_id = state.get("task_id", "")
            confidence, diverged, issues = _heuristic_validate(result_text, step.get("description", ""))
            await emit_event(task_id, "validation_heuristic", {
                "confidence": confidence,
                "diverged": diverged,
                "issues": issues,
                "spent_pct": round(spent_pct, 1),
            })
            return {
                "validator_confidence": confidence,
                "reasoning_diverged": diverged,
                "validation_skipped": False,
                "validation_heuristic": True,
                "status": "validating",
                "consumed_tokens": 0,
                "consumed_cost": 0.0,
                "logs": [f"Validation: HEURISTIC (budget at {spent_pct:.0f}%), confidence={confidence:.2f}, issues={issues}"],
            }

    task_type = detect_task_type(state["task"])
    validator_prompt = VALIDATOR_PROMPTS.get(task_type, VALIDATOR_PROMPTS["general"])

    tier = state["decision"].model_tiers.get("validator", "cheap")
    llm = create_llm(tier)

    messages = [
        SystemMessage(content=validator_prompt),
        HumanMessage(content=(
            f"Task: {state['task']}\n"
            f"Step: {step['description']}\n"
            f"Result: {result_text}"
        )),
    ]

    try:
        response = await llm.ainvoke(messages)
    except Exception as e:
        # LLM failed — fall back to heuristic validation
        task_id = state.get("task_id", "")
        confidence, diverged, issues = _heuristic_validate(result_text, step.get("description", ""))
        await emit_event(task_id, "validation_heuristic", {
            "confidence": confidence,
            "diverged": diverged,
            "issues": issues,
            "reason": f"llm_error: {e}",
        })
        return {
            "validator_confidence": confidence,
            "reasoning_diverged": diverged,
            "validation_skipped": False,
            "validation_heuristic": True,
            "status": "validating",
            "consumed_tokens": 0,
            "consumed_cost": 0.0,
            "logs": [f"Validation: HEURISTIC (LLM error: {e}), confidence={confidence:.2f}"],
        }

    val_tokens = estimate_tokens(response)
    val_cost = estimate_cost(response, tier)

    content = _extract_text(response.content)

    try:
        parsed = json.loads(content)
        confidence = float(parsed.get("confidence", 0.5))
        diverged = bool(parsed.get("reasoning_diverged", False))
    except (json.JSONDecodeError, ValueError, TypeError):
        confidence = 0.5
        diverged = False

    task_id = state.get("task_id", "")
    budget = state.get("budget")
    prev_tokens = state.get("consumed_tokens", 0)
    prev_cost = state.get("consumed_cost", 0.0)
    acc_tokens = prev_tokens + val_tokens
    acc_cost = prev_cost + val_cost
    await emit_event(task_id, "validation_completed", {
        "confidence": confidence,
        "diverged": diverged,
        "tokens_used": acc_tokens,
        "cost_usd": round(acc_cost, 6),
        "budget_spent_pct": round(acc_cost / budget.max_cost_usd * 100, 1) if budget and budget.max_cost_usd > 0 else 0,
    })

    return {
        "validator_confidence": confidence,
        "reasoning_diverged": diverged,
        "validation_skipped": False,
        "status": "validating",
        "consumed_tokens": val_tokens,
        "consumed_cost": val_cost,
        "logs": [
            f"Validation: confidence={confidence:.2f}, diverged={diverged}"
        ],
    }
