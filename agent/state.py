from __future__ import annotations
from typing import Any, Literal, TypedDict

from core.budget import BudgetTracker
from core.optimizer import OptimizerDecision

class PlanStep(TypedDict):
    step_id: int
    description: str
    status: Literal["pending", "running", "completed", "failed"]
    result: Any
    error: str | None

class AgentState(TypedDict, total=False):
    task: str
    task_id: str
    decision: OptimizerDecision
    steps: list[PlanStep]
    current_step_index: int
    step_results: dict[int, Any]
    final_result: str | None
    judge_output: str | None
    budget: BudgetTracker
    escalation_triggered: bool
    validator_confidence: float | None
    reasoning_diverged: bool 
    errors: list[str]
    logs: list[str]
    status: Literal["pending", "planning", "executing", "validating", "escalating", "completed", "failed"]
