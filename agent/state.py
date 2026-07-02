from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from core.budget import BudgetTracker
from core.optimizer import OptimizerDecision


def merge_step_results(left: dict, right: dict) -> dict:
    new_dict = dict(left or {})
    new_dict.update(right or {})
    return new_dict


def merge_logs(left: list, right: list) -> list:
    return (left or []) + (right or [])


def merge_errors(left: list, right: list) -> list:
    return (left or []) + (right or [])


class PlanStep(TypedDict):
    step_id: int
    description: str
    status: Literal["pending", "running", "completed", "failed"]
    result: Any
    error: str | None


class AgentState(TypedDict, total=False):
    task: str
    task_id: str
    topology: str  # Actual (possibly degraded) topology being run
    decision: OptimizerDecision
    steps: list[PlanStep]
    current_step_index: int
    step_results: Annotated[dict[Any, Any], merge_step_results]
    final_result: str | None
    judge_output: str | None
    budget: BudgetTracker
    last_budget_band: str | None  # Tracks band transitions for mid-execution monitoring
    escalation_triggered: bool
    validator_confidence: float | None
    reasoning_diverged: bool
    errors: list[str]
    logs: Annotated[list[str], merge_logs]
    status: Literal["pending", "planning", "executing", "validating", "escalating", "completed", "failed"]
    retry_count: int
