from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from core.budget import BudgetTracker
from core.optimizer import OptimizerDecision


def merge_dicts(left: dict, right: dict) -> dict:
    """Merge two dicts. Right values override left. Used for annotated fields."""
    new_dict = dict(left or {})
    new_dict.update(right or {})
    return new_dict


merge_step_results = merge_dicts  # alias — same semantics


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
    # Task metadata & control
    task: str
    task_id: str
    topology: str
    decision: OptimizerDecision

    # Shared execution plan
    plan_steps: list[PlanStep]
    completed_step_ids: Annotated[list[int], operator.add]
    current_step_index: int

    # Execution outputs
    step_results: Annotated[dict[Any, Any], merge_step_results]
    candidate_outputs: Annotated[dict, merge_dicts]
    aggregated_output: str | None
    prior_context: str | None

    # Final result
    final_output: str | None
    judge_output: str | None

    # Budget & degradation
    budget: BudgetTracker
    last_budget_band: str | None
    degradation_requested: bool
    target_topology: str | None
    topology_history: Annotated[list[dict], operator.add]

    # Escalation & validation
    validator_confidence: float | None
    reasoning_diverged: bool
    skip_judge: bool
    escalation_triggered: bool

    # Resume signal
    resume_signal: str | None

    # Fanout-specific (optional)
    _worker_assignments: dict[str, list[PlanStep]] | None
    fanout_worker_results: list[dict] | None

    # Ensemble-specific (optional)
    agent_a_result: dict | None
    agent_b_result: dict | None
    agent_c_result: dict | None

    # Supervisor-specific (optional)
    supervisor_remaining_tasks: list[str] | None
    supervisor_completed_tasks: list[dict] | None

    # Error & logs
    error: str | None
    errors: Annotated[list[str], merge_errors]
    logs: Annotated[list[str], merge_logs]

    # Status & control
    status: Literal["pending", "planning", "executing", "validating", "escalating", "completed", "failed", "degraded_completion"]
    retry_count: int

    # Backward compatibility (deprecated - use plan_steps, final_output)
    steps: list[PlanStep]
    final_result: str | None
