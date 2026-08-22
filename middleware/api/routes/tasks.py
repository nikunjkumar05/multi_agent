import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from middleware.api.state import budget_manager, registry, tasks_db
from middleware.budget.budget_manager import BudgetAction
from middleware.classifier.task_classifier import classify_task
from middleware.models.schemas import TaskCreate, TaskResponse, TaskStatus
from middleware.adapters.base import AgentTask

import logging

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])


def _resolve_budget(task_id: str, budget_id: str | None, per_task_ceiling: float):
    """Return (budget, error_response). Creates an ephemeral task-scoped budget when needed."""
    if budget_id:
        budget = budget_manager.get_budget(budget_id)
        if budget is None:
            return None, HTTPException(status_code=404, detail=f"Budget '{budget_id}' not found")
        if not budget.is_active:
            return None, HTTPException(
                status_code=402,
                detail=f"Budget '{budget_id}' is {budget.status.value}; cannot accept new tasks",
            )
        return budget, None

    # No persistent budget referenced — create a single-task ephemeral one.
    budget = budget_manager.create_budget(
        name=f"task-{task_id}",
        owner="task",
        max_cost_usd=per_task_ceiling,
        max_tasks=1,
    )
    return budget, None


@router.post("/", response_model=TaskResponse)
async def create_task(task_in: TaskCreate, background_tasks: BackgroundTasks):
    """Create a task, select the cheapest capable agent, enforce budget, queue execution."""

    # 1. Classify the task (rule-based; overrides/validates user input)
    classification = classify_task(task_in.prompt, task_in.context)
    task_type = classification.task_type.value

    # 2. Select agent: honor preferred_agents that can handle the type, else cheapest.
    selected = None
    for preferred in task_in.preferred_agents or []:
        candidate_adapter = registry.get(preferred)
        if candidate_adapter and registry.can_handle(preferred, task_type):
            selected = (preferred, candidate_adapter)
            break
    if selected is None:
        selected = registry.get_cheapest(task_type)
    if selected is None:
        raise HTTPException(
            status_code=400,
            detail=f"No capable agents found for task type: {task_type}",
        )

    agent_id, adapter = selected
    task_id = f"task_{uuid.uuid4().hex[:8]}"

    # 3. Resolve the budget BEFORE anything else — the seatbelt comes first.
    budget, err = _resolve_budget(task_id, task_in.budget_id, task_in.budget_usd)
    if err is not None:
        raise err

    # 4. Build the standardized AgentTask
    agent_task = AgentTask(
        task_id=task_id,
        task_type=task_type,
        prompt=task_in.prompt,
        context=task_in.context,
        budget_usd=task_in.budget_usd,
        timeout_seconds=task_in.timeout_seconds,
        preferred_agents=task_in.preferred_agents or [],
    )

    # 5. Estimate cost and enforce the budget gate
    estimated_cost = adapter.estimate_cost(agent_task)
    action = budget.can_afford(estimated_cost)

    if action == BudgetAction.DENY:
        raise HTTPException(
            status_code=402,
            detail=(
                f"Budget exhausted: ${budget.spent_usd:.6f} spent of "
                f"${budget.max_cost_usd:.6f}; estimated need ${estimated_cost:.6f}"
            ),
        )

    warning = (
        f"High usage: {budget.spent_percentage:.0%} of budget consumed"
        if action == BudgetAction.WARN
        else None
    )

    # 6. Persist task record
    tasks_db[task_id] = {
        "task_id": task_id,
        "status": TaskStatus.QUEUED,
        "estimated_cost_usd": estimated_cost,
        "estimated_tokens": 1000,  # refined after execution
        "selected_agent": agent_id,
        "selected_tier": "standard",
        "output": None,
        "error": None,
        "cost_usd": None,
        "tokens_used": None,
        "latency_ms": None,
        "budget_id": budget.budget_id,
        "budget_spent_usd": round(budget.spent_usd, 6),
        "budget_remaining_usd": round(budget.remaining_usd, 6),
        "budget_exceeded": False,
        "warning": warning,
        "attempts": [],
    }

    # 7. Queue execution
    background_tasks.add_task(execute_task_background, task_id, agent_id, agent_task)

    return TaskResponse(**tasks_db[task_id])


async def execute_task_background(task_id: str, agent_id: str, task_data: AgentTask):
    """Async background worker.

    Guarantees:
      - Never overwrites a CANCELLED status (cancel race guard).
      - Refuses execution when the linked budget cannot afford the estimate.
      - Records actual spend against the budget after completion.
    """
    task_info = tasks_db.get(task_id)
    if task_info is None:
        log.error("Task %s missing from store — skipping execution", task_id)
        return
    if task_info["status"] == TaskStatus.CANCELLED:
        log.info("Task %s cancelled before start — skipping execution", task_id)
        return

    adapter = registry.get(agent_id)
    if adapter is None:
        task_info["status"] = TaskStatus.FAILED
        task_info["output"] = f"Agent '{agent_id}' unavailable or unhealthy"
        task_info["error"] = f"Agent '{agent_id}' unavailable or unhealthy"
        return

    # Pre-execution budget gate (state may have changed since creation).
    budget = budget_manager.get_budget(task_info.get("budget_id") or "")
    if budget is not None:
        action = budget.can_afford(task_info.get("estimated_cost_usd", 0.0))
        if action == BudgetAction.DENY:
            task_info["status"] = TaskStatus.FAILED
            task_info["error"] = (
                f"Budget exhausted before start: ${budget.spent_usd:.6f} / "
                f"${budget.max_cost_usd:.6f} used"
            )
            log.warning("Task %s blocked: %s", task_id, task_info["error"])
            return

    task_info["status"] = TaskStatus.IN_PROGRESS
    task_info["attempts"].append({"agent": agent_id, "started_at": _now()})

    try:
        result = await adapter.execute(task_data)

        # Re-check: user may have cancelled while the agent was running.
        if task_info["status"] == TaskStatus.CANCELLED:
            log.info("Task %s cancelled during execution — discarding result", task_id)
            return

        # Mock results don't count toward real reliability/cost stats.
        is_mock = bool(result.metadata.get("mock"))

        attempts = task_info["attempts"][-1]
        attempts.update({
            "success": result.success,
            "cost_usd": result.cost_usd,
            "tokens_used": result.tokens_used,
            "latency_ms": result.latency_ms,
            "finished_at": _now(),
        })

        if result.success:
            task_info["status"] = TaskStatus.COMPLETED
            task_info["output"] = result.output
            task_info["cost_usd"] = result.cost_usd
            task_info["tokens_used"] = result.tokens_used
            task_info["latency_ms"] = result.latency_ms
            task_info["budget_exceeded"] = result.cost_usd > task_data.budget_usd

            if budget is not None:
                budget.record_usage(result.cost_usd, result.tokens_used)
                task_info["budget_spent_usd"] = round(budget.spent_usd, 6)
                task_info["budget_remaining_usd"] = round(budget.remaining_usd, 6)

            if not is_mock:
                registry.record_success(agent_id, result.cost_usd, result.latency_ms)
        else:
            task_info["status"] = TaskStatus.FAILED
            task_info["output"] = result.error
            task_info["error"] = result.error
            if not is_mock:
                registry.record_failure(agent_id)

    except Exception as e:
        log.error("Task execution failed: %s", e)
        task_info["status"] = TaskStatus.FAILED
        task_info["output"] = str(e)
        task_info["error"] = str(e)
        registry.record_failure(agent_id)


def _now() -> float:
    import time

    return time.time()


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """Get the current status, results, and budget receipt of a task."""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskResponse(**tasks_db[task_id])


@router.delete("/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a queued or in-progress task."""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")

    status = tasks_db[task_id]["status"]
    if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
        raise HTTPException(
            status_code=409,
            detail=f"Task already {status.value}; cannot cancel",
        )

    tasks_db[task_id]["status"] = TaskStatus.CANCELLED
    return {"message": "Task cancelled", "task_id": task_id}
