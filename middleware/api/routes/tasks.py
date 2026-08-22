import logging
import time
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from middleware import persistence
from middleware.adapters.base import AgentTask
from middleware.api.state import budget_manager, registry, tasks_db
from middleware.budget.budget_manager import BudgetAction
from middleware.classifier.task_classifier import classify_task
from middleware.models.schemas import TaskCreate, TaskResponse, TaskStatus
from middleware.selection import filter_capable, select_agents

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])

_MAX_ATTEMPTS = 3  # primary agent + up to 2 fallbacks


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
    persistence.save_budget(budget.to_dict())
    return budget, None


@router.post("/", response_model=TaskResponse)
async def create_task(task_in: TaskCreate, background_tasks: BackgroundTasks):
    """Create a task, rank agents by quality-within-budget, enforce budget, queue execution."""

    # 1. Classify the task (rule-based; overrides/validates user input)
    classification = classify_task(task_in.prompt, task_in.context)
    task_type = classification.task_type.value

    # 2. Budget FIRST — the seatbelt comes before any selection work.
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    budget, err = _resolve_budget(task_id, task_in.budget_id, task_in.budget_usd)
    if err is not None:
        raise err
    remaining_budget = budget.remaining_usd

    # 3. Budget-constrained selection (paper Component 1 adapted):
    #    maximize reliability subject to estimated_cost <= remaining budget.
    candidates = filter_capable(registry, task_type, task_in.preferred_agents)
    ordered = select_agents(candidates, _probe_task(task_id, task_in), remaining_budget)

    # Preferred agents that are capable keep absolute priority (explicit intent).
    for pid in reversed(task_in.preferred_agents or []):
        match = next(((i, c) for i, c in enumerate(ordered) if c[0] == pid), None)
        if match:
            ordered.insert(0, ordered.pop(match[0]))

    if not ordered:
        raise HTTPException(
            status_code=400,
            detail=f"No capable agents found for task type: {task_type}",
        )

    agent_ids = [aid for aid, _ in ordered]
    primary_id, primary_adapter = ordered[0]
    estimated_cost = float(primary_adapter.estimate_cost(_probe_task(task_id, task_in)))

    # 4. Budget gate on the estimate
    action = budget.can_afford(estimated_cost)
    if action == BudgetAction.DENY:
        raise HTTPException(
            status_code=402,
            detail=(
                f"Budget exhausted: ${budget.spent_usd:.6f} spent of "
                f"${budget.max_cost_usd:.6f}; cheapest capable estimate ${estimated_cost:.6f}"
            ),
        )

    warning = (
        f"High usage: {budget.spent_percentage:.0%} of budget consumed"
        if action == BudgetAction.WARN
        else None
    )

    # 5. Persist task record
    record = {
        "task_id": task_id,
        "status": TaskStatus.QUEUED,
        "estimated_cost_usd": estimated_cost,
        "estimated_tokens": 1000,  # refined after execution
        "selected_agent": primary_id,
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
    tasks_db[task_id] = record
    persistence.save_task(record)

    # 6. Queue execution with the full fallback chain
    background_tasks.add_task(
        execute_task_background, task_id, agent_ids, _probe_task(task_id, task_in)
    )

    return TaskResponse(**record)


def _probe_task(task_id: str, task_in: TaskCreate) -> AgentTask:
    return AgentTask(
        task_id=task_id,
        prompt=task_in.prompt,
        context=dict(task_in.context),
        budget_usd=task_in.budget_usd,
        timeout_seconds=task_in.timeout_seconds,
        preferred_agents=task_in.preferred_agents or [],
    )


async def execute_task_background(task_id: str, candidate_ids: list[str], task_data: AgentTask):
    """Async background worker walking the fallback chain.

    Guarantees:
      - Never overwrites a CANCELLED status (cancel race guard).
      - Refuses execution when the linked budget cannot afford the estimate.
      - On failure tries the next ranked candidate (up to _MAX_ATTEMPTS total).
      - Records actual spend against the budget on success.
    """
    task_info = tasks_db.get(task_id)
    if task_info is None:
        log.error("Task %s missing from store — skipping execution", task_id)
        return
    if task_info["status"] == TaskStatus.CANCELLED:
        log.info("Task %s cancelled before start — skipping execution", task_id)
        return

    budget = budget_manager.get_budget(task_info.get("budget_id") or "")
    last_error: str | None = None

    for attempt_no, agent_id in enumerate(candidate_ids[:_MAX_ATTEMPTS], start=1):
        # Cancelled while queued between attempts?
        if task_info["status"] == TaskStatus.CANCELLED:
            log.info("Task %s cancelled mid-chain — stopping", task_id)
            return

        adapter = registry.get(agent_id)
        if adapter is None:
            task_info["attempts"].append({
                "agent": agent_id, "success": False, "error": "unavailable/unhealthy",
                "finished_at": time.time(),
            })
            last_error = f"Agent '{agent_id}' unavailable or unhealthy"
            continue

        # Per-attempt budget gate.
        if budget is not None:
            est = task_info.get("estimated_cost_usd", 0.0) or 0.0
            if budget.can_afford(est) == BudgetAction.DENY:
                task_info["status"] = TaskStatus.FAILED
                task_info["error"] = (
                    f"Budget exhausted before attempt {attempt_no}: "
                    f"${budget.spent_usd:.6f} / ${budget.max_cost_usd:.6f} used"
                )
                log.warning("Task %s blocked: %s", task_id, task_info["error"])
                persistence.save_task(task_info)
                return

        first = attempt_no == 1
        if first and task_info["status"] == TaskStatus.QUEUED:
            task_info["status"] = TaskStatus.IN_PROGRESS
        elif not first:
            # A previous candidate failed — surface retry in status text path only.
            pass

        task_info["attempts"].append({
            "agent": agent_id, "started_at": time.time(), "attempt_no": attempt_no,
        })
        persistence.save_task(task_info)

        try:
            result = await adapter.execute(task_data)

            if task_info["status"] == TaskStatus.CANCELLED:
                log.info("Task %s cancelled during execution — discarding result", task_id)
                return

            attempt = task_info["attempts"][-1]
            attempt.update({
                "success": result.success,
                "cost_usd": result.cost_usd,
                "tokens_used": result.tokens_used,
                "latency_ms": result.latency_ms,
                "finished_at": time.time(),
            })

            is_mock = bool(result.metadata.get("mock"))

            if result.success:
                task_info["status"] = TaskStatus.COMPLETED
                task_info["output"] = result.output
                task_info["cost_usd"] = result.cost_usd
                task_info["tokens_used"] = result.tokens_used
                task_info["latency_ms"] = result.latency_ms
                task_info["budget_exceeded"] = bool(result.cost_usd > task_data.budget_usd)

                if budget is not None:
                    budget.record_usage(result.cost_usd, result.tokens_used)
                    task_info["budget_spent_usd"] = round(budget.spent_usd, 6)
                    task_info["budget_remaining_usd"] = round(budget.remaining_usd, 6)
                    persistence.save_budget(budget.to_dict())

                if not is_mock:
                    registry.record_success(agent_id, result.cost_usd, result.latency_ms)

                persistence.save_task(task_info)
                return

            task_info["error"] = result.error
            last_error = result.error
            if not is_mock:
                registry.record_failure(agent_id)

        except Exception as e:  # noqa: BLE001 — chain must survive anything
            log.error("Task %s attempt %d crashed: %s", task_id, attempt_no, e)
            task_info["attempts"][-1].update({"success": False, "error": str(e)})
            task_info["error"] = str(e)
            last_error = str(e)
            registry.record_failure(agent_id)

    # Every candidate failed.
    task_info["status"] = TaskStatus.FAILED
    task_info["error"] = last_error or "All candidate agents failed"
    persistence.save_task(task_info)


@router.get("")
async def list_tasks(limit: int = 50):
    """Most-recent-first task feed (for the dashboard)."""
    records = list(tasks_db.values())
    return [
        {k: v for k, v in r.items() if k in TaskResponse.model_fields}
        for r in reversed(records[-limit:])
    ]


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
    persistence.save_task(tasks_db[task_id])
    return {"message": "Task cancelled", "task_id": task_id}
