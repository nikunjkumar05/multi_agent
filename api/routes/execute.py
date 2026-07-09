import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends

from agent.graph import run_task
from api.middleware.auth import require_auth
from api.models.schemas import ExecuteRequest, TaskStatusResponse
from api.routes.estimate import _estimate_cost
from core.budget import BudgetTracker
from core.optimizer import CostTierOptimizer

log = logging.getLogger(__name__)

router = APIRouter()

_tasks: dict[str, TaskStatusResponse] = {}
_MAX_TASKS = 1000  # Evict oldest tasks when exceeded


def _evict_if_needed() -> None:
    """Remove oldest tasks when cache exceeds max size."""
    if len(_tasks) > _MAX_TASKS:
        to_remove = list(_tasks.keys())[:_MAX_TASKS // 2]
        for k in to_remove:
            _tasks.pop(k, None)


async def _run_background(task_id: str, task: str, budget: BudgetTracker, topology: str | None = None) -> None:
    _tasks[task_id] = TaskStatusResponse(
        task_id=task_id,
        status="running",
        budget_spent_pct=0.0,
        topology=topology or "pending",
        logs=["Task started"],
    )
    try:
        result = await run_task(task=task, budget=budget, task_id=task_id, topology_override=topology)
        _tasks[task_id] = TaskStatusResponse(**result)
    except Exception as e:
        log.exception("BG task %s FAILED", task_id)
        _tasks[task_id] = TaskStatusResponse(
            task_id=task_id,
            status="failed",
            final_result=None,
            budget_spent_pct=budget.spent_pct,
            topology="unknown",
            logs=["Task execution failed — check server logs for details"],
        )


@router.post("/execute", response_model=TaskStatusResponse, dependencies=[Depends(require_auth)])
async def execute(req: ExecuteRequest, bg: BackgroundTasks) -> TaskStatusResponse:
    _evict_if_needed()
    task_id = str(uuid.uuid4())
    budget = BudgetTracker(max_cost_usd=req.budget_usd)

    # Run optimizer to get topology + model tiers for cost estimate
    optimizer = CostTierOptimizer()
    decision = await optimizer.optimize(task=req.task, budget=budget, task_id=task_id)
    estimated_cost = _estimate_cost(decision.topology, decision.model_tiers)

    if req.budget_usd > 0:
        headroom_pct = max(0.0, 100.0 * (1.0 - estimated_cost / req.budget_usd))
        if headroom_pct > 30:
            risk = "LOW"
        elif headroom_pct > 10:
            risk = "MEDIUM"
        else:
            risk = "HIGH"
    else:
        risk = "HIGH"

    _tasks[task_id] = TaskStatusResponse(
        task_id=task_id,
        status="pending",
        budget_spent_pct=0.0,
        topology=decision.topology,
        estimated_cost=round(estimated_cost, 4),
        risk_level=risk,
        logs=["Task queued"],
    )

    bg.add_task(_run_background, task_id, req.task, budget, req.topology)
    return _tasks[task_id]
