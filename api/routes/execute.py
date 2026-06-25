import uuid

from fastapi import APIRouter, BackgroundTasks

from agent.graph import run_task
from api.models.schemas import ExecuteRequest, TaskStatusResponse
from core.budget import BudgetTracker

router = APIRouter()

_tasks: dict[str, TaskStatusResponse] = {}


async def _run_background(task_id: str, task: str, budget: BudgetTracker, topology: str | None = None) -> None:
    try:
        result = await run_task(task=task, budget=budget, task_id=task_id, topology_override=topology)
        _tasks[task_id] = TaskStatusResponse(**result)
    except Exception as e:
        _tasks[task_id] = TaskStatusResponse(
            task_id=task_id,
            status="failed",
            final_result=None,
            budget_spent_pct=budget.spent_pct,
            topology="unknown",
            logs=[f"Error: {str(e)}"],
        )


@router.post("/execute", response_model=TaskStatusResponse)
async def execute(req: ExecuteRequest, bg: BackgroundTasks) -> TaskStatusResponse:
    task_id = str(uuid.uuid4())
    budget = BudgetTracker(max_cost_usd=req.budget_usd)

    _tasks[task_id] = TaskStatusResponse(
        task_id=task_id,
        status="pending",
        budget_spent_pct=0.0,
        topology="pending",
        logs=["Task queued"],
    )

    bg.add_task(_run_background, task_id, req.task, budget, req.topology)
    return _tasks[task_id]
