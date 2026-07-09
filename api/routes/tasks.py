from fastapi import APIRouter, Depends, HTTPException, Query

from api.middleware.auth import require_auth
from api.models.schemas import TaskStatusResponse
from api.routes.execute import _tasks

router = APIRouter(dependencies=[Depends(require_auth)])


@router.get("/tasks", response_model=list[TaskStatusResponse])
async def list_tasks(limit: int = Query(default=50, ge=1, le=200)) -> list[TaskStatusResponse]:
    """List recent tasks, newest first."""
    tasks = list(_tasks.values())
    return tasks[-limit:][::-1]


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task(task_id: str) -> TaskStatusResponse:
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return _tasks[task_id]
