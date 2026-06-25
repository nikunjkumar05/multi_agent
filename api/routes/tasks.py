from fastapi import APIRouter, HTTPException

from api.models.schemas import TaskStatusResponse
from api.routes.execute import _tasks

router = APIRouter()


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task(task_id: str) -> TaskStatusResponse:
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return _tasks[task_id]
