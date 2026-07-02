from fastapi import APIRouter, HTTPException

from api.models.schemas import AuditResponse
from core.audit import get_audit_trail

router = APIRouter()


@router.get("/audit/{task_id}", response_model=AuditResponse)
async def get_audit(task_id: str) -> AuditResponse:
    audit = get_audit_trail()
    # Fast path: serve from in-memory list
    events = audit.get_task_audit(task_id)
    # Fallback: query SQLite (handles restarts where in-memory cache was cleared)
    if not events:
        events = await audit.load_from_db(task_id)
    if not events:
        raise HTTPException(status_code=404, detail="No audit entries for this task")
    return AuditResponse(task_id=task_id, events=events)
