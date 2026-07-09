from pydantic import BaseModel, Field


class ExecuteRequest(BaseModel):
    task: str = Field(..., description="Plain-English task description")
    budget_usd: float = Field(default=1.0, ge=0.001, le=100.0, description="Max budget in USD")
    topology: str | None = Field(default=None, description="Force topology: single, pipeline, supervisor, fanout, ensemble")


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    final_result: str | None = None
    judge_output: str | None = None
    budget_spent_pct: float
    topology: str
    logs: list[str] = []
    estimated_cost: float | None = None
    risk_level: str | None = None


class AuditResponse(BaseModel):
    task_id: str
    events: list[dict]
