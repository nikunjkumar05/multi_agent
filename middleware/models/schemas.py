from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskCreate(BaseModel):
    task_type: str = Field(..., description="e.g., code_generation, code_review")
    prompt: str = Field(..., description="The main task instruction")
    context: Dict[str, Any] = Field(default_factory=dict, description="Files, language, project info")
    budget_usd: float = Field(..., gt=0, description="Per-task spending ceiling in USD")
    timeout_seconds: int = Field(default=120, ge=1, description="Timeout in seconds")
    preferred_agents: Optional[List[str]] = Field(default=None, description="e.g., ['opencode', 'aider']")
    fallback_agents: Optional[List[str]] = Field(default=None, description="e.g., ['codex']")
    budget_id: Optional[str] = Field(
        default=None,
        description="Link to a persistent budget (POST /api/v1/budgets). "
        "If omitted, an ephemeral task-scoped budget is created from budget_usd.",
    )


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    estimated_cost_usd: Optional[float] = None
    estimated_tokens: Optional[int] = None
    selected_agent: Optional[str] = None
    selected_tier: Optional[str] = None

    # Populated upon completion / failure
    output: Optional[str] = None
    error: Optional[str] = None
    cost_usd: Optional[float] = None
    tokens_used: Optional[int] = None
    latency_ms: Optional[int] = None
    quality_score: Optional[float] = None

    # Budget receipt — the middleware's core promise
    budget_id: Optional[str] = None
    budget_spent_usd: Optional[float] = None
    budget_remaining_usd: Optional[float] = None
    budget_exceeded: bool = False
    warning: Optional[str] = None

    # Fallback audit trail (populated when agent retries occur)
    attempts: Optional[List[Dict[str, Any]]] = None


class BudgetCreate(BaseModel):
    """Create a persistent budget. Mirrors BudgetManager.create_budget."""

    name: str = ""
    owner: str = ""
    max_cost_usd: float = Field(..., gt=0)
    max_tasks: int = Field(default=100, ge=1)
    warn_threshold: float = Field(default=0.80, gt=0, le=1)
    ttl_seconds: Optional[float] = Field(default=None, description="Expiry in seconds; None = never")


class BudgetResponse(BaseModel):
    budget_id: str
    name: str
    owner: str
    max_cost_usd: float
    spent_usd: float
    remaining_usd: float
    tasks_completed: int
    remaining_tasks: int
    status: str


class AgentInfoResponse(BaseModel):
    agent_id: str
    display_name: str
    version: str
    api_type: str
    capabilities: List[str]
    pricing: Dict[str, float]
    max_tokens: int
    latency_p50_ms: int
    latency_p99_ms: int
    reliability: float
    health_endpoint: Optional[str]
    enabled: bool
