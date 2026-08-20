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
    budget_usd: float = Field(..., description="Budget in USD for this task")
    timeout_seconds: int = Field(default=120, description="Timeout in seconds")
    preferred_agents: Optional[List[str]] = Field(default=None, description="e.g., ['opencode', 'aider']")
    fallback_agents: Optional[List[str]] = Field(default=None, description="e.g., ['codex']")

class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    estimated_cost_usd: Optional[float] = None
    estimated_tokens: Optional[int] = None
    selected_agent: Optional[str] = None
    selected_tier: Optional[str] = None
    ws_url: Optional[str] = None
    
    # Fields populated upon completion
    output: Optional[str] = None
    cost_usd: Optional[float] = None
    tokens_used: Optional[int] = None
    latency_ms: Optional[int] = None
    quality_score: Optional[float] = None
    budget_remaining_usd: Optional[float] = None

class BudgetCreate(BaseModel):
    limit_usd: float
    description: str = ""

class BudgetResponse(BaseModel):
    budget_id: str
    limit_usd: float
    spent_usd: float
    remaining_usd: float
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
