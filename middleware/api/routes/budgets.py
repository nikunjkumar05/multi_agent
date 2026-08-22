"""Budget and agent-info routes.

POST /api/v1/budgets        create a persistent budget
GET  /api/v1/budgets        list budgets (optional ?owner=)
GET  /api/v1/budgets/{id}   budget status with spend/remaining
GET  /api/v1/agents         registry contents: health, pricing, stats
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from middleware.api.state import budget_manager, registry
from middleware.models.schemas import BudgetCreate, BudgetResponse

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["budgets"])


def _to_response(data: dict) -> BudgetResponse:
    """Build BudgetResponse from MiddlewareBudget.to_dict(), dropping extras."""
    fields = set(BudgetResponse.model_fields.keys())
    return BudgetResponse(**{k: v for k, v in data.items() if k in fields})


@router.post("/budgets", response_model=BudgetResponse, status_code=201)
async def create_budget(payload: BudgetCreate):
    """Create a persistent budget that tasks can reference via budget_id."""
    budget = budget_manager.create_budget(
        name=payload.name,
        owner=payload.owner,
        max_cost_usd=payload.max_cost_usd,
        max_tasks=payload.max_tasks,
        warn_threshold=payload.warn_threshold,
        ttl_seconds=payload.ttl_seconds,
    )
    log.info(
        "Budget created via API: %s ($%.2f, %d tasks, owner=%s)",
        budget.budget_id, payload.max_cost_usd, payload.max_tasks, payload.owner,
    )
    return _to_response(budget.to_dict())


@router.get("/budgets")
async def list_budgets(owner: str | None = Query(default=None)):
    """List all budgets, optionally filtered by owner."""
    return budget_manager.list_budgets(owner=owner)


@router.get("/budgets/{budget_id}", response_model=BudgetResponse)
async def get_budget(budget_id: str):
    budget = budget_manager.get_budget(budget_id)
    if budget is None:
        raise HTTPException(status_code=404, detail="Budget not found")
    return _to_response(budget.to_dict())


@router.delete("/budgets/{budget_id}")
async def delete_budget(budget_id: str):
    if not budget_manager.delete_budget(budget_id):
        raise HTTPException(status_code=404, detail="Budget not found")
    return {"message": "Budget deleted", "budget_id": budget_id}


@router.get("/agents")
async def list_agents():
    """List registered agents with health, capabilities, and live stats."""
    return {
        "summary": registry.get_summary(),
        "agents": registry.list_agents(),
    }
