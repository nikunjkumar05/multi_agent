"""
POST /estimate — dry-run cost estimation endpoint.

Runs only the cost-tier optimizer (no LangGraph execution) and returns a
budget-burn risk report.  Powers the bamas-cli tool and lets developers
sanity-check their budget before committing to a full execute call.

Cost model
----------
Token estimates per topology (rough averages measured from typical runs):

    single     ~2 000 tokens   planner + executor + validator
    pipeline   ~3 000 tokens   + always-on judge
    supervisor ~4 500 tokens   supervisor + 2 workers + judge
    fanout     ~5 500 tokens   dispatcher + 3 workers + aggregator + judge
    ensemble   ~6 500 tokens   planner + 3 specialist agents + judge

Each agent role's token cost is priced at its assigned model tier
(settings.tier_cost_per_1k_tokens).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.middleware.auth import require_auth
from api.models.schemas import ExecuteRequest
from core.budget import BudgetTracker
from core.config import settings
from core.llm import estimate_cost_from_tokens
from core.optimizer import CostTierOptimizer

router = APIRouter(dependencies=[Depends(require_auth)])

# Rough token estimates (in thousands) per role per topology
# Split into input/output: ~80% input, ~20% output
_TOPOLOGY_TOKEN_PROFILE: dict[str, dict[str, tuple[float, float]]] = {
    "single": {
        "planner": (0.4, 0.1),
        "executor": (0.8, 0.2),
        "validator": (0.4, 0.1),
        "judge": (0.0, 0.0),
    },
    "pipeline": {
        "planner": (0.4, 0.1),
        "executor": (0.8, 0.2),
        "validator": (0.4, 0.1),
        "judge": (0.8, 0.2),
    },
    "supervisor": {
        "planner": (0.4, 0.1),
        "executor": (1.6, 0.4),  # supervisor + 2 workers averaged
        "validator": (0.4, 0.1),
        "judge": (1.2, 0.3),
    },
    "fanout": {
        "planner": (0.4, 0.1),
        "executor": (2.4, 0.6),  # 3 parallel workers
        "validator": (0.4, 0.1),
        "judge": (1.2, 0.3),
    },
    "ensemble": {
        "planner": (0.4, 0.1),
        "executor": (2.4, 0.6),  # 3 specialist agents
        "validator": (0.4, 0.1),
        "judge": (2.0, 0.5),
    },
}


def _estimate_cost(topology: str, model_tiers: dict[str, str]) -> float:
    """
    Estimate USD cost for a single task run given topology + model tier mapping.
    Paper Eq. 1: c_i = T_in * P_in + T_out * P_out
    """
    profile = _TOPOLOGY_TOKEN_PROFILE.get(topology, _TOPOLOGY_TOKEN_PROFILE["single"])
    total = 0.0
    for role, (input_k, output_k) in profile.items():
        tier = model_tiers.get(role, "standard")
        # Convert from thousands to actual tokens
        input_tokens = int(input_k * 1000)
        output_tokens = int(output_k * 1000)
        total += estimate_cost_from_tokens(input_tokens, output_tokens, tier)
    return round(total, 6)


class EstimateResponse(BaseModel):
    topology: str
    model_tiers: dict[str, str]
    rationale: str
    estimated_cost_usd: float
    budget_usd: float
    budget_headroom_pct: float
    risk_level: str  # "LOW" | "MEDIUM" | "HIGH"
    budget_warning: str | None = None
    alternatives_considered: list[dict[str, str]]


@router.post("/estimate", response_model=EstimateResponse)
async def estimate_task(req: ExecuteRequest) -> EstimateResponse:
    """
    Dry-run: returns the optimizer's decision and an estimated cost without
    executing any agent graph.
    """
    budget = BudgetTracker(max_cost_usd=req.budget_usd)
    optimizer = CostTierOptimizer()
    # Use a dummy task_id — no audit entry is written for estimates
    decision = await optimizer.optimize(
        task=req.task,
        budget=budget,
        task_id="estimate-dry-run",
    )

    estimated_cost = _estimate_cost(decision.topology, decision.model_tiers)
    headroom = max(0.0, 100.0 * (1.0 - estimated_cost / req.budget_usd)) if req.budget_usd > 0 else 0.0

    if headroom > 30:
        risk = "LOW"
        warning = None
    elif headroom > 10:
        risk = "MEDIUM"
        warning = None
    else:
        risk = "HIGH"
        warning = (
            f"Budget likely insufficient. Estimated cost ${estimated_cost:.4f} "
            f"uses {100-headroom:.0f}% of your ${req.budget_usd:.2f} budget. "
            f"Steps may be skipped or topology degraded. "
            f"Consider increasing budget to ${estimated_cost * 2:.2f} for reliable results."
        )

    return EstimateResponse(
        topology=decision.topology,
        model_tiers=decision.model_tiers,
        rationale=decision.rationale,
        estimated_cost_usd=estimated_cost,
        budget_usd=req.budget_usd,
        budget_headroom_pct=round(headroom, 1),
        risk_level=risk,
        budget_warning=warning,
        alternatives_considered=decision.alternatives_considered,
    )
