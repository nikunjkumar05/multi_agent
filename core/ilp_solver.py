"""
ILP-based LLM provisioning solver.

Paper Component 1: Budget-Constrained LLM Provisioning

Formulation (Paper Eq. 3):
    maximize  sum(W_i * x_i)
    subject to  sum(c_i * x_i) <= B
                sum(x_i) >= 2
                x_i in {0, 1}

Decision weights (Paper Eq. 2):
    W_i = 1 + sum_{j: c_j < c_i} W_j * floor((B - c_i) / c_j)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from core.config import settings
from core.llm import ModelTier

log = logging.getLogger(__name__)

# LLM tier definitions with costs from paper Eq. 1
# c_i = T_in * P_in + T_out * P_out
# Using paper's T_in=500, T_out estimated per tier
TIER_COSTS: dict[ModelTier, float] = {}
TIER_QUALITY: dict[ModelTier, float] = {
    "cheap": 1.0,
    "standard": 2.0,
    "frontier": 3.0,
}

# Pre-compute costs using paper's Eq. 1
for _tier in ["cheap", "standard", "frontier"]:
    _p_in = settings.tier_input_cost_per_1k.get(_tier, 0.001)
    _p_out = settings.tier_output_cost_per_1k.get(_tier, 0.003)
    _t_in = settings.cost_estimation_input_tokens  # 500
    _t_out = int(_t_in * 0.5)  # ~250 output tokens typical
    TIER_COSTS[_tier] = (_t_in / 1000.0) * _p_in + (_t_out / 1000.0) * _p_out


@dataclass
class ILPResult:
    """Result from ILP solver."""
    selected_tiers: dict[ModelTier, bool]
    total_cost: float
    total_quality: float
    feasible: bool
    status: str


def compute_decision_weights(budget: float, tiers: list[ModelTier] | None = None) -> dict[ModelTier, float]:
    """Compute decision weights per paper Eq. 2.

    W_i = 1 + sum_{j: c_j < c_i} W_j * floor((B - c_i) / c_j)

    Higher cost tiers get higher weights (quality premium).
    """
    if tiers is None:
        tiers = ["cheap", "standard", "frontier"]

    weights: dict[ModelTier, float] = {}
    sorted_tiers = sorted(tiers, key=lambda t: TIER_COSTS.get(t, 0))

    for tier in sorted_tiers:
        cost_i = TIER_COSTS.get(tier, 0.001)
        w = 1.0
        for other_tier in sorted_tiers:
            cost_j = TIER_COSTS.get(other_tier, 0.001)
            if cost_j < cost_i and other_tier in weights:
                remaining_after_i = max(0, budget - cost_i)
                w += weights[other_tier] * (remaining_after_i / cost_j)
        weights[tier] = w

    return weights


def solve_ilp(budget: float, tiers: list[ModelTier] | None = None) -> ILPResult:
    """Solve the ILP for optimal LLM tier selection.

    Args:
        budget: Total budget in USD.
        tiers: Available LLM tiers (default: cheap, standard, frontier).

    Returns:
        ILPResult with selected tiers and metadata.
    """
    if tiers is None:
        tiers = ["cheap", "standard", "frontier"]

    n = len(tiers)
    costs = np.array([TIER_COSTS.get(t, 0.001) for t in tiers])
    weights = compute_decision_weights(budget, tiers)
    w = np.array([weights.get(t, 1.0) for t in tiers])

    # Objective: maximize sum(w_i * x_i) => minimize -sum(w_i * x_i)
    c = -w

    # Constraints:
    # 1. sum(c_i * x_i) <= B  =>  c^T x <= B
    # 2. sum(x_i) >= 2  =>  -sum(x_i) <= -2
    A = np.vstack([costs, np.ones(n)])
    upper = np.array([budget, np.inf])
    lower = np.array([-np.inf, -2.0])
    constraints = LinearConstraint(A, lower, upper)

    # Bounds: x_i in {0, 1}
    bounds = Bounds(lb=0, ub=1)

    # Integrality: all variables are integers
    integrality = np.ones(n, dtype=int)

    try:
        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
        )

        if result.success:
            selected = {tiers[i]: bool(round(result.x[i])) for i in range(n)}
            total_cost = sum(costs[i] * result.x[i] for i in range(n))
            total_quality = sum(w[i] * result.x[i] for i in range(n))
            n_selected = sum(1 for v in selected.values() if v)

            # Enforce minimum 2 LLMs
            if n_selected < 2:
                # Add cheapest tiers until we have 2
                sorted_by_cost = sorted(tiers, key=lambda t: TIER_COSTS.get(t, 0))
                for t in sorted_by_cost:
                    if not selected[t]:
                        selected[t] = True
                        total_cost += TIER_COSTS.get(t, 0.001)
                        total_quality += weights.get(t, 1.0)
                        n_selected += 1
                        if n_selected >= 2:
                            break

            return ILPResult(
                selected_tiers=selected,
                total_cost=total_cost,
                total_quality=total_quality,
                feasible=True,
                status="optimal",
            )
        else:
            log.warning("ILP solver failed: %s", result.message)
            return ILPResult(
                selected_tiers={t: True for t in tiers},
                total_cost=sum(costs),
                total_quality=sum(w),
                feasible=False,
                status=result.message or "infeasible",
            )
    except Exception as e:
        log.error("ILP solver error: %s", e)
        return ILPResult(
            selected_tiers={t: True for t in tiers},
            total_cost=sum(costs),
            total_quality=sum(w),
            feasible=False,
            status=f"error: {e}",
        )


def select_tiers_for_budget(budget: float) -> dict[ModelTier, bool]:
    """Convenience function: select which tiers fit within budget.

    Returns a dict mapping tier -> whether it's selected.
    """
    result = solve_ilp(budget)
    return result.selected_tiers


def get_ilp_tier_allocation(budget: float) -> dict[str, str]:
    """Get tier allocation for agent roles using ILP.

    Returns a dict mapping role -> tier, where the best available tier
    is assigned to the most critical role.
    """
    result = solve_ilp(budget)
    available = [t for t, selected in result.selected_tiers.items() if selected]

    if not available:
        return {"planner": "cheap", "executor": "cheap", "validator": "cheap", "judge": "cheap"}

    # Sort available tiers by quality (best first)
    sorted_tiers = sorted(available, key=lambda t: TIER_QUALITY.get(t, 0), reverse=True)

    # Assign best tier to most critical roles
    role_priority = ["judge", "executor", "planner", "validator"]
    allocation = {}
    tier_idx = 0

    for role in role_priority:
        if tier_idx < len(sorted_tiers):
            allocation[role] = sorted_tiers[tier_idx]
            # Don't advance tier_idx for validator (can share with planner)
            if role != "validator":
                tier_idx += 1
        else:
            allocation[role] = "cheap"

    return allocation
