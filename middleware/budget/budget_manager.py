"""Budget manager — tracks costs across tasks and enforces budget limits.

The middleware's budget lifecycle:
  1. User creates a budget ($1.00 for 10 tasks)
  2. Each task is checked against the budget before execution
  3. After execution, cost is deducted from the budget
  4. When budget is exhausted, tasks are rejected

This is different from the BAMAS core budget gate (which enforces per-task
budgets). This manager handles per-user and per-project budgets.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


class BudgetStatus(str, Enum):
    ACTIVE = "active"
    EXHAUSTED = "exhausted"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class BudgetAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    WARN = "warn"  # near limit but still allowed


@dataclass
class MiddlewareBudget:
    """A budget that tracks spending across multiple tasks."""

    budget_id: str = field(default_factory=lambda: f"budget_{uuid.uuid4().hex[:8]}")
    name: str = ""
    owner: str = ""  # user or project ID

    # Limits
    max_cost_usd: float = 1.0
    max_tasks: int = 100
    max_tokens: int = 200_000

    # Current usage
    spent_usd: float = 0.0
    tasks_completed: int = 0
    tokens_used: int = 0

    # Status
    status: BudgetStatus = BudgetStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None  # None = never expires

    # Thresholds (percentage of budget)
    warn_threshold: float = 0.80  # warn at 80%
    hard_limit: float = 1.0  # hard stop at 100%

    @property
    def remaining_usd(self) -> float:
        """Remaining budget in USD."""
        return max(0.0, self.max_cost_usd - self.spent_usd)

    @property
    def remaining_tasks(self) -> int:
        """Remaining task count."""
        return max(0, self.max_tasks - self.tasks_completed)

    @property
    def spent_percentage(self) -> float:
        """Percentage of budget spent (0.0 to 1.0+)."""
        if self.max_cost_usd <= 0:
            return 0.0
        return self.spent_usd / self.max_cost_usd

    @property
    def is_active(self) -> bool:
        """Check if budget is still active."""
        if self.status != BudgetStatus.ACTIVE:
            return False
        if self.expires_at and time.time() > self.expires_at:
            self.status = BudgetStatus.EXPIRED
            return False
        return True

    def can_afford(self, estimated_cost: float) -> BudgetAction:
        """Check if we can afford a task with estimated cost.

        Returns:
            ALLOW if within budget, WARN if near limit, DENY if over.
        """
        if not self.is_active:
            return BudgetAction.DENY

        # Check hard limit
        if self.spent_usd + estimated_cost > self.max_cost_usd * self.hard_limit:
            return BudgetAction.DENY

        # Check warn threshold
        if self.spent_usd + estimated_cost > self.max_cost_usd * self.warn_threshold:
            return BudgetAction.WARN

        # Check task count
        if self.tasks_completed >= self.max_tasks:
            return BudgetAction.DENY

        return BudgetAction.ALLOW
 
    def record_usage(self, cost_usd: float, tokens: int = 0) -> None:
        """Record usage after task completion."""
        self.spent_usd += cost_usd
        self.tokens_used += tokens
        self.tasks_completed += 1

        # Check if budget is now exhausted
        if self.spent_usd >= self.max_cost_usd * self.hard_limit:
            self.status = BudgetStatus.EXHAUSTED
            log.warning(
                "Budget %s exhausted: $%.4f / $%.4f",
                self.budget_id, self.spent_usd, self.max_cost_usd,
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "budget_id": self.budget_id,
            "name": self.name,
            "owner": self.owner,
            "max_cost_usd": self.max_cost_usd,
            "max_tasks": self.max_tasks,
            "max_tokens": self.max_tokens,
            "spent_usd": round(self.spent_usd, 6),
            "tasks_completed": self.tasks_completed,
            "tokens_used": self.tokens_used,
            "remaining_usd": round(self.remaining_usd, 6),
            "remaining_tasks": self.remaining_tasks,
            "spent_percentage": round(self.spent_percentage, 4),
            "status": self.status.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


class BudgetManager:
    """Manages multiple budgets for users and projects.

    Usage:
        manager = BudgetManager()

        # Create a budget
        budget = manager.create_budget(
            name="My Project",
            owner="user_123",
            max_cost_usd=1.0,
            max_tasks=100,
        )

        # Check before task
        action = manager.check_budget(budget.budget_id, estimated_cost=0.05)
        if action == BudgetAction.ALLOW:
            # execute task
            manager.record_usage(budget.budget_id, cost_usd=0.03, tokens=750)
    """

    def __init__(self):
        self._budgets: dict[str, MiddlewareBudget] = {}

    def create_budget(
        self,
        name: str = "",
        owner: str = "",
        max_cost_usd: float = 1.0,
        max_tasks: int = 100,
        max_tokens: int = 100_000,
        warn_threshold: float = 0.80,
        hard_limit: float = 1.0,
        ttl_seconds: float | None = None,
    ) -> MiddlewareBudget:
        """Create a new budget.

        Args:
            name: Human-readable name.
            owner: User or project ID.
            max_cost_usd: Maximum spending in USD.
            max_tasks: Maximum number of tasks.
            max_tokens: Maximum token usage.
            warn_threshold: Percentage to start warning (0.80 = 80%).
            hard_limit: Percentage to hard stop (1.0 = 100%).
            ttl_seconds: Time-to-live in seconds (None = no expiry).

        Returns:
            The created budget.
        """
        budget = MiddlewareBudget(
            name=name,
            owner=owner,
            max_cost_usd=max_cost_usd,
            max_tasks=max_tasks,
            max_tokens=max_tokens,
            warn_threshold=warn_threshold,
            hard_limit=hard_limit,
        )

        if ttl_seconds:
            budget.expires_at = time.time() + ttl_seconds

        self._budgets[budget.budget_id] = budget
        log.info(
            "Created budget %s: $%.2f, %d tasks, owner=%s",
            budget.budget_id, max_cost_usd, max_tasks, owner,
        )
        return budget

    def get_budget(self, budget_id: str) -> MiddlewareBudget | None:
        """Get a budget by ID."""
        return self._budgets.get(budget_id)

    def check_budget(self, budget_id: str, estimated_cost: float = 0.0) -> BudgetAction:
        """Check if a budget can afford a task.

        Args:
            budget_id: The budget to check.
            estimated_cost: Estimated cost of the task.

        Returns:
            ALLOW, WARN, or DENY.
        """
        budget = self._budgets.get(budget_id)
        if budget is None:
            log.warning("Budget %s not found", budget_id)
            return BudgetAction.DENY
        return budget.can_afford(estimated_cost)

    def record_usage(
        self,
        budget_id: str,
        cost_usd: float,
        tokens: int = 0,
    ) -> bool:
        """Record usage for a budget.

        Returns:
            True if recorded successfully, False if budget not found.
        """
        budget = self._budgets.get(budget_id)
        if budget is None:
            log.warning("Budget %s not found", budget_id)
            return False
        budget.record_usage(cost_usd, tokens)
        return True

    def delete_budget(self, budget_id: str) -> bool:
        """Delete a budget.

        Returns:
            True if deleted, False if not found.
        """
        if budget_id in self._budgets:
            del self._budgets[budget_id]
            log.info("Deleted budget %s", budget_id)
            return True
        return False

    def list_budgets(self, owner: str | None = None) -> list[dict[str, Any]]:
        """List all budgets, optionally filtered by owner."""
        budgets = []
        for budget in self._budgets.values():
            if owner and budget.owner != owner:
                continue
            budgets.append(budget.to_dict())
        return budgets

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all budgets."""
        total_budgets = len(self._budgets)
        active = sum(1 for b in self._budgets.values() if b.is_active)
        total_spent = sum(b.spent_usd for b in self._budgets.values())
        total_tasks = sum(b.tasks_completed for b in self._budgets.values())

        return {
            "total_budgets": total_budgets,
            "active_budgets": active,
            "total_spent_usd": round(total_spent, 6),
            "total_tasks_completed": total_tasks,
        }

    def cleanup_expired(self) -> int:
        """Remove expired budgets.

        Returns:
            Number of budgets removed.
        """
        expired_ids = [
            bid for bid, budget in self._budgets.items()
            if not budget.is_active
        ]
        for bid in expired_ids:
            del self._budgets[bid]
        if expired_ids:
            log.info("Cleaned up %d expired budgets", len(expired_ids))
        return len(expired_ids)
