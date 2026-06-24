from pydantic import BaseModel, Field

from core.audit import get_audit_trail
from core.budget import BudgetTracker
from core.llm import create_llm

Topology = str

class OptimizerDecision(BaseModel):
    topology: Topology
    model_tiers: dict[str, str] = Field(..., description="Model tier per agent role: planner, executor, validator, judge")
    rationale: str = Field(description="Why this topology and tier allocation was chosen")
    alternatives_considered: list[dict[str, str]] = Field(description="Other topologies considered and why rejected")

OPTIMIZER_PROMPT = """You are a cost-tier optimizer for a multi-agent system. Given a task and remaining budget,
you must decide:
1. Which collaboration topology to use (single, supervisor, pipeline, fanout, ensemble)
2. Which model tier each agent role gets (cheap, standard, frontier)

Topology selection rules:
- single: simple Q&A, one-step tasks, complexity < 3
- supervisor: multi-step research, branching decisions, complexity 3-7
- pipeline: content generation, ETL, fixed sequential steps
- fanout: data analysis, parallel independent subtasks
- ensemble: high-stakes decisions, cross-validation needed

Model tier rules:
- cheap (gpt-4o-mini): routing, classification, simple validation
- standard (gpt-4o): planning, code gen, web search
- frontier (gpt-4o): deep reasoning, ensemble judge

Budget constraints (spent %):
- <70% spent: full flexibility
- 70-90% spent: downgrade frontier to standard, standard to cheap
- >90% spent: only cheap model, simplest topology

Task: {task}
Budget spent: {spent_pct:.1f}%
"""
class CostTierOptimizer:
    def __init__(self) -> None:
        self._llm = create_llm("cheap")
        self._structured_llm = self._llm.with_structured_output(OptimizerDecision)

    def optimize(self, task: str, budget: BudgetTracker, task_id: str) -> OptimizerDecision:
        prompt = OPTIMIZER_PROMPT.format(task=task, spent_pct=budget.spent_pct)
        decision: OptimizerDecision = self._structured_llm.invoke(prompt)

        audit = get_audit_trail()
        audit.record_topology_decision(
            task_id=task_id,
            topology=decision.topology,
            model_tiers=decision.model_tiers,
            budget=budget.remaining_pct,
            rationale=decision.rationale,
            alternatives=decision.alternatives_considered,
        )

        return decision