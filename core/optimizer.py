import asyncio

from pydantic import BaseModel, Field, field_validator

from core.audit import get_audit_trail
from core.budget import BudgetTracker
from core.llm import create_llm

Topology = str

VALID_TOPOLOGIES = {"single", "supervisor", "pipeline", "fanout", "ensemble"}


class OptimizerDecision(BaseModel):
    topology: Topology
    model_tiers: dict[str, str] = Field(
        ..., description="Model tier per agent role: planner, executor, validator, judge"
    )
    rationale: str = Field(description="Why this topology and tier allocation was chosen")
    alternatives_considered: list[dict[str, str]] = Field(description="Other topologies considered and why rejected")

    @field_validator("alternatives_considered", mode="before")
    @classmethod
    def normalize_alternatives(cls, v: object) -> list[dict[str, str]]:
        if isinstance(v, dict):
            return [{"topology": k, "reason": str(v[k])} for k in v]
        if isinstance(v, list):
            return v
        return [{"raw": str(v)}]


OPTIMIZER_PROMPT = """You are a cost-tier optimizer for a multi-agent system. Given a task and remaining budget,
you must decide:
1. Which collaboration topology to use (single, supervisor, pipeline, fanout, ensemble)
2. Which model tier each agent role gets (cheap, standard, frontier)

Topology selection rules:
- single: trivial Q&A, one-liner answers, simple math (e.g. "what is 2+2?")
- pipeline: code generation, content creation, writing, step-by-step building (e.g. "write a function", "generate code", "create a document")
- supervisor: research tasks, explanations, comparisons, multi-source synthesis (e.g. "explain X", "compare Y and Z", "research topic")
- fanout: data analysis, parallel subtasks, bulk processing (e.g. "analyze these datasets", "process multiple items")
- ensemble: high-stakes decisions, critical validation, cross-verification (e.g. "verify this proof", "audit this code")

Model tier rules:
- cheap: routing, classification, simple validation
- standard: planning, code gen, web search, most tasks
- frontier: deep reasoning, ensemble judge, complex analysis

Budget constraints (spent %):
- <70% spent: full flexibility
- 70-90% spent: downgrade frontier to standard, standard to cheap
- >90% spent: only cheap model, simplest topology

IMPORTANT: For code generation and writing tasks, prefer "pipeline" topology.
For research and explanation tasks, prefer "supervisor" topology.

Task: {task}
Budget spent: {spent_pct:.1f}%
"""


def rule_based_select_topology(task: str) -> str:
    task_lower = task.lower()

    ensemble_kw = ["verify", "audit", "validate", "critical", "security", "proof"]
    if any(kw in task_lower for kw in ensemble_kw):
        return "ensemble"

    fanout_kw = ["analyze", "data", "parallel", "bulk", "multiple datasets", "compare all"]
    if any(kw in task_lower for kw in fanout_kw):
        return "fanout"

    supervisor_kw = ["explain", "research", "compare", "why", "how does", "describe", "summarize", "review"]
    if any(kw in task_lower for kw in supervisor_kw):
        return "supervisor"

    pipeline_kw = ["function", "implement", "code", "class", "script", "create"]
    if any(kw in task_lower for kw in pipeline_kw):
        return "pipeline"

    return "single"


class CostTierOptimizer:
    def __init__(self) -> None:
        self._llm = create_llm("standard")
        self._structured_llm = self._llm.with_structured_output(OptimizerDecision)

    @staticmethod
    def _make_fallback_decision(task: str, topology: str) -> OptimizerDecision:
        return OptimizerDecision(
            topology=topology,
            model_tiers={"planner": "cheap", "executor": "standard", "validator": "cheap", "judge": "standard"},
            rationale=f"Topology forced to {topology}",
            alternatives_considered=[],
        )

    async def optimize(self, task: str, budget: BudgetTracker, task_id: str) -> OptimizerDecision:
        from core.redis_client import get_redis
        from core.rl_policy import RLPolicy

        # 1. Try LLM semantic classification
        llm_decision: OptimizerDecision | None = None
        try:
            prompt = OPTIMIZER_PROMPT.format(task=task, spent_pct=budget.spent_pct)
            llm_decision = await asyncio.wait_for(
                self._structured_llm.ainvoke(prompt),
                timeout=10.0,
            )
            if llm_decision.topology not in VALID_TOPOLOGIES:
                llm_decision = None
        except Exception:
            llm_decision = None

        # 2. Fallback: rule-based keywords if LLM failed
        if llm_decision is None:
            rule_topo = rule_based_select_topology(task)
            llm_decision = self._make_fallback_decision(task, rule_topo)

        # 3. RL refinement: override when trained enough
        redis = await get_redis()
        rl = RLPolicy(redis)
        rl_topology = await rl.select_topology(task, budget.get_band().value)

        from core.config import settings

        rule_topo = rule_based_select_topology(task)
        if rl_topology and (rule_topo == "single" or rl.total_tasks >= settings.rl_min_tasks_for_override):
            chosen = rl_topology
            rationale = f"RL policy selected {rl_topology} (trained on {rl.total_tasks} tasks)"
        else:
            chosen = llm_decision.topology
            rationale = llm_decision.rationale

        # 4. Use LLM's model tiers (with RL topology override)
        decision = OptimizerDecision(
            topology=chosen,
            model_tiers=llm_decision.model_tiers,
            rationale=rationale,
            alternatives_considered=llm_decision.alternatives_considered,
        )

        if decision.topology not in VALID_TOPOLOGIES:
            decision.topology = rule_based_select_topology(task)

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
