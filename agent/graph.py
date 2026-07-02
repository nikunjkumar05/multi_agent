import uuid
from typing import Any

from agent.state import AgentState
from agent.topologies.builder import compile_graph
from core.audit import get_audit_trail
from core.budget import BudgetBand, BudgetTracker
from core.degrader import degrade_topology
from core.learning import record_task_result
from core.node_events import emit_event
from core.optimizer import CostTierOptimizer
from core.redis_client import get_redis
from core.rl_policy import RLPolicy


async def run_task(
    task: str,
    budget: BudgetTracker,
    task_id: str | None = None,
    topology_override: str | None = None,
) -> dict[str, Any]:
    task_id = task_id or str(uuid.uuid4())

    if topology_override:
        degraded_topology = topology_override
        decision = CostTierOptimizer._make_fallback_decision(task, degraded_topology)
        await emit_event(
            task_id,
            "topology_selected",
            {
                "topology": degraded_topology,
                "rationale": f"User override: {topology_override}",
            },
        )
    else:
        optimizer = CostTierOptimizer()
        decision = await optimizer.optimize(task=task, budget=budget, task_id=task_id)
        degraded_topology = degrade_topology(
            budget=budget,
            current_topology=decision.topology,
            task_id=task_id,
        )

    graph = compile_graph(degraded_topology)

    initial_state: AgentState = {
        "task": task,
        "task_id": task_id,
        "topology": degraded_topology,
        "decision": decision,
        "budget": budget,
        "last_budget_band": BudgetBand.HEALTHY.value,
        "step_results": {},
        "final_result": None,
        "judge_output": None,
        "errors": [],
        "retry_count": 0,
        "logs": [f"Topology: {degraded_topology}", f"Decision: {decision.rationale}"],
        "status": "pending",
    }

    config = {"configurable": {"thread_id": task_id}}

    result = await graph.ainvoke(initial_state, config=config)

    audit = get_audit_trail()
    audit.record(
        task_id=task_id,
        event_type="task_completed",
        detail={
            "topology": degraded_topology,
            "status": result.get("status", "unknown"),
            "budget_spent_pct": budget.spent_pct,
            "final_result_preview": str(result.get("final_result", ""))[:200],
        },
    )

    await emit_event(
        task_id,
        "task_completed",
        {
            "status": result.get("status", "failed"),
            "final_result": str(result.get("final_result", ""))[:500],
            "budget_spent_pct": budget.spent_pct,
            "topology": degraded_topology,
        },
    )

    if not topology_override:
        redis = await get_redis()
        if redis:
            rl = RLPolicy(redis)
            status = result.get("status", "failed")
            quality = 1.0 if status == "completed" else 0.0
            await record_task_result(
                rl_policy=rl,
                topology=degraded_topology,
                budget_band=budget.get_band().value,
                task=task,
                quality_score=quality,
                cost_usd=budget.consumed_cost,
                budget_total=budget.max_cost_usd,
            )

    return {
        "task_id": task_id,
        "status": result.get("status", "failed"),
        "final_result": result.get("final_result"),
        "judge_output": result.get("judge_output"),
        "budget_spent_pct": budget.spent_pct,
        "topology": degraded_topology,
        "logs": result.get("logs", []),
    }
