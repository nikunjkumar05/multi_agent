import uuid
from typing import Any

from agent.state import AgentState
from agent.topologies.builder import compile_graph
from core.audit import get_audit_trail
from core.budget import BudgetTracker
from core.degrader import degrade_topology
from core.optimizer import CostTierOptimizer


async def run_task(
    task: str,
    budget: BudgetTracker,
    task_id: str | None = None,
) -> dict[str, Any]:
    task_id = task_id or str(uuid.uuid4())

    optimizer = CostTierOptimizer()
    decision = optimizer.optimize(task=task, budget=budget, task_id=task_id)

    degraded_topology = degrade_topology(
        budget=budget,
        current_topology=decision.topology,
        task_id=task_id,
    )

    graph = compile_graph(degraded_topology)

    initial_state: AgentState = {
        "task": task,
        "task_id": task_id,
        "decision": decision,
        "budget": budget,
        "step_results": {},
        "final_result": None,
        "judge_output": None,
        "errors": [],
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

    return {
        "task_id": task_id,
        "status": result.get("status", "failed"),
        "final_result": result.get("final_result"),
        "judge_output": result.get("judge_output"),
        "budget_spent_pct": budget.spent_pct,
        "topology": degraded_topology,
        "logs": result.get("logs", []),
    }