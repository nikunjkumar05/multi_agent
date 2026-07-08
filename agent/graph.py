import uuid
from typing import Any

from agent.orchestrator import run_task_with_degradation
from agent.state import AgentState
from agent.topologies.builder import compile_graph, checkpointer
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
        # Execution plan
        "plan_steps": [],
        "completed_step_ids": [],
        "current_step_index": 0,
        "step_results": {},
        "candidate_outputs": {},
        "prior_context": None,
        # Final result
        "final_output": None,
        "judge_output": None,
        # Budget & degradation
        "degradation_requested": False,
        "target_topology": None,
        "topology_history": [],
        # Escalation
        "validator_confidence": None,
        "reasoning_diverged": False,
        "skip_judge": False,
        "escalation_triggered": False,
        # Resume
        "resume_signal": None,
        # Fanout / ensemble / supervisor specific
        "_worker_assignments": None,
        "fanout_worker_results": None,
        "agent_a_result": None,
        "agent_b_result": None,
        "agent_c_result": None,
        "supervisor_remaining_tasks": None,
        "supervisor_completed_tasks": None,
        # Error & logs
        "errors": [],
        "retry_count": 0,
        "logs": [f"Topology: {degraded_topology}", f"Decision: {decision.rationale}"],
        "status": "pending",
        # Backward compatibility — topologies still use `steps` and `final_result`
        "steps": [],
        "final_result": None,
    }

    # Run with mid-execution degradation support
    result = await run_task_with_degradation(
        graph=graph,
        initial_state=initial_state,
        task_id=task_id,
        topology=degraded_topology,
        checkpointer=checkpointer,
    )

    # Use final topology from orchestrator (may have degraded)
    final_topology = result.get("topology", degraded_topology)

    audit = get_audit_trail()
    audit.record(
        task_id=task_id,
        event_type="task_completed",
        detail={
            "topology": final_topology,
            "status": result.get("status", "unknown"),
            "budget_spent_pct": budget.spent_pct,
            "final_result_preview": str(result.get("final_output") or result.get("final_result", ""))[:200],
            "degradation_count": result.get("degradation_count", 0),
        },
    )

    await emit_event(
        task_id,
        "task_completed",
        {
            "status": result.get("status", "failed"),
            "final_result": str(result.get("final_output") or result.get("final_result", ""))[:500],
            "budget_spent_pct": round(budget.spent_pct, 1),
            "tokens_used": budget.consumed_tokens,
            "cost_usd": round(budget.consumed_cost, 6),
            "topology": final_topology,
            "degradation_count": result.get("degradation_count", 0),
        },
    )

    if not topology_override:
        redis = await get_redis()
        if redis:
            rl = RLPolicy(redis)
            status = result.get("status", "failed")
            quality = 1.0 if status in ("completed", "degraded_completion") else 0.0
            await record_task_result(
                rl_policy=rl,
                topology=final_topology,
                budget_band=budget.get_band().value,
                task=task,
                quality_score=quality,
                cost_usd=budget.consumed_cost,
                budget_total=budget.max_cost_usd,
                llm_topology=decision.llm_topology,
            )

    # Read final_output (canonical) with fallback to final_result (backward compat)
    final_output = result.get("final_output") or result.get("final_result")

    return {
        "task_id": task_id,
        "status": result.get("status", "failed"),
        "final_result": final_output,
        "judge_output": result.get("judge_output"),
        "budget_spent_pct": budget.spent_pct,
        "topology": final_topology,
        "degradation_count": result.get("degradation_count", 0),
        "logs": result.get("logs", []),
    }
