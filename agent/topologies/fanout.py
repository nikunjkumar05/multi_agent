import asyncio
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agent.nodes.budget_gate import budget_gate_node
from agent.nodes.judge import ensemble_judge
from agent.nodes.planner import plan_task
from agent.state import AgentState
from core.llm import create_llm, estimate_cost, estimate_tokens
from core.node_events import emit_event

log = logging.getLogger(__name__)

WORKER_SYSTEM = """You are a worker agent assigned a specific subtask.
Complete your assigned step(s) thoroughly and accurately.
Return your result as plain text. Be specific and complete."""

_WORKER_TIMEOUT = 90


def dispatcher_node(state: AgentState) -> dict:
    steps = state.get("steps", [])

    workers = ["worker_1", "worker_2", "worker_3"]
    chunk_size = max(1, len(steps) // len(workers))

    assignments: dict[str, list[dict[str, Any]]] = {}
    for i, worker in enumerate(workers):
        start = i * chunk_size
        end = start + chunk_size if i < len(workers) - 1 else len(steps)
        assigned = steps[start:end]
        if assigned:
            assignments[worker] = assigned

    return {
        "_worker_assignments": assignments,
        "logs": [
            f"Dispatcher assigned {len(assignments)} workers with {len(steps)} total steps"
        ],
    }


def _make_worker_node(worker_name: str):
    async def worker_node(state: AgentState) -> dict:
        assignments = state.get("_worker_assignments", {})
        my_steps = assignments.get(worker_name, [])
        task_id = state.get("task_id", "")
        tier = state["decision"].model_tiers.get("executor", "standard")
        llm = create_llm(tier)

        results = {}
        worker_tokens = 0
        worker_cost = 0.0
        for step in my_steps:
            messages = [
                SystemMessage(content=WORKER_SYSTEM),
                HumanMessage(content=(
                    f"Task: {state['task']}\n"
                    f"Step {step['step_id']}: {step['description']}"
                )),
            ]

            try:
                response = await asyncio.wait_for(
                    llm.ainvoke(messages),
                    timeout=_WORKER_TIMEOUT,
                )
                output = response.content if isinstance(response.content, str) else str(response.content)
                if not output or not output.strip():
                    output = f"[Worker {worker_name} returned empty output for step {step['step_id']}]"

                step_tokens = estimate_tokens(response)
                step_cost = estimate_cost(response, tier)

            except asyncio.TimeoutError:
                log.warning("Worker %s timed out on step %s", worker_name, step["step_id"])
                output = f"[Worker {worker_name} timed out on step {step['step_id']}]"
                response = None
                step_tokens = 0
                step_cost = 0.0
            except Exception as e:
                log.warning("Worker %s failed on step %s: %s", worker_name, step["step_id"], e)
                output = f"[Worker {worker_name} failed on step {step['step_id']}: {e}]"
                response = None
                step_tokens = 0
                step_cost = 0.0

            results[step["step_id"]] = output
            worker_tokens += step_tokens
            worker_cost += step_cost

            await emit_event(task_id, "step_completed", {
                "step_id": step["step_id"],
                "worker": worker_name,
                "result_preview": str(output)[:200],
                "tokens_used": worker_tokens,
                "cost_usd": round(worker_cost, 6),
            })

        return {
            "step_results": results,
            "consumed_tokens": worker_tokens,
            "consumed_cost": worker_cost,
            "logs": [f"Worker {worker_name} completed {len(results)} steps"],
        }
    return worker_node


async def parallel_workers_node(state: AgentState) -> dict:
    assignments = state.get("_worker_assignments", {})
    tasks = []
    for worker_name in assignments:
        node_fn = _make_worker_node(worker_name)
        tasks.append(node_fn(state))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    merged_step_results = {}
    merged_logs = []
    fanout_worker_results = []
    for i, r in enumerate(results):
        worker_name = list(assignments.keys())[i] if i < len(assignments) else f"worker_{i}"
        if isinstance(r, Exception):
            merged_logs.append(f"Worker error: {r}")
            fanout_worker_results.append({"worker": worker_name, "steps": [], "status": "error"})
            continue
        merged_step_results.update(r.get("step_results", {}))
        merged_logs.extend(r.get("logs", []))
        fanout_worker_results.append({
            "worker": worker_name,
            "steps": list(r.get("step_results", {}).keys()),
            "status": "completed",
        })

    return {
        "step_results": merged_step_results,
        "fanout_worker_results": fanout_worker_results,
        "logs": merged_logs,
    }


def aggregator_node(state: AgentState) -> dict:
    step_results = state.get("step_results", {})
    combined = "\n\n".join(
        f"Step {sid}: {result}" for sid, result in sorted(step_results.items())
    )
    return {
        "final_output": combined,
        "final_result": combined,  # backward compat
        "status": "completed",
        "logs": ["Fanout aggregator combined results"],
    }


def build_fanout_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("planner", plan_task)
    builder.add_node("dispatcher", dispatcher_node)
    builder.add_node("parallel_workers", parallel_workers_node)
    builder.add_node("budget_gate", budget_gate_node)
    builder.add_node("aggregator", aggregator_node)
    builder.add_node("judge", ensemble_judge)
    builder.add_node("budget_gate_post_judge", budget_gate_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "dispatcher")
    builder.add_edge("dispatcher", "parallel_workers")
    builder.add_edge("parallel_workers", "budget_gate")
    builder.add_edge("budget_gate", "aggregator")
    builder.add_edge("aggregator", "judge")
    builder.add_edge("judge", "budget_gate_post_judge")
    builder.add_edge("budget_gate_post_judge", END)

    return builder
