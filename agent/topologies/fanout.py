import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agent.nodes.judge import ensemble_judge
from agent.nodes.planner import plan_task
from agent.state import AgentState
from core.llm import create_llm, estimate_cost, estimate_tokens
from core.node_events import emit_event

WORKER_SYSTEM = """You are a worker agent assigned a specific subtask.
Complete your assigned step(s) thoroughly and accurately.
Return your result as plain text. Be specific and complete."""


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
        for step in my_steps:
            messages = [
                SystemMessage(content=WORKER_SYSTEM),
                HumanMessage(content=(
                    f"Task: {state['task']}\n"
                    f"Step {step['step_id']}: {step['description']}"
                )),
            ]
            response = await llm.ainvoke(messages)

            budget = state.get("budget")
            if budget:
                budget.record_usage(
                    tokens=estimate_tokens(response),
                    cost=estimate_cost(response, tier),
                )

            output = response.content if isinstance(response.content, str) else str(response.content)
            results[step["step_id"]] = output

            await emit_event(task_id, "step_completed", {
                "step_id": step["step_id"],
                "worker": worker_name,
                "result_preview": str(output)[:200],
            })

        return {
            "step_results": results,
            "logs": [f"{worker_name} completed {len(results)} steps"],
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
    for r in results:
        if isinstance(r, Exception):
            merged_logs.append(f"Worker error: {r}")
            continue
        merged_step_results.update(r.get("step_results", {}))
        merged_logs.extend(r.get("logs", []))

    return {
        "step_results": merged_step_results,
        "logs": merged_logs,
    }


def aggregator_node(state: AgentState) -> dict:
    step_results = state.get("step_results", {})
    combined = "\n\n".join(
        f"Step {sid}: {result}" for sid, result in sorted(step_results.items())
    )
    return {
        "final_result": combined,
        "status": "completed",
        "logs": ["Fanout aggregator combined results"],
    }


def build_fanout_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("planner", plan_task)
    builder.add_node("dispatcher", dispatcher_node)
    builder.add_node("parallel_workers", parallel_workers_node)
    builder.add_node("aggregator", aggregator_node)
    builder.add_node("judge", ensemble_judge)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "dispatcher")
    builder.add_edge("dispatcher", "parallel_workers")
    builder.add_edge("parallel_workers", "aggregator")
    builder.add_edge("aggregator", "judge")
    builder.add_edge("judge", END)

    return builder
