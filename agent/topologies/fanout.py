from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.nodes.judge import ensemble_judge
from agent.nodes.planner import plan_task
from agent.state import AgentState
from core.llm import create_llm

def dispatcher_node(state: AgentState) -> dict:
    steps = state.get("steps", [])
    tier = state["decision"].model_tiers.get("executor", "standard")
    llm = create_llm(tier)

    worker_assignments: list[dict[str, Any]] = []
    workers = ["worker_1", "worker_2", "worker_3"]
    chunk_size = max(1, len(steps) // len(workers))

    for i, worker in enumerate(workers):
        start = i * chunk_size
        end = start + chunk_size if i < len(workers) - 1 else len(steps)
        assigned = steps[start:end]
        if assigned:
            worker_assignments.append({
                "worker": worker,
                "step_ids": [s["step_id"] for s in assigned],
                "descriptions": [s["description"] for s in assigned],
            })

    return {
        "logs": state.get("logs", []) + [
            f"Dispatcher assigned {len(worker_assignments)} workers"
        ],
    }

def aggregator_node(state: AgentState) -> dict:
    step_results = state.get("step_results", {})
    combined = "\n\n".join(
        f"Step {sid}: {result}" for sid, result in step_results.items()
    )
    return {
        "final_result": combined,
        "status": "completed",
        "logs": state.get("logs", []) + ["Fanout aggregator combined results"],
    }

def build_fanout_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("planner", plan_task)
    builder.add_node("dispatcher", dispatcher_node)
    builder.add_node("aggregator", aggregator_node)
    builder.add_node("judge", ensemble_judge)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "dispatcher")
    builder.add_edge("dispatcher", "aggregator")
    builder.add_edge("aggregator", "judge")
    builder.add_edge("judge", END)

    return builder