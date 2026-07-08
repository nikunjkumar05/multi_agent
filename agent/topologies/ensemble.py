import asyncio
from typing import Any

from agent.nodes.budget_gate import budget_gate_node
from agent.nodes.finalizer import finalize_result
from agent.nodes.judge import ensemble_judge
from agent.nodes.planner import plan_task
from agent.state import AgentState
from core.llm import create_llm
from langgraph.graph import END, START, StateGraph

ENSEMBLE_PROMPTS = [
    "You are an analytical expert. Provide rigorous, data-driven analysis.",
    "You are a creative problem solver. Think outside the box and propose innovative solutions.",
    "You are a domain expert with deep subject matter knowledge. Provide authoritative insights.",
]

ENSEMBLE_ROLES = ["analytic", "creative", "domain_expert"]
ENSEMBLE_TIERS = ["standard", "standard", "frontier"]


def _agent_variant(system_prompt: str, role: str, agent_key: str):
    """Create an agent node variant for the ensemble."""
    async def agent_node(state: AgentState) -> dict:
        task = state.get("task", "")
        decision = state.get("decision")
        tier = "standard"
        if decision and hasattr(decision, "model_tiers"):
            tier = decision.model_tiers.get("executor", "standard")

        if role == "domain_expert":
            tier = "frontier"

        llm = create_llm(tier)
        prompt = f"{system_prompt}\n\nTask: {task}"
        response = await llm.ainvoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        step_results = dict(state.get("step_results", {}))
        step_results[agent_key] = content

        candidate_outputs = dict(state.get("candidate_outputs", {}))
        candidate_outputs[agent_key] = {
            "output": content,
            "confidence": 0.85,
            "tool_calls_count": 0,
            "tool_errors_count": 0,
        }

        return {
            "step_results": step_results,
            "candidate_outputs": candidate_outputs,
            "logs": state.get("logs", []) + [f"Agent {role} completed"],
        }

    agent_node.__name__ = f"agent_{agent_key}"
    return agent_node


def build_ensemble_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("planner", plan_task)
    for i, (prompt, role) in enumerate(zip(ENSEMBLE_PROMPTS, ENSEMBLE_ROLES)):
        agent_key = chr(ord("a") + i)
        builder.add_node(f"agent_{agent_key}", _agent_variant(prompt, role, agent_key))

    builder.add_node("budget_gate", budget_gate_node)
    builder.add_node("judge", ensemble_judge)
    builder.add_node("finalizer", finalize_result)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "agent_a")
    builder.add_edge("planner", "agent_b")
    builder.add_edge("planner", "agent_c")
    builder.add_edge("agent_a", "budget_gate")
    builder.add_edge("agent_b", "budget_gate")
    builder.add_edge("agent_c", "budget_gate")
    builder.add_edge("budget_gate", "judge")
    builder.add_edge("judge", "finalizer")
    builder.add_edge("finalizer", END)

    return builder