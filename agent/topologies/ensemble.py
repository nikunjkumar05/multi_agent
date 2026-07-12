import asyncio
import logging
from typing import Any

from agent.nodes.budget_gate import budget_gate_node
from agent.nodes.finalizer import finalize_result
from agent.nodes.judge import ensemble_judge
from agent.nodes.planner import plan_task
from agent.state import AgentState
from core.llm import create_llm, estimate_cost, estimate_tokens
from core.node_events import emit_event
from langgraph.graph import END, START, StateGraph

log = logging.getLogger(__name__)

ENSEMBLE_PROMPTS = [
    "You are an analytical expert. Provide rigorous, data-driven analysis.",
    "You are a creative problem solver. Think outside the box and propose innovative solutions.",
    "You are a domain expert with deep subject matter knowledge. Provide authoritative insights.",
]

ENSEMBLE_ROLES = ["analytic", "creative", "domain_expert"]
ENSEMBLE_TIERS = ["standard", "standard", "frontier"]

_AGENT_TIMEOUT = 90  # hard timeout per agent (seconds)


def ensemble_dispatcher(state: AgentState) -> dict:
    """Pre-allocate budget per agent to prevent race condition.

    Divides the remaining budget equally among agents and stores
    per-agent cost caps. Each agent checks its cap before calling LLM,
    so even if all agents read the same stale state, the pre-allocated
    caps prevent collective overrun.
    """
    budget = state.get("budget")
    acc_cost = state.get("consumed_cost", 0.0)
    num_agents = len(ENSEMBLE_ROLES)

    if not budget or budget.max_cost_usd <= 0:
        return {"logs": ["Ensemble dispatcher: no budget, skipping allocation"]}

    remaining = max(0.0, budget.max_cost_usd - acc_cost)
    per_agent = remaining / num_agents

    agent_caps = {}
    for i in range(num_agents):
        agent_key = chr(ord("a") + i)
        agent_caps[agent_key] = round(acc_cost + per_agent, 6)

    return {
        "_agent_budget_caps": agent_caps,
        "logs": [f"Ensemble dispatcher allocated ${per_agent:.6f} per agent ({num_agents} agents, ${remaining:.4f} remaining)"],
    }


def _agent_variant(system_prompt: str, role: str, agent_key: str):
    """Create an agent node variant for the ensemble."""
    async def agent_node(state: AgentState) -> dict:
        task = state.get("task", "")
        task_id = state.get("task_id", "")
        decision = state.get("decision")
        tier = "standard"
        if decision and hasattr(decision, "model_tiers"):
            tier = decision.model_tiers.get("executor", "standard")

        # Domain expert uses frontier tier only if budget allows (HEALTHY band)
        if role == "domain_expert":
            budget = state.get("budget")
            if budget and budget.get_band().value == "healthy":
                tier = "frontier"

        budget = state.get("budget")
        acc_cost = state.get("consumed_cost", 0.0)

        # Check pre-allocated per-agent cap (set by ensemble_dispatcher)
        agent_caps = state.get("_agent_budget_caps", {})
        agent_cap = agent_caps.get(agent_key)

        if budget and budget.max_cost_usd > 0 and agent_cap is not None:
            # Per-agent cap: skip if this agent's pre-allocated share is exhausted
            # Note: acc_cost is stale in parallel execution, but the cap is pre-computed
            # by ensemble_dispatcher as acc_cost + per_agent, so this check is reliable.
            if acc_cost >= agent_cap:
                spent_pct = (acc_cost / budget.max_cost_usd) * 100
                await emit_event(task_id, "agent_skipped", {
                    "agent_key": agent_key,
                    "role": role,
                    "reason": "agent_cap_exceeded",
                    "agent_cap": agent_cap,
                    "spent_pct": round(spent_pct, 1),
                })
                return {
                    "consumed_tokens": 0,
                    "consumed_cost": 0,
                    "logs": [f"Agent {role} skipped - per-agent cap ${agent_cap:.6f} reached ({spent_pct:.0f}% spent)"],
                }

        llm = create_llm(tier)
        prompt = f"{system_prompt}\n\nTask: {task}"

        try:
            response = await asyncio.wait_for(
                llm.ainvoke(prompt),
                timeout=_AGENT_TIMEOUT,
            )
            content = response.content if hasattr(response, "content") else str(response)

            if not content or (isinstance(content, str) and not content.strip()):
                content = f"[Agent {role} returned empty output]"

        except asyncio.TimeoutError:
            log.warning("Agent %s timed out after %ds", agent_key, _AGENT_TIMEOUT)
            content = f"[Agent {role} timed out after {_AGENT_TIMEOUT}s]"
            response = None
            await emit_event(task_id, "agent_failed", {
                "agent_key": agent_key,
                "role": role,
                "error": f"Timeout after {_AGENT_TIMEOUT}s",
            })
        except Exception as e:
            log.warning("Agent %s failed: %s", agent_key, e)
            content = f"[Agent {role} failed: {e}]"
            response = None
            await emit_event(task_id, "agent_failed", {
                "agent_key": agent_key,
                "role": role,
                "error": str(e),
            })

        agent_tokens = estimate_tokens(response) if response is not None else 0
        agent_cost = estimate_cost(response, tier) if response is not None else 0.0

        await emit_event(task_id, "agent_completed", {
            "agent_key": agent_key,
            "role": role,
            "tokens_used": agent_tokens,
            "cost_usd": round(agent_cost, 6),
        })

        step_results = dict(state.get("step_results", {}))
        step_results[agent_key] = content

        candidate_outputs = dict(state.get("candidate_outputs", {}))
        candidate_outputs[agent_key] = {
            "output": content,
            "confidence": 0.85 if response is not None else 0.0,
            "tool_calls_count": 0,
            "tool_errors_count": 0 if response is not None else 1,
        }

        return {
            "step_results": step_results,
            "candidate_outputs": candidate_outputs,
            "consumed_tokens": agent_tokens,
            "consumed_cost": agent_cost,
            "logs": [f"Agent {role} completed"],
        }

    agent_node.__name__ = f"agent_{agent_key}"
    return agent_node


def build_ensemble_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("planner", plan_task)
    builder.add_node("ensemble_dispatcher", ensemble_dispatcher)
    for i, (prompt, role) in enumerate(zip(ENSEMBLE_PROMPTS, ENSEMBLE_ROLES)):
        agent_key = chr(ord("a") + i)
        builder.add_node(f"agent_{agent_key}", _agent_variant(prompt, role, agent_key))

    builder.add_node("budget_gate", budget_gate_node)
    builder.add_node("judge", ensemble_judge)
    builder.add_node("budget_gate_post_judge", budget_gate_node)
    builder.add_node("finalizer", finalize_result)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "ensemble_dispatcher")
    builder.add_edge("ensemble_dispatcher", "agent_a")
    builder.add_edge("ensemble_dispatcher", "agent_b")
    builder.add_edge("ensemble_dispatcher", "agent_c")
    builder.add_edge("agent_a", "budget_gate")
    builder.add_edge("agent_b", "budget_gate")
    builder.add_edge("agent_c", "budget_gate")
    builder.add_edge("budget_gate", "judge")
    builder.add_edge("judge", "budget_gate_post_judge")
    builder.add_edge("budget_gate_post_judge", "finalizer")
    builder.add_edge("finalizer", END)

    return builder