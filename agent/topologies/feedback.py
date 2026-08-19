"""
Feedback topology: Generate-Critique-Revise loop.

Paper Component 3: Agent Collaboration Topology
- Linear topology (pipeline) for sequential reasoning
- Feedback topology for iterative refinement

The feedback topology implements a generate-critique-revise loop:
1. Generator produces initial output
2. Critic audits and provides feedback (no task-side computation)
3. Generator revises based on feedback
4. Loop until acceptance or budget limit

The critic is the highest-weight LLM and only evaluates, never generates.
"""

from agent.nodes.budget_gate import budget_gate_node
from agent.nodes.entry_router import entry_router_node
from agent.nodes.executor import execute_step
from agent.nodes.finalizer import finalize_result
from agent.nodes.planner import plan_task
from agent.state import AgentState
from langgraph.graph import END, START, StateGraph

MAX_ITERATIONS = 3  # Max generate-critique-revise cycles


def _route_after_planner(state: AgentState) -> str:
    """Route to executor after planning."""
    if state.get("skip_judge"):
        return "finalizer"
    return "executor"


def _route_after_critic(state: AgentState) -> str:
    """Route based on critic feedback.

    If critic accepts or budget exhausted or max iterations reached -> finalizer
    Otherwise -> executor for revision
    """
    if state.get("skip_judge"):
        return "finalizer"

    # Check iteration count
    iteration = state.get("feedback_iteration", 0)
    if iteration >= MAX_ITERATIONS:
        return "finalizer"

    # Check if critic accepted
    critic_accepted = state.get("critic_accepted", False)
    if critic_accepted:
        return "finalizer"

    # Continue with revision
    return "executor"


async def feedback_critic(state: AgentState) -> dict:
    """Critic node: evaluates the latest output and provides feedback.

    The critic does NOT perform task-side computation. It only:
    1. Reviews the executor's output
    2. Provides structured feedback (accept/reject + reasons)
    3. Returns the best output if accepted

    This matches the paper's description: "the critic does not perform any
    task-side computation but only plays the role of auditing and giving
    feedback to the executor."
    """
    import asyncio

    from langchain_core.messages import HumanMessage, SystemMessage

    from agent.nodes.executor import _extract_text
    from core.llm import create_llm, estimate_cost, estimate_tokens
    from core.node_events import emit_event

    task_id = state.get("task_id", "")

    # Get the latest output from step_results
    step_results = state.get("step_results", {})
    steps = state.get("steps", [])

    # Get the most recent step result
    latest_output = ""
    if steps:
        last_step_id = steps[-1]["step_id"]
        latest_output = step_results.get(last_step_id, "")

    if not latest_output:
        # No output to critique
        return {
            "critic_accepted": True,
            "critic_feedback": "No output to critique",
            "logs": ["Critic: no output found, accepting"],
        }

    # Budget check before critic call
    from core.budget import should_skip_llm
    if should_skip_llm(state, threshold=0.80):
        return {
            "critic_accepted": True,
            "critic_feedback": "Budget exhausted, accepting current output",
            "logs": ["Critic: budget exhausted, accepting"],
        }

    # Use the judge tier for critic (highest quality)
    tier = state.get("decision", {}).get("model_tiers", {}).get("judge", "standard")

    # Estimate cost before calling
    from core.llm import estimate_cost_from_tokens
    input_chars = len(state.get("task", "")) + len(latest_output)
    est_input_tokens = max(1, input_chars // 4)
    est_output_tokens = max(50, int(est_input_tokens * 0.3))
    est_cost = estimate_cost_from_tokens(est_input_tokens, est_output_tokens, tier)

    budget = state.get("budget")
    acc_cost = state.get("consumed_cost", 0.0)
    remaining = (budget.max_cost_usd - acc_cost) if budget and budget.max_cost_usd > 0 else 0

    if est_cost > remaining and remaining > 0:
        return {
            "critic_accepted": True,
            "critic_feedback": f"Estimated critic cost ${est_cost:.4f} exceeds remaining ${remaining:.4f}",
            "logs": [f"Critic: cost ${est_cost:.4f} exceeds remaining budget, accepting"],
        }

    iteration = state.get("feedback_iteration", 0)

    critic_system = """You are a critic agent. Your job is to evaluate the executor's output.

Review the output for:
1. Correctness: Is the answer/solution correct?
2. Completeness: Does it fully address the task?
3. Quality: Is it well-written and clear?

Return EXACTLY one of these responses:
- "ACCEPT" if the output is good enough (correct, complete, clear)
- "REJECT: <specific feedback>" if improvements needed

Be concise. Focus on critical issues only."""

    llm = create_llm(tier, temperature=0.2)
    messages = [
        SystemMessage(content=critic_system),
        HumanMessage(content=(
            f"Task: {state['task']}\n\n"
            f"Executor output (iteration {iteration + 1}):\n{latest_output}\n\n"
            f"Evaluate this output. Return ACCEPT or REJECT with specific feedback."
        )),
    ]

    try:
        response = await asyncio.wait_for(
            llm.ainvoke(messages),
            timeout=30,
        )
        critic_text = _extract_text(response.content).strip()

        # Parse critic response
        accepted = critic_text.upper().startswith("ACCEPT")
        feedback = critic_text

        # Track cost
        critic_tokens = estimate_tokens(response)
        critic_cost = estimate_cost(response, tier)

        await emit_event(task_id, "critic_completed", {
            "iteration": iteration + 1,
            "accepted": accepted,
            "feedback_preview": feedback[:200],
            "tokens_used": critic_tokens,
            "cost_usd": round(critic_cost, 6),
        })

        return {
            "critic_accepted": accepted,
            "critic_feedback": feedback,
            "feedback_iteration": iteration + 1,
            "consumed_tokens": critic_tokens,
            "consumed_cost": critic_cost,
            "logs": [f"Critic iteration {iteration + 1}: {'ACCEPT' if accepted else 'REJECT'}"],
        }

    except (asyncio.TimeoutError, Exception) as e:
        await emit_event(task_id, "critic_error", {"error": str(e)})
        return {
            "critic_accepted": True,  # Accept on error to avoid loop
            "critic_feedback": f"Critic error: {e}",
            "feedback_iteration": iteration + 1,
            "logs": [f"Critic error: {e}, accepting output"],
        }


def build_feedback_graph() -> StateGraph:
    """Build the feedback topology graph.

    Flow: entry_router -> planner -> executor -> budget_gate -> critic
          -> [executor for revision] or [finalizer if accepted]

    The critic reviews the executor's output and decides:
    - ACCEPT -> finalizer
    - REJECT -> executor (revision with feedback)
    """
    builder = StateGraph(AgentState)

    builder.add_node("entry_router", entry_router_node)
    builder.add_node("planner", plan_task)
    builder.add_node("budget_gate_post_planner", budget_gate_node)
    builder.add_node("executor", execute_step)
    builder.add_node("budget_gate", budget_gate_node)
    builder.add_node("critic", feedback_critic)
    builder.add_node("finalizer", finalize_result)

    builder.add_edge(START, "entry_router")
    builder.add_edge("entry_router", "planner")
    builder.add_edge("planner", "budget_gate_post_planner")
    builder.add_edge("budget_gate_post_planner", "executor")
    builder.add_edge("executor", "budget_gate")
    builder.add_edge("budget_gate", "critic")
    builder.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "executor": "executor",
            "finalizer": "finalizer",
        },
    )
    builder.add_edge("finalizer", END)

    return builder
