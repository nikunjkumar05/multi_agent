from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agent.nodes.judge import ensemble_judge
from agent.nodes.planner import plan_task
from agent.state import AgentState
from core.llm import create_llm

ENSEMBLE_PROMPTS = [
    "You are Agent A. Approach this task with analytical rigor and precision.",
    "You are Agent B. Approach this task with creative problem-solving.",
    "You are Agent C. Approach this task with domain expertise.",
]

def _agent_variant(system_prompt: str):
    async def agent_node(state: AgentState) -> dict:
        tier = state["decision"].model_tiers.get("executor", "standard")
        model = "frontier" if "Agent C" in system_prompt else tier
        llm = create_llm(model)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=f"Task: {state['task']}\n\nComplete this task.")]

        response = await llm.ainvoke(messages)
        output = response.content if isinstance(response.content, str) else str(response.content)

        agent_name = "Agent A"
        if "Agent B" in system_prompt:
            agent_name = "Agent B"
        elif "Agent C" in system_prompt:
            agent_name = "Agent C"

        agent_key = agent_name.lower().replace(" ", "_")

        return {
            "step_results": {agent_key: output},
            "logs": [f"{agent_name} completed"],
        }
    return agent_node


def build_ensemble_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("planner", plan_task)
    builder.add_node("agent_a", _agent_variant(ENSEMBLE_PROMPTS[0]))
    builder.add_node("agent_b", _agent_variant(ENSEMBLE_PROMPTS[1]))
    builder.add_node("agent_c", _agent_variant(ENSEMBLE_PROMPTS[2]))
    builder.add_node("judge", ensemble_judge)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "agent_a")
    builder.add_edge("planner", "agent_b")
    builder.add_edge("planner", "agent_c")
    builder.add_edge("agent_a", "judge")
    builder.add_edge("agent_b", "judge")
    builder.add_edge("agent_c", "judge")
    builder.add_edge("judge", END)

    return builder