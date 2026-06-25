from langgraph.graph import END, START, StateGraph

from agent.nodes.escalation import check_escalation
from agent.nodes.executor import execute_step
from agent.nodes.judge import ensemble_judge
from agent.nodes.planner import plan_task
from agent.nodes.validator import validate_result
from agent.state import AgentState

def build_single_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("planner", plan_task)
    builder.add_node("executor", execute_step)
    builder.add_node("validator", validate_result)
    builder.add_node("judge", ensemble_judge)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "validator")
    builder.add_conditional_edges(
        "validator",
        check_escalation,
        {"judge": "judge", "continue": END},
    )
    builder.add_edge("judge", END)

    return builder