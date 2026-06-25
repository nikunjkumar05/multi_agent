from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.topologies.single import build_single_graph
from agent.topologies.supervisor import build_supervisor_graph
from agent.topologies.pipeline import build_pipeline_graph
from agent.topologies.fanout import build_fanout_graph
from agent.topologies.ensemble import build_ensemble_graph

_TOPOLOGY_BUILDERS = {
    "single": build_single_graph,
    "supervisor": build_supervisor_graph,
    "pipeline": build_pipeline_graph,
    "fanout": build_fanout_graph,
    "ensemble": build_ensemble_graph,
}

checkpointer = MemorySaver()


def compile_graph(topology: str) -> StateGraph:
    builder_fn = _TOPOLOGY_BUILDERS.get(topology)
    if builder_fn is None:
        builder_fn = build_single_graph
    graph = builder_fn()
    return graph.compile(checkpointer=checkpointer)