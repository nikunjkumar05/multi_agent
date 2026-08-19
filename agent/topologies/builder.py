from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph

from agent.state import AgentState
from agent.topologies.single import build_single_graph
from agent.topologies.supervisor import build_supervisor_graph
from agent.topologies.pipeline import build_pipeline_graph
from agent.topologies.fanout import build_fanout_graph
from agent.topologies.ensemble import build_ensemble_graph
from agent.topologies.feedback import build_feedback_graph

_TOPOLOGY_BUILDERS = {
    "single": build_single_graph,
    "supervisor": build_supervisor_graph,
    "pipeline": build_pipeline_graph,
    "fanout": build_fanout_graph,
    "ensemble": build_ensemble_graph,
    "feedback": build_feedback_graph,
}

checkpointer: BaseCheckpointSaver | None = None


def get_checkpointer() -> BaseCheckpointSaver | None:
    """Return the current checkpointer (must be set via init_checkpointer during lifespan)."""
    return checkpointer


async def init_checkpointer() -> None:
    """Initialize MemorySaver checkpointer (in-memory, no persistence across restarts).
    
    PostgreSQL (AsyncPostgresSaver) is not yet fully compatible with the
    synchronous update_state calls in the orchestrator. See:
    - orchestrator.py:67  current_graph.update_state(..., as_node=START)  -- sync call
    - This would require converting all checkpointer operations to async (aupdate_state, aget_tuple, etc.)
    
    For now, MemorySaver is used. Persistent checkpointer can be re-enabled later
    by restoring the AsyncPostgresSaver block above and converting the orchestrator to async.
    """
    global checkpointer
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()


async def close_checkpointer() -> None:
    """Close the checkpointer. Call on shutdown."""
    global checkpointer
    if checkpointer is not None and hasattr(checkpointer, "pool"):
        await checkpointer.pool.close()
    checkpointer = None


def compile_graph(topology: str) -> StateGraph:
    builder_fn = _TOPOLOGY_BUILDERS.get(topology)
    if builder_fn is None:
        builder_fn = build_single_graph
    graph = builder_fn()
    return graph.compile(checkpointer=checkpointer)
