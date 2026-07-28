from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph

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

checkpointer: BaseCheckpointSaver | None = None


async def init_checkpointer() -> None:
    """Initialize PostgreSQL checkpointer. Falls back to MemorySaver if unavailable."""
    global checkpointer
    from core.db import get_psycopg_pool

    pool = await get_psycopg_pool()
    if pool is not None:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        checkpointer = AsyncPostgresSaver(pool)
        return

    # Fallback: in-memory (no persistence across restarts)
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
