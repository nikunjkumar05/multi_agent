"""Toy test: Validates LangGraph mid-execution topology change mechanism.

Key finding: update_state with annotated fields MERGES, not replaces.
Projection functions must NOT include annotated fields (like step_results)
in the output dict. The checkpoint already has the accumulated value.

Flow:
  Graph A (2 nodes): start -> a1 -> interrupt_node -> a2 -> end
  Graph B (1 node):  start -> b1 -> end

1. Start Graph A via ainvoke
2. interrupt_node fires, returning __interrupt__
3. Orchestrator catches interrupt, projects state from A->B format
4. Builds Graph B (new topology) on same checkpointer + thread_id
5. update_state with projected state (ONLY non-annotated fields)
6. ainvoke(None) to resume on Graph B
7. Graph B runs b1, returns final state
8. Verify: accumulated outputs from both graphs are present
"""
import asyncio
import uuid
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import interrupt


# ----- Shared state schema (mirrors BAMASState subset) -----
class SharedState(TypedDict):
    task: str
    step_results: Annotated[list[str], lambda a, b: (a or []) + (b or [])]
    current_topology: str
    final_output: str | None


# ----- Graph A (has interrupt) -----
def node_a1(state: SharedState):
    return {"step_results": ["a1_done"]}


def interrupt_node(state: SharedState):
    """Simulates budget gate detecting STRUCTURAL_DEGRADE."""
    interrupt({"reason": "STRUCTURAL_DEGRADE", "from_topology": "graph_a"})
    return {}


def node_a2(state: SharedState):
    return {"step_results": ["a2_done"], "final_output": "graph_a_complete"}


# ----- Graph B (no interrupt) -----
def node_b1(state: SharedState):
    return {"step_results": ["b1_done"], "final_output": "graph_b_complete"}


# ----- Graph C (for multi-step degrade test) -----
def node_c1(state: SharedState):
    return {"step_results": ["c1_done"], "final_output": "graph_c_complete"}


# ----- Projection functions -----
# IMPORTANT: These must NOT include annotated fields like step_results.
# update_state MERGES annotated fields, so including them causes duplication.
# The checkpoint already has the accumulated value.
def project_a_to_b(state: dict) -> dict:
    """Project Graph A state -> Graph B state."""
    return {
        "task": state.get("task", ""),
        "current_topology": "graph_b",
        "final_output": None,
    }


def project_b_to_c(state: dict) -> dict:
    """Project Graph B state -> Graph C state."""
    return {
        "task": state.get("task", ""),
        "current_topology": "graph_c",
        "final_output": None,
    }


# ===== Test 1: Single degrade (A -> B) =====
async def test_single_degrade():
    checkpointer = MemorySaver()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # --- Build and run Graph A ---
    builder_a = StateGraph(SharedState)
    builder_a.add_node("a1", node_a1)
    builder_a.add_node("interrupt", interrupt_node)
    builder_a.add_node("a2", node_a2)
    builder_a.add_edge(START, "a1")
    builder_a.add_edge("a1", "interrupt")
    builder_a.add_edge("interrupt", "a2")
    graph_a = builder_a.compile(checkpointer=checkpointer)

    result_a = await graph_a.ainvoke(
        {"task": "test_task", "step_results": [], "current_topology": "graph_a", "final_output": None},
        config,
    )

    assert "__interrupt__" in result_a, f"Expected interrupt, got: {result_a}"
    state_a = graph_a.get_state(config).values
    assert state_a["step_results"] == ["a1_done"]
    print("TEST 1: PASS - Graph A interrupted after a1")

    # --- Project -> Graph B ---
    projected = project_a_to_b(state_a)

    builder_b = StateGraph(SharedState)
    builder_b.add_node("b1", node_b1)
    builder_b.add_edge(START, "b1")
    graph_b = builder_b.compile(checkpointer=checkpointer)
    graph_b.update_state(config, projected)

    result_b = await graph_b.ainvoke(None, config)
    assert "__interrupt__" not in result_b, f"Unexpected interrupt in B: {result_b}"

    final = graph_b.get_state(config).values
    assert "a1_done" in final["step_results"], f"Missing a1: {final}"
    assert "b1_done" in final["step_results"], f"Missing b1: {final}"
    assert final["current_topology"] == "graph_b"
    assert final["final_output"] == "graph_b_complete"
    print(f"TEST 1: PASS - step_results={final['step_results']}")
    print("TEST 1: PASSED\n")


# ===== Test 2: Multi-step degrade (A -> B -> C) =====
async def test_multi_step_degrade():
    checkpointer = MemorySaver()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # --- Graph A ---
    builder_a = StateGraph(SharedState)
    builder_a.add_node("a1", node_a1)
    builder_a.add_node("interrupt", interrupt_node)
    builder_a.add_node("a2", node_a2)
    builder_a.add_edge(START, "a1")
    builder_a.add_edge("a1", "interrupt")
    builder_a.add_edge("interrupt", "a2")
    graph_a = builder_a.compile(checkpointer=checkpointer)

    result_a = await graph_a.ainvoke(
        {"task": "test_task", "step_results": [], "current_topology": "graph_a", "final_output": None},
        config,
    )
    assert "__interrupt__" in result_a
    state_a = graph_a.get_state(config).values
    print("TEST 2: PASS - Graph A interrupted")

    # --- Degrade A -> B (but B also has interrupt) ---
    def b_interrupt(state: SharedState):
        """Graph B also has a budget gate that fires."""
        interrupt({"reason": "STRUCTURAL_DEGRADE", "from_topology": "graph_b"})
        return {}

    builder_b = StateGraph(SharedState)
    builder_b.add_node("b1", node_b1)
    builder_b.add_node("b_interrupt", b_interrupt)
    builder_b.add_edge(START, "b1")
    builder_b.add_edge("b1", "b_interrupt")
    graph_b = builder_b.compile(checkpointer=checkpointer)

    projected_b = project_a_to_b(state_a)
    graph_b.update_state(config, projected_b)
    result_b = await graph_b.ainvoke(None, config)

    assert "__interrupt__" in result_b, f"Expected second interrupt, got: {result_b}"
    state_b = graph_b.get_state(config).values
    print(f"TEST 2: PASS - Graph B interrupted. step_results={state_b['step_results']}")

    # --- Degrade B -> C ---
    builder_c = StateGraph(SharedState)
    builder_c.add_node("c1", node_c1)
    builder_c.add_edge(START, "c1")
    graph_c = builder_c.compile(checkpointer=checkpointer)

    projected_c = project_b_to_c(state_b)
    graph_c.update_state(config, projected_c)
    result_c = await graph_c.ainvoke(None, config)

    assert "__interrupt__" not in result_c
    final = graph_c.get_state(config).values
    assert "a1_done" in final["step_results"], f"Missing a1: {final}"
    assert "b1_done" in final["step_results"], f"Missing b1: {final}"
    assert "c1_done" in final["step_results"], f"Missing c1: {final}"
    assert final["current_topology"] == "graph_c"
    assert final["final_output"] == "graph_c_complete"
    print(f"TEST 2: PASS - step_results={final['step_results']}")
    print("TEST 2: PASSED\n")


# ===== Test 3: Projection preserves accumulated work =====
async def test_projection_preserves_state():
    """Verify that NOT including annotated fields in projection
    preserves the checkpoint value (no duplication)."""
    checkpointer = MemorySaver()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Graph A: writes to step_results once, then intercepts
    def node_write(state: SharedState):
        return {"step_results": ["write_a"]}

    def node_intercept(state: SharedState):
        interrupt("halt")
        return {}

    builder = StateGraph(SharedState)
    builder.add_node("write_a", node_write)
    builder.add_node("intercept", node_intercept)
    builder.add_edge(START, "write_a")
    builder.add_edge("write_a", "intercept")
    graph_a = builder.compile(checkpointer=checkpointer)

    await graph_a.ainvoke(
        {"task": "preserve_test", "step_results": [], "current_topology": "graph_a", "final_output": None},
        config,
    )
    state = graph_a.get_state(config).values
    assert state["step_results"] == ["write_a"]
    print(f"TEST 3: After graph A, step_results={state['step_results']}")

    # Project WITHOUT including step_results (the correct pattern)
    projected = {"task": "preserve_test", "current_topology": "graph_b", "final_output": None}

    builder_b = StateGraph(SharedState)
    builder_b.add_node("b1", node_b1)
    builder_b.add_edge(START, "b1")
    graph_b = builder_b.compile(checkpointer=checkpointer)
    graph_b.update_state(config, projected)

    # Verify: step_results should be preserved from checkpoint (not duplicated)
    final = graph_b.get_state(config).values
    assert final["step_results"] == ["write_a"], f"Expected ['write_a'], got {final['step_results']}"
    print("TEST 3: PASS - step_results preserved without duplication")
    print("TEST 3: PASSED\n")


# ===== Test 4: CRITICAL on single -> degraded completion =====
async def test_critical_on_single():
    checkpointer = MemorySaver()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Single topology graph with budget gate
    def single_start(state: SharedState):
        return {"step_results": ["single_work_done"]}

    def budget_gate_single(state: SharedState):
        interrupt({"reason": "CRITICAL", "from_topology": "single"})
        return {}

    def single_final(state: SharedState):
        return {"final_output": "best_available_output"}

    builder = StateGraph(SharedState)
    builder.add_node("start", single_start)
    builder.add_node("budget_gate", budget_gate_single)
    builder.add_node("final", single_final)
    builder.add_edge(START, "start")
    builder.add_edge("start", "budget_gate")
    builder.add_edge("budget_gate", "final")
    graph = builder.compile(checkpointer=checkpointer)

    result = await graph.ainvoke(
        {"task": "critical_test", "step_results": [], "current_topology": "single", "final_output": None},
        config,
    )
    assert "__interrupt__" in result
    state = graph.get_state(config).values
    assert state["step_results"] == ["single_work_done"]

    # Project: only set non-annotated fields (don't include step_results)
    projected = {"task": "critical_test", "current_topology": "single", "final_output": "best_available_output"}

    builder2 = StateGraph(SharedState)
    builder2.add_node("final", single_final)
    builder2.add_edge(START, "final")
    graph2 = builder2.compile(checkpointer=checkpointer)
    graph2.update_state(config, projected)

    result2 = await graph2.ainvoke(None, config)
    assert "__interrupt__" not in result2
    final = graph2.get_state(config).values
    assert final["final_output"] == "best_available_output"
    assert "single_work_done" in final["step_results"]
    print("TEST 4: PASS - CRITICAL on single produces degraded completion")
    print("TEST 4: PASSED\n")


# ===== Test 5: Interrupt returns correct value =====
async def test_interrupt_value():
    checkpointer = MemorySaver()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    def budget_gate_node(state: SharedState):
        interrupt({"reason": "TIER_DOWNGRADE", "from_topology": "ensemble", "current_band": "HEALTHY"})
        return {}

    builder = StateGraph(SharedState)
    builder.add_node("gate", budget_gate_node)
    builder.add_edge(START, "gate")
    graph = builder.compile(checkpointer=checkpointer)

    result = await graph.ainvoke(
        {"task": "interrupt_val_test", "step_results": [], "current_topology": "ensemble", "final_output": None},
        config,
    )
    assert "__interrupt__" in result
    interrupts = result["__interrupt__"]
    assert len(interrupts) == 1
    assert interrupts[0].value == {"reason": "TIER_DOWNGRADE", "from_topology": "ensemble", "current_band": "HEALTHY"}
    print(f"TEST 5: PASS - Interrupt value preserved: {interrupts[0].value}")
    print("TEST 5: PASSED\n")


# ===== Test 6: Multiple step_results accumulate across graphs =====
async def test_step_results_accumulate():
    """Verify that step_results from multiple graph phases accumulate correctly
    when projection excludes the annotated field."""
    checkpointer = MemorySaver()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Graph A: two nodes that both write to step_results
    def node_a1(state: SharedState):
        return {"step_results": ["a_step1"]}

    def node_a2(state: SharedState):
        return {"step_results": ["a_step2"]}

    builder = StateGraph(SharedState)
    builder.add_node("a1", node_a1)
    builder.add_node("a2", node_a2)
    builder.add_edge(START, "a1")
    builder.add_edge("a1", "a2")
    graph_a = builder.compile(checkpointer=checkpointer)

    await graph_a.ainvoke(
        {"task": "accum_test", "step_results": [], "current_topology": "graph_a", "final_output": None},
        config,
    )
    state = graph_a.get_state(config).values
    assert state["step_results"] == ["a_step1", "a_step2"]
    print(f"TEST 6: Graph A completed with step_results={state['step_results']}")

    # Now intercept mid-execution of a second graph phase
    checkpointer2 = MemorySaver()
    thread_id2 = str(uuid.uuid4())
    config2 = {"configurable": {"thread_id": thread_id2}}

    def node_b1(state: SharedState):
        return {"step_results": ["b_step1"]}

    def node_b_intercept(state: SharedState):
        interrupt("halt")
        return {}

    builder_b = StateGraph(SharedState)
    builder_b.add_node("b1", node_b1)
    builder_b.add_node("b_intercept", node_b_intercept)
    builder_b.add_edge(START, "b1")
    builder_b.add_edge("b1", "b_intercept")
    graph_b = builder_b.compile(checkpointer=checkpointer2)

    await graph_b.ainvoke(
        {"task": "accum_test", "step_results": [], "current_topology": "graph_b", "final_output": None},
        config2,
    )
    state_b = graph_b.get_state(config2).values
    assert state_b["step_results"] == ["b_step1"]

    # Project B (exclude step_results) and resume on Graph C
    projected = {"task": "accum_test", "current_topology": "graph_c", "final_output": None}

    def node_c1(state: SharedState):
        return {"step_results": ["c_step1"], "final_output": "c_done"}

    builder_c = StateGraph(SharedState)
    builder_c.add_node("c1", node_c1)
    builder_c.add_edge(START, "c1")
    graph_c = builder_c.compile(checkpointer=checkpointer2)
    graph_c.update_state(config2, projected)

    await graph_c.ainvoke(None, config2)
    final = graph_c.get_state(config2).values
    assert final["step_results"] == ["b_step1", "c_step1"], f"Expected accumulation, got {final['step_results']}"
    print(f"TEST 6: PASS - step_results accumulated: {final['step_results']}")
    print("TEST 6: PASSED\n")


# ===== Run all tests =====
async def main():
    print("=" * 60)
    print("MID-EXECUTION TOPOLOGY CHANGE MECHANISM TESTS")
    print("=" * 60 + "\n")

    tests = [
        ("Single degrade A->B", test_single_degrade),
        ("Multi-step degrade A->B->C", test_multi_step_degrade),
        ("Projection preserves state", test_projection_preserves_state),
        ("CRITICAL on single", test_critical_on_single),
        ("Interrupt value passthrough", test_interrupt_value),
        ("Step results accumulate", test_step_results_accumulate),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            await test_fn()
            passed += 1
        except Exception as e:
            print(f"TEST FAILED [{name}]: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"RESULTS: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\nALL TESTS PASSED")
        print("\nCore mechanism validated:")
        print("  - interrupt() returns __interrupt__ in ainvoke result")
        print("  - update_state resets execution pointer to new graph START")
        print("  - ainvoke(None) resumes on new graph with projected state")
        print("  - step_results accumulate across graph transitions")
        print("  - Annotated fields preserved by checkpoint (no duplication)")
        print("  - Multi-step degrade chain works (A->B->C)")
        print("\nKey insight for projection design:")
        print("  - DO NOT include annotated fields in projection dict")
        print("  - update_state MERGES annotated fields (concat for lists)")
        print("  - Non-annotated fields are replaced directly")
    else:
        print(f"\n{failed} TEST(S) FAILED")
    return failed == 0


if __name__ == "__main__":
    asyncio.run(main())
