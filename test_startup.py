"""Quick smoke test — runs the full pipeline without uvicorn."""
import asyncio
import traceback


async def main():
    errors = []

    # 1. Test imports
    print("[1/6] Testing imports...", end=" ")
    try:
        from core.config import settings
        from core.budget import BudgetTracker, BudgetBand, get_band_from_state
        from core.db import get_db, close_db
        from core.audit import audit_trail
        from core.redis_client import get_redis, close_redis
        from core.projections import project_state, get_valid_projection_edges
        from core.degrader import degrade_topology
        from core.optimizer import CostTierOptimizer
        from agent.graph import run_task
        from agent.orchestrator import run_task_with_degradation
        from api.main import app
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return

    # 2. Test budget tracker
    print("[2/6] Testing BudgetTracker...", end=" ")
    try:
        bt = BudgetTracker(max_cost_usd=1.0)
        assert bt.get_band() == BudgetBand.HEALTHY
        bt.record_usage(tokens=100, cost=0.80)
        assert bt.get_band() == BudgetBand.TIER_DOWNGRADE
        bt.record_usage(tokens=100, cost=0.15)
        assert bt.get_band() == BudgetBand.STRUCTURAL_DEGRADE
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        errors.append(e)

    # 3. Test DB
    print("[3/6] Testing database...", end=" ")
    try:
        db = await get_db()
        print(f"OK (backend={db.backend})")
    except Exception as e:
        print(f"FAIL: {e}")
        errors.append(e)

    # 4. Test projections
    print("[4/6] Testing projections...", end=" ")
    try:
        edges = get_valid_projection_edges()
        assert len(edges) == 10, f"Expected 10 edges, got {len(edges)}"
        print(f"OK ({len(edges)} edges)")
    except Exception as e:
        print(f"FAIL: {e}")
        errors.append(e)

    # 5. Test full task execution (requires Mistral API key)
    print("[5/6] Testing full task execution...", end=" ")
    try:
        budget = BudgetTracker(max_cost_usd=0.10)
        result = await run_task(
            task="What is 2+2? Answer with just the number.",
            budget=budget,
            topology_override="single",
        )
        print(f"OK (status={result['status']}, topology={result['topology']})")
        print(f"       result: {result.get('final_result', 'N/A')[:100]}")
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        errors.append(e)

    # 6. Cleanup
    print("[6/6] Cleanup...", end=" ")
    try:
        await close_redis()
        await close_db()
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")

    print()
    if errors:
        print(f"FAILED — {len(errors)} error(s)")
        for i, e in enumerate(errors, 1):
            print(f"  {i}. {e}")
    else:
        print("ALL PASSED")


if __name__ == "__main__":
    asyncio.run(main())
