"""RL Policy monitoring and management endpoints."""

from fastapi import APIRouter, Depends

from api.middleware.auth import require_auth
from core.redis_client import get_redis
from core.rl_policy import RLPolicy

router = APIRouter(prefix="/rl", tags=["rl"], dependencies=[Depends(require_auth)])


@router.get("/stats")
async def get_rl_stats() -> dict:
    """Get current RL policy statistics."""
    redis = await get_redis()
    if not redis:
        return {"error": "Redis not available"}
    rl = RLPolicy(redis)
    return await rl.get_stats()


@router.get("/overrides")
async def get_rl_overrides(limit: int = 50) -> list[dict]:
    """Get recent RL override decisions."""
    redis = await get_redis()
    if not redis:
        return []
    rl = RLPolicy(redis)
    return await rl.get_override_history(limit=limit)


@router.get("/rewards")
async def get_rl_rewards(topology: str | None = None, limit: int = 100) -> list[dict]:
    """Get reward history for monitoring."""
    redis = await get_redis()
    if not redis:
        return []
    rl = RLPolicy(redis)
    return await rl.get_reward_history(topology=topology, limit=limit)


@router.post("/reset")
async def reset_rl_policy(confirm: str = "") -> dict:
    """Full reset: flush Redis, truncate SQLite, reset arms to uniform priors.

    Requires confirm=yes to prevent accidental wipes.
    """
    if confirm != "yes":
        return {"error": "Pass confirm=yes to reset RL state"}
    redis = await get_redis()
    if not redis:
        return {"error": "Redis not available"}
    rl = RLPolicy(redis)
    success = await rl.reset()
    if success:
        stats = await rl.get_stats()
        return {"status": "reset", "arms": stats}
    return {"error": "Reset failed"}
