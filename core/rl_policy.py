import random

import redis.asyncio as aioredis

TOPOLOGIES = ["single", "pipeline", "supervisor", "fanout", "ensemble"]

CODE_KEYWORDS = ["write", "code", "function", "implement", "create", "generate", "build", "develop", "script"]
RESEARCH_KEYWORDS = ["explain", "research", "compare", "why", "how does", "describe", "summarize", "review"]
DATA_KEYWORDS = ["analyze", "data", "parallel", "bulk", "process"]
VERIFY_KEYWORDS = ["verify", "audit", "validate", "critical", "security", "proof"]

# Context boosts: when a feature is active, multiply the arm's sample by this weight.
# Higher weight = more likely to be selected for that task type.
CONTEXT_WEIGHTS: dict[str, dict[str, float]] = {
    "is_code":     {"pipeline": 2.0, "single": 0.5, "supervisor": 0.7, "fanout": 0.8, "ensemble": 0.6},
    "is_research": {"supervisor": 2.0, "pipeline": 0.7, "single": 0.5, "fanout": 0.8, "ensemble": 0.6},
    "is_data":     {"fanout": 2.0, "pipeline": 0.8, "single": 0.5, "supervisor": 0.7, "ensemble": 0.6},
    "is_verify":   {"ensemble": 2.0, "pipeline": 0.6, "single": 0.5, "supervisor": 0.7, "fanout": 0.8},
}

MIN_TASKS_TO_LEARN = 5


class RLPolicy:
    def __init__(self, redis: aioredis.Redis) -> None:
        self.redis = redis
        self.arms: dict[str, dict[str, float]] = {}

    async def load(self) -> None:
        self.arms = {}
        for topo in TOPOLOGIES:
            arm_key = f"rl_policy:arm:{topo}"
            raw = await self.redis.hgetall(arm_key)
            if raw:
                self.arms[topo] = {
                    "alpha": float(raw.get(b"alpha", raw.get("alpha", 1.0))),
                    "beta": float(raw.get(b"beta", raw.get("beta", 1.0))),
                }
            else:
                self.arms[topo] = {"alpha": 1.0, "beta": 1.0}

    async def _get_total_tasks(self) -> int:
        val = await self.redis.get("rl_policy:total_tasks")
        return int(val) if val else 0

    async def _increment_tasks(self) -> None:
        await self.redis.incr("rl_policy:total_tasks")

    def _extract_features(self, task: str) -> dict[str, bool]:
        task_lower = task.lower()
        return {
            "is_code": any(kw in task_lower for kw in CODE_KEYWORDS),
            "is_research": any(kw in task_lower for kw in RESEARCH_KEYWORDS),
            "is_data": any(kw in task_lower for kw in DATA_KEYWORDS),
            "is_verify": any(kw in task_lower for kw in VERIFY_KEYWORDS),
        }

    def _thompson_sample(self, context_weights: dict[str, float] | None = None) -> str:
        samples = {}
        for topo, params in self.arms.items():
            sample = random.betavariate(params["alpha"], params["beta"])
            if context_weights and topo in context_weights:
                sample *= context_weights[topo]
            samples[topo] = sample
        return max(samples, key=samples.get)

    def _compute_context_weights(self, features: dict[str, bool]) -> dict[str, float]:
        weights = {t: 1.0 for t in TOPOLOGIES}
        for feature, active in features.items():
            if active and feature in CONTEXT_WEIGHTS:
                for topo, multiplier in CONTEXT_WEIGHTS[feature].items():
                    weights[topo] *= multiplier
        return weights

    async def select_topology(self, task: str, budget_band: str) -> str | None:
        total = await self._get_total_tasks()
        if total < MIN_TASKS_TO_LEARN:
            return None

        await self.load()
        features = self._extract_features(task)
        weights = self._compute_context_weights(features)
        return self._thompson_sample(weights)

    async def reward(self, topology: str, quality: float, cost_efficiency: float) -> None:
        combined = quality * 0.7 + cost_efficiency * 0.3

        arm_key = f"rl_policy:arm:{topology}"
        if combined > 0.5:
            await self.redis.hincrbyfloat(arm_key, "alpha", combined)
        else:
            await self.redis.hincrbyfloat(arm_key, "beta", 1.0 - combined)

        await self.redis.incr("rl_policy:total_tasks")
