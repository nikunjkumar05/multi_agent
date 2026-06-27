import json
import os
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

_DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PERSIST_FILE = os.path.join(_DATA_DIR, "rl_policy.json")


class RLPolicy:
    def __init__(self, redis: aioredis.Redis, persist_path: str = _PERSIST_FILE) -> None:
        self.redis = redis
        self.persist_path = persist_path
        self.arms: dict[str, dict[str, float]] = {}
        self.total_tasks: int = 0

    async def load(self) -> None:
        pipe = await self.redis.pipeline()
        for topo in TOPOLOGIES:
            pipe.hgetall(f"rl_policy:arm:{topo}")
        pipe.get("rl_policy:total_tasks")
        results = await pipe.execute()

        *arm_results, task_count = results
        self.total_tasks = int(task_count) if task_count else 0

        self.arms = {}
        for topo, raw in zip(TOPOLOGIES, arm_results):
            if raw:
                self.arms[topo] = {
                    "alpha": float(raw.get("alpha", 1.0)),
                    "beta": float(raw.get("beta", 1.0)),
                }
            else:
                self.arms[topo] = {"alpha": 1.0, "beta": 1.0}

        if self.total_tasks == 0:
            self._load_from_file()

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
        await self.load()
        if self.total_tasks < MIN_TASKS_TO_LEARN:
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
        self.total_tasks += 1

        self.arms[topology] = self.arms.get(topology, {"alpha": 1.0, "beta": 1.0})
        if combined > 0.5:
            self.arms[topology]["alpha"] += combined
        else:
            self.arms[topology]["beta"] += (1.0 - combined)
        self._save_to_file()

    def _load_from_file(self) -> None:
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
            self.arms = data.get("arms", {})
            self.total_tasks = data.get("total_tasks", 0)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save_to_file(self) -> None:
        data = {"arms": self.arms, "total_tasks": self.total_tasks}
        tmp = self.persist_path + ".tmp"
        with open(tmp, "w") as f:
            f.write(json.dumps(data))
        os.replace(tmp, self.persist_path)
