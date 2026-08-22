"""
RL Policy module — Contextual Thompson Sampling for topology selection.

Production-grade implementation with:
- Redis for fast reads (hot state)
- PostgreSQL/SQLite for persistent storage (cold state + audit trail)
- Confidence-gated override to prevent wrong topology selections
- Async-safe via core.db abstraction (supports both PostgreSQL and SQLite)
"""

import json
import os
import random
from datetime import datetime

import redis.asyncio as aioredis

TOPOLOGIES = ["single", "pipeline", "supervisor", "fanout", "ensemble"]

# Expanded keywords for better task type detection
CODE_KEYWORDS = ["function", "implement", "code", "class", "script", "write", "create", "build", "develop", "program"]
RESEARCH_KEYWORDS = ["explain", "research", "compare", "why", "how does", "describe", "summarize", "review",
                     "way to", "how to", "what are", "is there", "options for", "alternatives"]
DATA_KEYWORDS = ["analyze", "data", "parallel", "bulk", "process", "dataset", "statistics", "chart", "graph"]
VERIFY_KEYWORDS = ["verify", "audit", "validate", "critical", "security", "proof", "check", "review code"]

# Context boosts: when a feature is active, multiply the arm's sample by this weight.
CONTEXT_WEIGHTS: dict[str, dict[str, float]] = {
    "is_code": {"pipeline": 2.0, "single": 0.5, "supervisor": 0.7, "fanout": 0.8, "ensemble": 0.6},
    "is_research": {"supervisor": 2.0, "pipeline": 0.7, "single": 0.5, "fanout": 0.8, "ensemble": 0.6},
    "is_data": {"fanout": 2.0, "pipeline": 0.8, "single": 0.5, "supervisor": 0.7, "ensemble": 0.6},
    "is_verify": {"ensemble": 2.0, "pipeline": 0.6, "single": 0.5, "supervisor": 0.7, "fanout": 0.8},
}

# Minimum confidence required for RL to override LLM
RL_MIN_CONFIDENCE_FOR_OVERRIDE = 0.85

_DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PERSIST_FILE = os.path.join(_DATA_DIR, "rl_policy.json")

# ── DDL schemas (dual-backend) ───────────────────────────────────────

_DDL_POSTGRESQL = """
CREATE TABLE IF NOT EXISTS rl_snapshots (
    id SERIAL PRIMARY KEY,
    timestamp TEXT NOT NULL,
    arms_json TEXT NOT NULL,
    total_tasks INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON rl_snapshots(timestamp);

CREATE TABLE IF NOT EXISTS rl_overrides (
    id SERIAL PRIMARY KEY,
    timestamp TEXT NOT NULL,
    task_id TEXT,
    task TEXT,
    llm_topology TEXT,
    rl_topology TEXT,
    confidence REAL,
    task_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_overrides_ts ON rl_overrides(timestamp);
CREATE INDEX IF NOT EXISTS idx_overrides_task_id ON rl_overrides(task_id);

CREATE TABLE IF NOT EXISTS rl_rewards (
    id SERIAL PRIMARY KEY,
    timestamp TEXT NOT NULL,
    topology TEXT NOT NULL,
    reward REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rewards_ts ON rl_rewards(timestamp);
CREATE INDEX IF NOT EXISTS idx_rewards_topo ON rl_rewards(topology);
"""

_DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS rl_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    arms_json TEXT NOT NULL,
    total_tasks INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON rl_snapshots(timestamp);

CREATE TABLE IF NOT EXISTS rl_overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    task_id TEXT,
    task TEXT,
    llm_topology TEXT,
    rl_topology TEXT,
    confidence REAL,
    task_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_overrides_ts ON rl_overrides(timestamp);
CREATE INDEX IF NOT EXISTS idx_overrides_task_id ON rl_overrides(task_id);

CREATE TABLE IF NOT EXISTS rl_rewards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    topology TEXT NOT NULL,
    reward REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rewards_ts ON rl_rewards(timestamp);
CREATE INDEX IF NOT EXISTS idx_rewards_topo ON rl_rewards(topology);
"""


def _convert_params(sql: str, backend: str) -> str:
    """Convert SQLite ? placeholders to PostgreSQL $N placeholders."""
    if backend == "postgresql":
        parts = sql.split("?")
        result = parts[0]
        for i, part in enumerate(parts[1:], 1):
            result += f"${i}{part}"
        return result
    return sql


class RLPolicy:
    def __init__(self, redis: aioredis.Redis, persist_path: str = _PERSIST_FILE) -> None:
        self.redis = redis
        self.persist_path = persist_path
        self.arms: dict[str, dict[str, float]] = {}
        self.total_tasks: int = 0
        self._db_initialized = False

    # ── DB helpers (via shared core.db abstraction) ────────────────────

    async def _ensure_db(self) -> None:
        """Initialize DB schema once per process lifetime."""
        if self._db_initialized:
            return
        from core.db import get_db
        db = await get_db()
        ddl = _DDL_POSTGRESQL if db.backend == "postgresql" else _DDL_SQLITE
        await db.init_db(ddl)
        self._db_initialized = True

    async def _execute(self, sql: str, params: tuple = ()) -> None:
        """Execute a statement via the shared DB abstraction."""
        from core.db import get_db
        db = await get_db()
        sql = _convert_params(sql, db.backend)
        await db.execute(sql, params)

    async def _fetchall(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Fetch all rows via the shared DB abstraction."""
        from core.db import get_db
        db = await get_db()
        sql = _convert_params(sql, db.backend)
        return await db.fetchall(sql, params)

    async def _fetchone(self, sql: str, params: tuple = ()) -> tuple | None:
        """Fetch a single row via the shared DB abstraction."""
        from core.db import get_db
        db = await get_db()
        sql = _convert_params(sql, db.backend)
        return await db.fetchone(sql, params)

    # ── Initialization ─────────────────────────────────────────────────

    async def load(self) -> None:
        """Load RL state from Redis, falling back to file, then DB."""
        if self.redis is None:
            self._load_from_file()
            return
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

        # Fallback chain: Redis → file → DB
        if self.total_tasks == 0:
            self._load_from_file()
        if self.total_tasks == 0:
            await self._load_from_db()

    # ── Feature extraction ─────────────────────────────────────────────

    def _extract_features(self, task: str) -> dict[str, bool]:
        task_lower = task.lower()
        return {
            "is_code": any(kw in task_lower for kw in CODE_KEYWORDS),
            "is_research": any(kw in task_lower for kw in RESEARCH_KEYWORDS),
            "is_data": any(kw in task_lower for kw in DATA_KEYWORDS),
            "is_verify": any(kw in task_lower for kw in VERIFY_KEYWORDS),
        }

    # ── Thompson Sampling ──────────────────────────────────────────────

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

    def _compute_confidence(self, topology: str) -> float:
        """Confidence = alpha / (alpha + beta). Range: [0, 1]."""
        params = self.arms.get(topology, {"alpha": 1.0, "beta": 1.0})
        alpha = params["alpha"]
        beta = params["beta"]
        return alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5

    # ── Selection ──────────────────────────────────────────────────────

    async def select_topology(self, task: str, budget_band: str) -> str | None:
        """Select topology via Thompson Sampling. Returns None if not confident enough."""
        from core.config import settings

        await self.load()
        if self.total_tasks < settings.rl_min_tasks_for_selection:
            return None

        features = self._extract_features(task)
        weights = self._compute_context_weights(features)
        chosen = self._thompson_sample(weights)

        confidence = self._compute_confidence(chosen)
        if confidence < RL_MIN_CONFIDENCE_FOR_OVERRIDE:
            return None  # Not confident enough to override

        return chosen

    # ── Reward update ──────────────────────────────────────────────────

    async def reward(self, topology: str, quality: float, cost_efficiency: float) -> None:
        """Update RL state in Redis and persist to DB."""
        from core.config import settings

        combined = quality * settings.rl_quality_weight + cost_efficiency * settings.rl_cost_efficiency_weight

        # Update Redis (fast path) — skip if Redis unavailable
        if self.redis is not None:
            arm_key = f"rl_policy:arm:{topology}"
            if combined > 0.5:
                await self.redis.hincrbyfloat(arm_key, "alpha", combined)
            else:
                await self.redis.hincrbyfloat(arm_key, "beta", 1.0 - combined)
            await self.redis.incr("rl_policy:total_tasks")

        self.total_tasks += 1

        # Update in-memory state
        self.arms[topology] = self.arms.get(topology, {"alpha": 1.0, "beta": 1.0})
        if combined > 0.5:
            self.arms[topology]["alpha"] += combined
        else:
            self.arms[topology]["beta"] += 1.0 - combined

        # Persist to file and DB (async, non-blocking)
        self._save_to_file()
        await self._save_reward_to_db(topology, combined)

    # ── File persistence (fast fallback) ───────────────────────────────

    def _load_from_file(self) -> None:
        try:
            with open(self.persist_path) as f:
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

    # ── DB persistence (durable + audit) ───────────────────────────────

    async def _load_from_db(self) -> None:
        """Load RL state from the most recent DB snapshot."""
        try:
            await self._ensure_db()
            rows = await self._fetchall(
                "SELECT arms_json, total_tasks FROM rl_snapshots ORDER BY id DESC LIMIT 1"
            )
            if rows:
                self.arms = json.loads(rows[0][0])
                self.total_tasks = rows[0][1]
        except Exception:
            pass

    async def _save_reward_to_db(self, topology: str, reward: float) -> None:
        """Append a reward record for historical tracking."""
        try:
            await self._ensure_db()
            await self._execute(
                "INSERT INTO rl_rewards (timestamp, topology, reward) VALUES (?, ?, ?)",
                (datetime.utcnow().isoformat(), topology, reward),
            )
        except Exception:
            pass  # DB errors should never crash the system

    async def save_snapshot(self) -> int | None:
        """Save current RL state as a snapshot. Returns snapshot ID (None for PostgreSQL)."""
        try:
            await self._ensure_db()
            await self._execute(
                "INSERT INTO rl_snapshots (timestamp, arms_json, total_tasks) VALUES (?, ?, ?)",
                (datetime.utcnow().isoformat(), json.dumps(self.arms), self.total_tasks),
            )
            return None  # PostgreSQL doesn't return lastrowid via our abstraction
        except Exception:
            return None

    async def rollback(self, snapshot_id: int | None = None) -> bool:
        """
        Rollback RL state to a previous snapshot.
        Syncs to both Redis and file for consistency.
        """
        try:
            await self._ensure_db()

            if snapshot_id:
                rows = await self._fetchall(
                    "SELECT arms_json, total_tasks FROM rl_snapshots WHERE id = ?",
                    (snapshot_id,),
                )
            else:
                rows = await self._fetchall(
                    "SELECT arms_json, total_tasks FROM rl_snapshots ORDER BY id DESC LIMIT 1"
                )

            if not rows:
                return False

            self.arms = json.loads(rows[0][0])
            self.total_tasks = rows[0][1]

            # Sync back to Redis (only if available)
            if self.redis is not None:
                pipe = await self.redis.pipeline()
                for topo in TOPOLOGIES:
                    params = self.arms.get(topo, {"alpha": 1.0, "beta": 1.0})
                    pipe.hset(f"rl_policy:arm:{topo}", mapping={
                        "alpha": str(params["alpha"]),
                        "beta": str(params["beta"]),
                    })
                pipe.set("rl_policy:total_tasks", self.total_tasks)
                await pipe.execute()

            # Sync back to file
            self._save_to_file()
            return True
        except Exception:
            return False

    # ── Full reset ─────────────────────────────────────────────────────

    async def reset(self) -> bool:
        """
        Full reset: flush Redis, truncate DB, reset arms to uniform priors.
        Saves a fresh snapshot so rollback has a clean starting point.

        Writes fresh state BEFORE deleting old state. If anything fails
        mid-way, old state is still recoverable via load().
        """
        try:
            # Phase 1: Write fresh state to Redis first
            if self.redis is not None:
                pipe = await self.redis.pipeline()
                for topo in TOPOLOGIES:
                    pipe.hset(f"rl_policy:arm:{topo}", mapping={
                        "alpha": "1.0", "beta": "1.0",
                    })
                pipe.set("rl_policy:total_tasks", 0)
                await pipe.execute()

            # Phase 2: Truncate DB and write fresh snapshot
            await self._ensure_db()
            await self._execute("DELETE FROM rl_snapshots")
            await self._execute("DELETE FROM rl_overrides")
            await self._execute("DELETE FROM rl_rewards")

            # Phase 3: Now safe to flush old Redis keys (fresh keys already written)
            if self.redis is not None:
                keys = []
                async for key in self.redis.scan_iter("rl_policy:*"):
                    keys.append(key)
                fresh_keys = {f"rl_policy:arm:{t}" for t in TOPOLOGIES} | {"rl_policy:total_tasks"}
                old_keys = [k for k in keys if k not in fresh_keys]
                if old_keys:
                    await self.redis.delete(*old_keys)

            # Phase 4: Reset in-memory, save file and snapshot
            self.arms = {topo: {"alpha": 1.0, "beta": 1.0} for topo in TOPOLOGIES}
            self.total_tasks = 0
            self._save_to_file()
            await self.save_snapshot()
            return True
        except Exception:
            return False

    # ── Audit trail ────────────────────────────────────────────────────

    async def log_override(
        self,
        task_id: str,
        task: str,
        llm_topology: str,
        rl_topology: str,
        confidence: float,
        task_type: str,
    ) -> None:
        """Log an RL override decision for monitoring."""
        try:
            await self._ensure_db()
            await self._execute(
                "INSERT INTO rl_overrides (timestamp, task_id, task, llm_topology, rl_topology, confidence, task_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), task_id, task[:200], llm_topology, rl_topology, confidence, task_type),
            )
        except Exception:
            pass

    async def get_override_history(self, limit: int = 50) -> list[dict]:
        """Get recent RL override decisions."""
        try:
            await self._ensure_db()
            rows = await self._fetchall(
                "SELECT id, timestamp, task_id, task, llm_topology, rl_topology, confidence, task_type "
                "FROM rl_overrides ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return [
                dict(zip(["id", "timestamp", "task_id", "task", "llm_topology", "rl_topology", "confidence", "task_type"], row))
                for row in rows
            ]
        except Exception:
            return []

    async def get_reward_history(self, topology: str | None = None, limit: int = 100) -> list[dict]:
        """Get reward history for monitoring."""
        try:
            await self._ensure_db()
            if topology:
                rows = await self._fetchall(
                    "SELECT id, timestamp, topology, reward FROM rl_rewards WHERE topology = ? ORDER BY id DESC LIMIT ?",
                    (topology, limit),
                )
            else:
                rows = await self._fetchall(
                    "SELECT id, timestamp, topology, reward FROM rl_rewards ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
            return [
                dict(zip(["id", "timestamp", "topology", "reward"], row))
                for row in rows
            ]
        except Exception:
            return []

    async def get_stats(self) -> dict:
        """Get RL policy statistics for monitoring dashboard."""
        await self.load()
        from core.config import settings
        stats = {}
        for topo in TOPOLOGIES:
            params = self.arms.get(topo, {"alpha": 1.0, "beta": 1.0})
            stats[topo] = {
                "alpha": params["alpha"],
                "beta": params["beta"],
                "confidence": self._compute_confidence(topo),
                "expected_reward": params["alpha"] / (params["alpha"] + params["beta"]),
            }
        stats["total_tasks"] = self.total_tasks
        stats["min_tasks_for_override"] = settings.rl_min_tasks_for_override
        stats["min_confidence_for_override"] = RL_MIN_CONFIDENCE_FOR_OVERRIDE
        return stats
