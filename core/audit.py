"""
Audit trail — records every significant event during task execution.

Dual-layer architecture:
  - In-memory deque for fast reads (last 10,000 events)
  - Database persistence via core/db.py (PostgreSQL or SQLite)

Public API unchanged: record(), record_topology_decision(),
record_budget_band(), record_degradation(), get_task_audit().
"""

import asyncio
import collections
import json
import logging
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger(__name__)

_MAX_IN_MEMORY_ENTRIES = 10000

# ── DDL (PostgreSQL and SQLite variants) ───────────────────────────────

_DDL_POSTGRESQL = """
CREATE TABLE IF NOT EXISTS audit_events (
    id          SERIAL PRIMARY KEY,
    task_id     TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    detail      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_task_id ON audit_events(task_id);
"""

_DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS audit_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    detail      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_task_id ON audit_events(task_id);
"""


class AuditTrail:
    def __init__(self) -> None:
        self._entries: collections.deque[dict[str, Any]] = collections.deque(maxlen=_MAX_IN_MEMORY_ENTRIES)
        self._db_initialized = False

    # ------------------------------------------------------------------
    # Database lifecycle
    # ------------------------------------------------------------------

    async def _ensure_db(self) -> None:
        """Initialize the database table on first use. Idempotent."""
        if self._db_initialized:
            return
        try:
            from core.db import get_db
            db = await get_db()
            ddl = _DDL_POSTGRESQL if db.backend == "postgresql" else _DDL_SQLITE
            await db.init_db(ddl)
            self._db_initialized = True
        except Exception as e:
            log.warning("Audit DB init failed: %s", e)

    async def _write_to_db(self, entry: dict[str, Any]) -> None:
        """Fire-and-forget database write — never raises to caller.

        Retries up to 3 times with exponential backoff on transient failures.
        """
        try:
            from core.db import get_db
            db = await get_db()

            sql = "INSERT INTO audit_events (task_id, event_type, timestamp, detail) VALUES ($1, $2, $3, $4)" if db.backend == "postgresql" else "INSERT INTO audit_events (task_id, event_type, timestamp, detail) VALUES (?, ?, ?, ?)"
            params = (
                entry["task_id"],
                entry["event_type"],
                entry["timestamp"],
                json.dumps(entry.get("detail", {}), default=str),
            )

            for attempt in range(3):
                try:
                    await db.execute(sql, params)
                    return
                except Exception as e:
                    if attempt == 2:
                        log.warning(
                            "Audit DB write failed after 3 attempts for task %s: %s",
                            entry.get("task_id"),
                            e,
                        )
                    else:
                        await asyncio.sleep(0.1 * (2 ** attempt))
        except Exception as e:
            log.warning("Audit DB write failed for task %s: %s", entry.get("task_id"), e)

    async def load_from_db(self, task_id: str) -> list[dict[str, Any]]:
        """
        Fetch audit entries from the database for a given task_id.
        Used as a fallback when the in-memory deque is empty (e.g. after restart).
        """
        try:
            from core.db import get_db
            db = await get_db()

            sql = "SELECT event_type, timestamp, detail FROM audit_events WHERE task_id = $1 ORDER BY id" if db.backend == "postgresql" else "SELECT event_type, timestamp, detail FROM audit_events WHERE task_id = ? ORDER BY id"
            rows = await db.fetchall(sql, (task_id,))
            return [
                {
                    "task_id": task_id,
                    "event_type": row[0],
                    "timestamp": row[1],
                    "detail": json.loads(row[2]),
                }
                for row in rows
            ]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Public API (unchanged)
    # ------------------------------------------------------------------

    def record(
        self,
        task_id: str,
        event_type: str,
        detail: dict[str, Any],
    ) -> None:
        entry = {
            "task_id": task_id,
            "event_type": event_type,
            "timestamp": datetime.now(UTC).isoformat(),
            "detail": detail,
        }
        self._entries.append(entry)
        # Persist asynchronously without blocking the caller
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._safe_db_write(entry))
        except RuntimeError:
            pass  # No running event loop (unit tests, CLI scripts)

    async def _safe_db_write(self, entry: dict[str, Any]) -> None:
        """Write to DB with lazy initialization."""
        await self._ensure_db()
        await self._write_to_db(entry)

    def record_topology_decision(
        self,
        task_id: str,
        topology: str,
        model_tiers: dict[str, str],
        budget: float,
        rationale: str,
        alternatives: list[dict[str, str]],
    ) -> None:
        self.record(
            task_id=task_id,
            event_type="topology_decision",
            detail={
                "topology": topology,
                "model_tiers": model_tiers,
                "budget_at_decision": budget,
                "rationale": rationale,
                "alternatives_considered": alternatives,
            },
        )

    def record_budget_band(
        self,
        task_id: str,
        band: str,
        remaining_budget: float,
        action: str,
    ) -> None:
        self.record(
            task_id=task_id,
            event_type="budget_band_crossed",
            detail={
                "band": band,
                "remaining_budget": remaining_budget,
                "action": action,
            },
        )

    def record_degradation(
        self,
        task_id: str,
        from_topology: str,
        to_topology: str,
        reason: str,
    ) -> None:
        self.record(
            task_id=task_id,
            event_type="structural_degradation",
            detail={
                "from_topology": from_topology,
                "to_topology": to_topology,
                "reason": reason,
            },
        )

    def get_task_audit(self, task_id: str) -> list[dict[str, Any]]:
        return [e for e in self._entries if e["task_id"] == task_id]

    def to_json(self, task_id: str) -> str:
        return json.dumps(self.get_task_audit(task_id), indent=2, default=str)


audit_trail = AuditTrail()


def get_audit_trail() -> AuditTrail:
    return audit_trail
