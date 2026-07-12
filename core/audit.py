import asyncio
import collections
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

log = logging.getLogger(__name__)

_MAX_IN_MEMORY_ENTRIES = 10000


class AuditTrail:
    def __init__(self) -> None:
        self._entries: collections.deque[dict[str, Any]] = collections.deque(maxlen=_MAX_IN_MEMORY_ENTRIES)
        from core.config import settings

        self._db_path: str = settings.audit_db_path

    # ------------------------------------------------------------------
    # SQLite lifecycle
    # ------------------------------------------------------------------

    async def init_db(self) -> None:
        """Create the SQLite table and index on startup.  Idempotent."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id     TEXT NOT NULL,
                    event_type  TEXT NOT NULL,
                    timestamp   TEXT NOT NULL,
                    detail      TEXT NOT NULL
                )
                """
            )
            await db.execute("CREATE INDEX IF NOT EXISTS idx_task_id ON audit_events(task_id)")
            await db.commit()

    async def _write_to_db(self, entry: dict[str, Any]) -> None:
        """Fire-and-forget SQLite write — never raises to caller."""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    "INSERT INTO audit_events (task_id, event_type, timestamp, detail) VALUES (?, ?, ?, ?)",
                    (
                        entry["task_id"],
                        entry["event_type"],
                        entry["timestamp"],
                        json.dumps(entry.get("detail", {}), default=str),
                    ),
                )
                await db.commit()
        except Exception as e:
            log.warning("Audit SQLite write failed for task %s: %s", entry.get("task_id"), e)

    async def load_from_db(self, task_id: str) -> list[dict[str, Any]]:
        """
        Fetch audit entries from SQLite for a given task_id.
        Used as a fallback when the in-memory list is empty (e.g. after a restart).
        """
        try:
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(
                    "SELECT event_type, timestamp, detail FROM audit_events WHERE task_id = ? ORDER BY id",
                    (task_id,),
                ) as cursor:
                    rows = await cursor.fetchall()
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

    def record(
        self,
        task_id: str,
        event_type: str,
        detail: dict[str, Any],
    ) -> None:
        entry = {
            "task_id": task_id,
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "detail": detail,
        }
        self._entries.append(entry)
        # Persist asynchronously without blocking the caller
        try:
            asyncio.get_running_loop().create_task(self._write_to_db(entry))
        except RuntimeError:
            pass  # No running event loop (unit tests, CLI scripts)

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
