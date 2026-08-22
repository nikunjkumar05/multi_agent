"""SQLite persistence — tasks and budgets survive restarts.

Write-through: every mutation point in routes/workers mirrors state here.
Load-on-startup: main.py restores the in-memory managers from the DB.

Design notes:
- stdlib sqlite3 only — zero new dependencies.
- Rows store the full record as JSON; updates rewrite the row.
- DB path via BAMAS_DB_PATH (read per-call so tests can redirect).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading

log = logging.getLogger(__name__)

_LOCK = threading.Lock()


def get_db_path() -> str:
    """Resolve the database path at call time (test-friendly)."""
    return os.getenv("BAMAS_DB_PATH", "bamas_middleware.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path())
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            updated_at REAL NOT NULL DEFAULT (strftime('%s','now'))
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS budgets (
            budget_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            updated_at REAL NOT NULL DEFAULT (strftime('%s','now'))
        )
        """
    )
    return conn


def _upsert(table: str, pk_col: str, pk_value: str, data: dict) -> None:
    try:
        with _LOCK, _connect() as conn:
            conn.execute(
                f"INSERT INTO {table} ({pk_col}, data) VALUES (?, ?) "
                f"ON CONFLICT({pk_col}) DO UPDATE SET data = excluded.data, "
                "updated_at = strftime('%s','now')",
                (pk_value, json.dumps(data, default=str)),
            )
    except sqlite3.Error as e:
        # Persistence must never take the API down — log and continue.
        log.error("Persistence write failed (%s/%s): %s", table, pk_value, e)


def save_task(record: dict) -> None:
    _upsert("tasks", "task_id", record.get("task_id", ""), record)


def delete_task(task_id: str) -> None:
    try:
        with _LOCK, _connect() as conn:
            conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
    except sqlite3.Error as e:
        log.error("Persistence delete failed (tasks/%s): %s", task_id, e)


def save_budget(data: dict) -> None:
    _upsert("budgets", "budget_id", data.get("budget_id", ""), data)


def delete_budget(budget_id: str) -> None:
    try:
        with _LOCK, _connect() as conn:
            conn.execute("DELETE FROM budgets WHERE budget_id = ?", (budget_id,))
    except sqlite3.Error as e:
        log.error("Persistence delete failed (budgets/%s): %s", budget_id, e)


def load_all() -> tuple[list[dict], list[dict]]:
    """Return ([task_records], [budget_dicts]) for startup restore."""
    tasks: list[dict] = []
    budgets: list[dict] = []
    if not os.path.exists(get_db_path()):
        return tasks, budgets
    try:
        with _LOCK, _connect() as conn:
            for row in conn.execute("SELECT data FROM tasks"):
                try:
                    tasks.append(json.loads(row[0]))
                except json.JSONDecodeError:
                    log.warning("Skipping corrupt task row")
            for row in conn.execute("SELECT data FROM budgets"):
                try:
                    budgets.append(json.loads(row[0]))
                except json.JSONDecodeError:
                    log.warning("Skipping corrupt budget row")
    except sqlite3.Error as e:
        log.error("Persistence load failed: %s", e)
    log.info("Persistence loaded: %d tasks, %d budgets", len(tasks), len(budgets))
    return tasks, budgets
