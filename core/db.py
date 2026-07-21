"""
Async database abstraction — PostgreSQL (asyncpg) or SQLite (aiosqlite).

Usage:
    from core.db import get_db, init_db
    db = await get_db()
    await db.execute("INSERT ...", (param1,))
    rows = await db.fetchall("SELECT ...", (param1,))
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

log = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY = 0.5  # seconds, doubles each attempt


# ── Abstract interface ─────────────────────────────────────────────────


class AsyncDB(ABC):
    @abstractmethod
    async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None: ...

    @abstractmethod
    async def executemany(self, sql: str, params_list: list[tuple[Any, ...]]) -> None: ...

    @abstractmethod
    async def fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> tuple[Any, ...] | None: ...

    @abstractmethod
    async def fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]: ...

    @abstractmethod
    async def init_db(self, ddl: str) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @property
    @abstractmethod
    def backend(self) -> str: ...


# ── PostgreSQL (asyncpg) ──────────────────────────────────────────────


class PostgresDB(AsyncDB):
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: Any = None

    @property
    def backend(self) -> str:
        return "postgresql"

    async def _get_pool(self) -> Any:
        if self._pool is None:
            import asyncpg
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=1,
                max_size=5,
                command_timeout=10,
            )
        return self._pool

    async def init_db(self, ddl: str) -> None:
        pool = await self._get_pool()
        # DDL statements are not parameterized — execute directly
        async with pool.acquire() as conn:
            await conn.execute(ddl)

    async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(sql, *params)

    async def executemany(self, sql: str, params_list: list[tuple[Any, ...]]) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.executemany(sql, params_list)

    async def fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> tuple[Any, ...] | None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchrow(sql, *params)

    async def fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
            return [tuple(r) for r in rows]

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


# ── SQLite (aiosqlite) ────────────────────────────────────────────────


class SQLiteDB(AsyncDB):
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: Any = None

    @property
    def backend(self) -> str:
        return "sqlite"

    async def _get_conn(self) -> Any:
        if self._conn is None:
            import aiosqlite
            from pathlib import Path
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = await aiosqlite.connect(self._db_path)
        return self._conn

    async def init_db(self, ddl: str) -> None:
        conn = await self._get_conn()
        await conn.executescript(ddl)
        await conn.commit()

    async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        conn = await self._get_conn()
        await conn.execute(sql, params)
        await conn.commit()

    async def executemany(self, sql: str, params_list: list[tuple[Any, ...]]) -> None:
        conn = await self._get_conn()
        await conn.executemany(sql, params_list)
        await conn.commit()

    async def fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> tuple[Any, ...] | None:
        conn = await self._get_conn()
        cursor = await conn.execute(sql, params)
        row = await cursor.fetchone()
        return row

    async def fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        conn = await self._get_conn()
        cursor = await conn.execute(sql, params)
        return await cursor.fetchall()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None


# ── Factory ───────────────────────────────────────────────────────────

_db: AsyncDB | None = None


async def get_db() -> AsyncDB:
    """Get or create the database connection. Singleton per process."""
    global _db
    if _db is not None:
        return _db

    from core.config import settings

    if settings.database_url:
        _db = PostgresDB(settings.database_url)
        log.info("Audit DB: PostgreSQL (%s)", settings.database_url.split("@")[-1] if "@" in settings.database_url else "configured")
    else:
        _db = SQLiteDB(settings.audit_db_path)
        log.info("Audit DB: SQLite (%s)", settings.audit_db_path)

    return _db


async def close_db() -> None:
    """Close the database connection. Call on shutdown."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None
