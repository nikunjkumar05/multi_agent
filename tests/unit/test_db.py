"""Unit tests for core/db.py — async database abstraction."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.db import PostgresDB, SQLiteDB, get_db

# ── SQLite Backend ─────────────────────────────────────────────────────


class TestSQLiteDB:
    @pytest.fixture
    def db_path(self, tmp_path):
        return str(tmp_path / "test_audit.db")

    async def test_init_db_creates_table(self, db_path):
        db = SQLiteDB(db_path)
        await db.init_db("""
            CREATE TABLE IF NOT EXISTS test_table (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL
            );
        """)
        await db.execute("INSERT INTO test_table (name) VALUES (?)", ("hello",))
        row = await db.fetchone("SELECT name FROM test_table")
        assert row == ("hello",)
        await db.close()

    async def test_execute_and_fetchone(self, db_path):
        db = SQLiteDB(db_path)
        await db.init_db("""
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                val TEXT
            );
        """)
        await db.execute("INSERT INTO items (val) VALUES (?)", ("test1",))
        row = await db.fetchone("SELECT val FROM items WHERE val = ?", ("test1",))
        assert row == ("test1",)
        await db.close()

    async def test_fetchall(self, db_path):
        db = SQLiteDB(db_path)
        await db.init_db("""
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                val TEXT
            );
        """)
        await db.execute("INSERT INTO items (val) VALUES (?)", ("a",))
        await db.execute("INSERT INTO items (val) VALUES (?)", ("b",))
        rows = await db.fetchall("SELECT val FROM items ORDER BY val")
        assert rows == [("a",), ("b",)]
        await db.close()

    async def test_fetchone_returns_none_when_empty(self, db_path):
        db = SQLiteDB(db_path)
        await db.init_db("""
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                val TEXT
            );
        """)
        row = await db.fetchone("SELECT val FROM items")
        assert row is None
        await db.close()

    async def test_executemany(self, db_path):
        db = SQLiteDB(db_path)
        await db.init_db("""
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                val TEXT
            );
        """)
        await db.executemany(
            "INSERT INTO items (val) VALUES (?)",
            [("x",), ("y",), ("z",)],
        )
        rows = await db.fetchall("SELECT val FROM items ORDER BY val")
        assert rows == [("x",), ("y",), ("z",)]
        await db.close()

    async def test_backend_property(self, db_path):
        db = SQLiteDB(db_path)
        assert db.backend == "sqlite"
        await db.close()

    async def test_close_idempotent(self, db_path):
        db = SQLiteDB(db_path)
        await db.init_db("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY);")
        await db.close()
        await db.close()  # second close should not raise


# ── PostgreSQL Backend (mocked) ────────────────────────────────────────


class _MockAcquireCtx:
    """Async context manager that yields a mock connection."""
    def __init__(self, conn):
        self._conn = conn
    async def __aenter__(self):
        return self._conn
    async def __aexit__(self, *args):
        pass


class TestPostgresDB:
    async def test_backend_property(self):
        db = PostgresDB("postgresql://user:pass@localhost/testdb")
        assert db.backend == "postgresql"

    async def test_init_db_calls_execute(self):
        db = PostgresDB("postgresql://user:pass@localhost/testdb")
        mock_conn = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        mock_pool.acquire.return_value = _MockAcquireCtx(mock_conn)
        db._pool = mock_pool

        await db.init_db("CREATE TABLE IF NOT EXISTS t (id SERIAL PRIMARY KEY);")
        mock_conn.execute.assert_called_once_with("CREATE TABLE IF NOT EXISTS t (id SERIAL PRIMARY KEY);")
        await db.close()

    async def test_execute_uses_pool(self):
        db = PostgresDB("postgresql://user:pass@localhost/testdb")
        mock_conn = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        mock_pool.acquire.return_value = _MockAcquireCtx(mock_conn)
        db._pool = mock_pool

        await db.execute("INSERT INTO t (val) VALUES ($1)", ("hello",))
        mock_conn.execute.assert_called_once_with("INSERT INTO t (val) VALUES ($1)", "hello")
        await db.close()

    async def test_fetchone_uses_fetchrow(self):
        db = PostgresDB("postgresql://user:pass@localhost/testdb")
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=("hello",))
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        mock_pool.acquire.return_value = _MockAcquireCtx(mock_conn)
        db._pool = mock_pool

        result = await db.fetchone("SELECT val FROM t")
        assert result == ("hello",)
        await db.close()

    async def test_fetchall_converts_records(self):
        db = PostgresDB("postgresql://user:pass@localhost/testdb")
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[("a",), ("b",)])
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        mock_pool.acquire.return_value = _MockAcquireCtx(mock_conn)
        db._pool = mock_pool

        rows = await db.fetchall("SELECT val FROM t ORDER BY val")
        assert rows == [("a",), ("b",)]
        await db.close()

    async def test_close_closes_pool(self):
        db = PostgresDB("postgresql://user:pass@localhost/testdb")
        mock_pool = AsyncMock()
        db._pool = mock_pool

        await db.close()
        mock_pool.close.assert_called_once()
        assert db._pool is None


# ── Factory ────────────────────────────────────────────────────────────


class TestGetDB:
    async def test_sqlite_when_no_database_url(self):
        with patch("core.db.get_db"):
            # Reset singleton
            import core.db
            core.db._db = None

            with patch("core.config.settings") as mock_settings:
                mock_settings.database_url = None
                mock_settings.audit_db_path = "/tmp/test_bamas.db"
                db = await get_db()
                assert isinstance(db, SQLiteDB)

            await db.close()
            core.db._db = None

    async def test_postgres_when_database_url_set(self):
        import core.db
        core.db._db = None

        with patch("core.config.settings") as mock_settings:
            mock_settings.database_url = "postgresql://user:pass@localhost/testdb"
            db = await get_db()
            assert isinstance(db, PostgresDB)

        await db.close()
        core.db._db = None

    async def test_singleton_returns_same_instance(self):
        import core.db
        core.db._db = None

        with patch("core.config.settings") as mock_settings:
            mock_settings.database_url = None
            mock_settings.audit_db_path = "/tmp/test_bamas2.db"
            db1 = await get_db()
            db2 = await get_db()
            assert db1 is db2

        await db1.close()
        core.db._db = None
