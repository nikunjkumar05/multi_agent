# BAMAS Backend Architecture — Migration Plan

## Goal
Migrate BAMAS to PostgreSQL + Celery + Persistent Checkpointer for production readiness.

## Deployment Target
Docker Compose (single server). Kubernetes later.

```
┌─────────────────────────────────────────────────────┐
│                 Docker Compose                       │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ FastAPI  │  │ Celery   │  │ Celery Beat      │  │
│  │ (port    │  │ Worker   │  │ (periodic tasks) │  │
│  │  8000)   │  │ (x2)     │  │                  │  │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────┘  │
│       │              │                  │            │
│       └──────────────┼──────────────────┘            │
│                      │                               │
│              ┌───────▼───────┐                       │
│              │     Redis     │ (broker + events)     │
│              │  (port 6379)  │                       │
│              └───────────────┘                       │
│                                                      │
│              ┌───────────────┐                       │
│              │  PostgreSQL   │ (persistent state)    │
│              │  (port 5432)  │                       │
│              └───────────────┘                       │
└─────────────────────────────────────────────────────┘
```

---

## Phase 1: PostgreSQL Setup ✅ COMPLETE

### 1.1 pyproject.toml — Add Dependencies ✅
- [x] Add `psycopg[binary]>=3.2.0`
- [x] Add `psycopg-pool>=3.2.0`
- [x] Add `langgraph-checkpoint-postgres>=3.1.0`
- [x] Add `celery[redis]>=5.4.0`
- [x] Keep `asyncpg>=0.29.0` (audit trail)
- [x] Keep `aiosqlite>=0.19.0` (fallback)

### 1.2 core/config.py — Add PostgreSQL Config ✅
- [x] Add `database_url` (already existed)
- [x] Add `db_pool_min_size: int = 2`
- [x] Add `db_pool_max_size: int = 10`
- [x] Add `celery_broker_url: str | None = None`

### 1.3 core/db.py — Add psycopg Pool for LangGraph ✅
- [x] Add `get_psycopg_pool()` function
- [x] Add `_psycopg_pool` global singleton
- [x] Add `close_psycopg_pool()` function
- [x] Keep existing `asyncpg` pool for audit trail

### 1.4 core/rl_policy.py — Rewrite 9 Raw SQLite Calls ✅
- [x] Remove all `import aiosqlite` and direct `aiosqlite.connect()` calls
- [x] Add private `_execute()` and `_fetchall()` helper methods
- [x] Auto-convert `?` params to `$1` for PostgreSQL
- [x] Add separate DDL for PostgreSQL vs SQLite
- [x] Replace all 9 `aiosqlite.connect()` calls with helpers
- [x] Keep file fallback (`rl_policy.json`) as optional

### 1.5 docker-compose.yml — Add PostgreSQL Service ✅
- [x] Add `postgres` service (postgres:16-alpine)
- [x] Add `postgres_data` volume
- [x] Add healthcheck: `pg_isready -U bamas`
- [x] Update `api` service to depend on `postgres`

### 1.6 .env.example — Add DATABASE_URL ✅
- [x] Add `DATABASE_URL=postgresql://bamas:bamas_dev@postgres:5432/bamas`

---

## Phase 2: Persistent Checkpointer ✅ COMPLETE

### 2.1 agent/topologies/builder.py — Replace MemorySaver ✅
- [x] Import `BaseCheckpointSaver` from `langgraph.checkpoint.base`
- [x] Replace `checkpointer = MemorySaver()` with `checkpointer: BaseCheckpointSaver | None`
- [x] Add `init_checkpointer()` async function (PostgreSQL + MemorySaver fallback)
- [x] Add `close_checkpointer()` async function
- [x] Update `compile_graph()` to use new checkpointer

### 2.2 api/main.py — Initialize Checkpointer in Lifespan ✅
- [x] Call `init_checkpointer()` in lifespan startup
- [x] Call `close_checkpointer()` in lifespan shutdown
- [x] Call `close_psycopg_pool()` in lifespan shutdown
- [x] Call `close_db()` in lifespan shutdown

### 2.3 agent/graph.py — No Changes Needed ✅
- [x] Already imports `checkpointer` from builder (same variable name)

### 2.4 agent/orchestrator.py — Update Type Annotation ✅
- [x] Change `checkpointer: MemorySaver` to `checkpointer: BaseCheckpointSaver`
- [x] Import `BaseCheckpointSaver` from `langgraph.checkpoint.base`

---

## Phase 3: Celery Integration (Day 3-5)

### 3.1 pyproject.toml — Already Done in Phase 1

### 3.2 celery_app/__init__.py — Celery App Init
- [ ] Create `celery_app/` directory
- [ ] Create `__init__.py` with Celery app initialization
- [ ] Configure broker (Redis), backend (Redis), serialization (JSON)
- [ ] Add task autodiscovery

### 3.3 celery_app/tasks.py — Celery Task Definitions
- [ ] Create `run_task_celery` task
- [ ] Serialize `budget_usd` (not BudgetTracker)
- [ ] Use `asyncio.run()` to bridge async code
- [ ] Add retry logic (max_retries=3)
- [ ] Add task timeout (300s)
- [ ] Add result callback to update task store

### 3.4 celery_app/worker.py — Worker Lifecycle
- [ ] Add `worker_process_init` signal (initialize DB connections)
- [ ] Add `worker_process_shutdown` signal (close DB connections)

### 3.5 api/routes/execute.py — Replace BackgroundTasks
- [ ] Import Celery task
- [ ] Replace `bg.add_task()` with `celery_task.delay()`
- [ ] Remove `_tasks` dict (move to TaskStore in Phase 4)

### 3.6 api/routes/proxy.py — Replace asyncio.create_task
- [ ] Replace `asyncio.create_task(run_task(...))` with Celery dispatch
- [ ] Add client disconnect detection

### 3.7 docker-compose.yml — Add Celery Services
- [ ] Add `celery_worker` service
- [ ] Add `celery_beat` service (for periodic tasks)
- [ ] Add depends on `redis` and `postgres`

---

## Phase 4: Task Store (Day 5-6)

### 4.1 core/task_store.py — TaskStore Abstraction
- [ ] Define `TaskStore` ABC with `create()`, `get()`, `list()`, `update()` methods
- [ ] Implement `PostgresTaskStore` using `core/db.py`
- [ ] Implement `InMemoryTaskStore` for testing
- [ ] Add factory function `get_task_store()`

### 4.2 PostgreSQL Schema for Tasks
```sql
CREATE TABLE IF NOT EXISTS tasks (
    task_id UUID PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    topology TEXT,
    task TEXT NOT NULL,
    budget_usd FLOAT,
    final_result TEXT,
    budget_spent_pct FLOAT,
    degradation_count INT DEFAULT 0,
    logs JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 4.3 api/routes/execute.py — Replace _tasks Dict
- [ ] Import `get_task_store()`
- [ ] Replace `_tasks[task_id] = ...` with `task_store.create(...)`
- [ ] Replace `_evict_if_needed()` with DB-level cleanup

### 4.4 api/routes/tasks.py — Replace _tasks Import
- [ ] Remove `from api.routes.execute import _tasks`
- [ ] Import `get_task_store()`
- [ ] Replace dict reads with `task_store.list()` and `task_store.get()`

### 4.5 api/routes/proxy.py — Replace _tasks Import
- [ ] Remove `from api.routes.execute import _tasks`
- [ ] Import `get_task_store()`
- [ ] Replace dict reads with `task_store.get()`

---

## Phase 5: Graceful Shutdown & Security (Day 6-7)

### 5.1 api/main.py — Graceful Shutdown
- [x] Add `await close_db()` to lifespan shutdown ✅ (Done in Phase 2.2)
- [ ] Add `await close_redis()` to lifespan shutdown
- [x] Add `await close_psycopg_pool()` to lifespan shutdown ✅ (Done in Phase 2.2)

### 5.2 api/main.py — Deep Health Check
- [ ] Check PostgreSQL connectivity
- [ ] Check Redis connectivity
- [ ] Check LLM provider connectivity (optional)
- [ ] Return detailed status: `{"status": "ok", "postgres": "ok", "redis": "ok"}`

### 5.3 api/main.py — Security Hardening
- [ ] Make CORS origins configurable via `ALLOWED_ORIGINS` env var
- [ ] Add request timeout middleware
- [ ] Add rate limiting (slowapi or custom)

---

## Final Tasks

### F.1 Test Suite
- [x] Run `pytest tests/unit/ -v` — all 358 tests must pass ✅
- [ ] Add tests for new TaskStore
- [ ] Add tests for PostgreSQL checkpointer
- [ ] Add integration test with PostgreSQL

### F.2 Commit and Push
- [ ] Commit all Phase 1-5 changes
- [ ] Push to origin/master

### F.3 Documentation
- [ ] Update README.md with new architecture
- [ ] Update .env.example with all new settings
- [ ] Add deployment guide

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| ~~`rl_policy.py` has 9 raw SQLite calls~~ | ~~HIGH~~ | ✅ Rewritten to use shared DB abstraction |
| Celery sync workers + async codebase | MEDIUM | Use `asyncio.run()` bridge |
| Two PostgreSQL drivers (psycopg + asyncpg) | LOW | Keep both — different purposes |
| ~~MemorySaver → AsyncPostgresSaver~~ | ~~MEDIUM~~ | ✅ Initialize in lifespan handler |
| `_tasks` dict imported by 3 modules | MEDIUM | Extract to TaskStore service |

---

## Implementation Order

```
Phase 1: PostgreSQL Setup          (Day 1-2) ✅ COMPLETE
    ↓
Phase 2: Persistent Checkpointer   (Day 2-3) ✅ COMPLETE
    ↓
Phase 3: Celery Integration        (Day 3-5) ← NEXT
    ↓
Phase 4: Task Store                (Day 5-6)
    ↓
Phase 5: Shutdown & Security       (Day 6-7)
```

**Estimated time remaining:** 5-7 days for a senior engineer.
