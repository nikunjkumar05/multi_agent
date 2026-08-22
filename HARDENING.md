# BAMAS Middleware Hardening Log

**Date:** Aug 22, 2026
**Scope:** Bug fixes + MVP hardening of the budget-aware middleware
**Result:** 127/127 tests passing, zero errors

---

## 1. Critical Bug Fixes

### 🔴 FIX-1: Orchestrator infinite loop (`agent/orchestrator.py`)
**Severity:** CRITICAL
**Bug:** Lines ~235–273 were wrongly indented at 12 spaces, nesting the graph-rebuild /
degradation logic inside `if budget_obj and budget.max_cost_usd > 0:` instead of the
while-loop level.

**Impact:** When no budget object existed, an interrupt caused `run_task_with_degradation()`
to loop forever — state was never updated and `degradation_count` never incremented.
With a budget it worked by accident.

**Fix:** Dedented the block to loop level; rebuild logic now executes unconditionally
after any interrupt. Also added `"feedback": 0.75` to the topology quality-factor map.

---

### 🟠 FIX-2: Cancel race condition (`middleware/api/routes/tasks.py`)
**Bug:** `cancel_task()` set status to CANCELLED but the background worker later
overwrote it with COMPLETED/FAILED.

**Fix:** Worker checks status before starting, after adapter lookup, and again after
execution returns — a cancelled task's result is discarded, never overwrites state.
Cancelling a finished task now returns **409 Conflict**.

---

### 🟠 FIX-3: Tests spawned real agents (`middleware/tests/`)
**Bug:** Unit tests registered real OpenCode/Aider adapters — pytest runs would spawn
live CLI subprocesses hitting actual LLM APIs.

**Fix:**
- NEW `middleware/adapters/mock.py` — deterministic `MockAdapter`, zero network/processes
- NEW `middleware/tests/conftest.py` — sets `BAMAS_MIDDLEWARE_TEST_MODE=1` before app import;
  registry builds mocks only in test mode

---

### 🟠 FIX-4: CORS wildcard with credentials (`middleware/api/main.py`)
**Bug:** `allow_origins=["*"]` combined with `allow_credentials=True`.

**Fix:** Restricted to localhost dev origins via `BAMAS_CORS_ORIGINS` env var
(defaults: localhost:3000/5173/8080), explicit method + header allowlist.

---

## 2. MVP Hardening — Budget Enforcement (the seatbelt)

The product's core promise ("budget-aware middleware") was disconnected from the request
flow. `BudgetManager` existed but nothing used it. Now wired end-to-end:

### Request flow (`create_task`)
| Step | Behaviour |
|------|-----------|
| Resolve budget | `budget_id` given → fetch persistent budget (404 if missing, 402 if inactive); else auto-create ephemeral single-task budget from `budget_usd` |
| Gate check | `can_afford(estimated_cost)` → **402 DENY** when exhausted, warning flag at ≥80% (WARN band) |
| Record | Task stores `budget_id`, spend snapshot, warning |

### Worker flow (`execute_task_background`)
| Step | Behaviour |
|------|-----------|
| Pre-execution re-check | Budget may have changed since queueing — DENY fails fast with clear error |
| Post-execution | `record_usage(actual_cost, tokens)` deducts from linked budget |
| Overrun guard | `budget_exceeded=true` when actual cost exceeds per-task ceiling |
| Stats hygiene | Mock results excluded from registry reliability/cost stats |

### Shared state (`middleware/api/state.py` — NEW)
Singletons for `registry`, `budget_manager`, `tasks_db` shared across route modules
without circular imports.

---

## 3. New API Endpoints (`middleware/api/routes/budgets.py`)

```
POST   /api/v1/budgets          create persistent budget (201)
GET    /api/v1/budgets          list budgets (?owner= filter)
GET    /api/v1/budgets/{id}     status: spent / remaining / tasks_completed
DELETE /api/v1/budgets/{id}     delete budget
GET    /api/v1/agents           registry contents: health, pricing, live stats
```

---

## 4. Schema Changes (`middleware/models/schemas.py`)

| Change | Why |
|--------|-----|
| `ws_url` **removed** from `TaskResponse` | Endpoint never existed — API was lying |
| + `budget_id`, `budget_spent_usd`, `budget_exceeded`, `warning` | Real cost receipt |
| + `attempts` | Fallback-chain audit trail (future) |
| + `error` field | Failure reason visible in GET response |
| `TaskCreate.budget_id` added | Link tasks to persistent budgets |
| `BudgetCreate` aligned to manager params | `max_cost_usd`, `max_tasks`, `warn_threshold`, `ttl_seconds` |

---

## 5. Stale Test Updates (pre-existing failures from feedback-topology alignment)

| File | Update |
|------|--------|
| `tests/unit/test_degrader.py` | Chain expectation includes `feedback`; fanout→feedback + feedback→supervisor degrade tests |
| `tests/unit/test_orchestrator.py` | `TestNextTopology`: fanout→feedback→supervisor path |
| `tests/unit/test_projections.py` | Dispatch-table edge list: 10 → 15 edges |
| `tests/unit/test_bamas_components.py` | `get_valid_projection_edges` count 10 → 15 |

---

## 6. Test Suite Results

```
$ docker exec bamas-testbox python -m pytest \
    tests/unit/test_orchestrator.py tests/unit/test_degrader.py \
    tests/unit/test_projections.py tests/unit/test_bamas_components.py \
    tests/unit/test_budget_gate.py middleware/tests/

127 passed in 9.57s
```

New middleware coverage (14 tests): health, task lifecycle + receipt, budget CRUD,
owner-filtered listing, **402 denial**, unknown-budget 404, WARN band, spend deduction,
agents listing, cancel conflict 409, queued-cancel, worker race guard,
worker budget block.

---

## 7. Files Changed

**Modified (11)**
```
agent/orchestrator.py                  critical indentation fix
middleware/adapters/__init__.py        export MockAdapter + ADAPTERS map
middleware/adapters/aider.py           comment indent cleanup
middleware/api/main.py                 CORS hardening, budgets router, startup banner
middleware/api/routes/tasks.py         budget enforcement, race guards, receipt fields
middleware/models/schemas.py           schema changes above
middleware/tests/test_api.py           14-test suite rewrite
tests/unit/test_bamas_components.py    edge count fix
tests/unit/test_degrader.py            chain expectations + new degrade tests
tests/unit/test_orchestrator.py        next_topology expectations
tests/unit/test_projections.py         dispatch edges fix
```

**New (4)**
```
middleware/adapters/mock.py            deterministic test agent
middleware/api/routes/budgets.py       budget + agents endpoints
middleware/api/state.py                shared singletons
middleware/tests/conftest.py           test-mode bootstrap
```

---

## 8. Known Deferred Items (post-demo roadmap)

- ~~ILP solver → agent selection wiring~~ ✅ Round 2
- ~~SQLite persistence~~ ✅ Round 2
- ~~API-key auth (`X-API-Key`)~~ ✅ Round 2
- ~~Fallback chain execution~~ ✅ Round 2
- WebSocket streaming (fake removed pending real implementation)

---

# ROUND 2 — Advanced Hardening (Aug 22, later session)

**Result:** 145/145 tests passing (23 middleware + 122 core)

---

## R2-1: Budget-Constrained Agent Selection (`middleware/selection.py` — NEW)

Replaced naive input-price sorting with paper Component 1 adapted to agents:

```
maximize   reliability(agent)              (quality proxy)
subject to estimated_cost(agent) <= remaining_budget
```

- Exact optimum via exhaustive evaluation (n<20 agents) — deterministic, zero deps
- Per-agent `estimate_cost()` uses real prompt/context (not static pricing table)
- Output = preference ORDER feeding the fallback chain
- Preferred agents keep absolute priority (explicit user intent wins)
- Unaffordable agents trail so caller can 402 with the cheapest estimate

## R2-2: Fallback Chain Execution

Worker walks ranked candidates (primary + up to 2 fallbacks):

| Guarantee | Implementation |
|-----------|----------------|
| "It runs" promise | Primary fails → next candidate auto-runs |
| Full audit | `attempts[]` records every try: agent, success, cost, latency, error |
| Budget safety | Per-attempt gate — DENY fails fast mid-chain |
| Crash safety | Any adapter exception caught; chain continues |

## R2-3: SQLite Persistence (`middleware/persistence.py` — NEW)

- stdlib `sqlite3`, JSON-blob rows in `tasks` + `budgets` tables
- Write-through at every mutation point (create/update/cancel/spend)
- Load-on-startup restores budgets & tasks into managers
- Interrupted tasks (queued/in_progress at restart) marked failed with reason
- Path via `BAMAS_DB_PATH`; write failures logged, never crash the API

## R2-4: API-Key Auth (`middleware/api/main.py`)

- `X-API-Key` required on `/api/*` when `BAMAS_API_KEY` env is set
- Unset → local-only trust model (allow-all + startup warning)
- `/health` stays open for liveness probes
- Key read per-request → testable via monkeypatch, no restart needed

## R2-5: API Contract Fix

- `TaskCreate.task_type`: required → **optional hint**. The classifier always
  determines the authoritative type; requiring a field that gets overridden
  was a misleading contract.

## R2-6: Modern Lifespan + Startup Restore Banner

- Replaced deprecated `@app.on_event("startup")` with `lifespan` context
- Logs restored counts + TEST MODE warning at boot

---

## Files Changed (Round 2)

```
NEW  middleware/selection.py           budget-constrained ranking
NEW  middleware/persistence.py         SQLite write-through/load
NEW  middleware/tests/test_hardening.py  9 tests (fallback/selection/persist/auth)
MOD  middleware/budget/budget_manager.py  restore_budget() for startup load
MOD  middleware/api/routes/tasks.py    selection order + fallback chain + persistence
MOD  middleware/api/routes/budgets.py  persistence hooks
MOD  middleware/api/main.py            lifespan, auth middleware, restore-on-boot
MOD  middleware/models/schemas.py      task_type optional hint
MOD  middleware/tests/conftest.py      isolated temp DB per session
MOD  .gitignore                        exclude *.db
```

## Verification

```
$ docker exec bamas-testbox python -m pytest \
    tests/unit/{test_orchestrator,test_degrader,test_projections,
                test_bamas_components,test_budget_gate,test_llm_helpers}.py \
    middleware/tests/

145 passed in 11.24s
```
