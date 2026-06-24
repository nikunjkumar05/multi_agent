# File Specifications — Multi-Agent Task Executor

## core/ — Core Engine

### config.py
Loads settings from `.env` (API keys, model names, budget limits, Redis URL).
Single `settings` object used everywhere via `from core.config import settings`.

### llm.py
Factory that creates LLM instances by tier.
Maps `cheap` → `gpt-4o-mini`, `standard` → `gpt-4o`, `frontier` → `gpt-4o`.
Swaps provider (OpenAI/Anthropic/Ollama) in one config change.
Key function: `create_llm(tier: ModelTier, temperature: float) -> BaseChatModel`

### budget.py
Tracks cost/token consumption. Knows which budget band you're in.
Bands: <70% healthy, 70-90% downgrade, 90%+ collapse, critical skip Judge.
Key class: `BudgetTracker` with `get_band()`, `get_allowed_tiers()`, `should_skip_judge()`

### optimizer.py
Cost-tier optimizer (BAMAS-adapted, ILP + RL).
Selects which models fit budget (ILP) + best topology (RL).
Key class: `CostTierOptimizer` with `optimize(task, budget) -> OptimizerDecision`

### escalation.py
Escalation engine. Checks if Validator and Executor diverged on reasoning chains.
Decides: skip Judge or escalate to frontier model.
Key function: `should_escalate(validator_output, executor_output) -> bool`

### degrader.py
Budget governor. When budget crosses 90%, collapses topology.
Chain: `ensemble → fanout → supervisor → pipeline → single`.
Key function: `degrade_topology(budget, current_topology) -> str`

### audit.py
Append-only log. Records every topology decision, budget band crossing,
and degradation event with timestamps.
Key function: `get_audit_trail() -> AuditTrail`

### stats.py (V3)
Tracks which topology × budget × task combo performs best.
Feeds back into optimizer.

### learning.py (V3)
Self-optimization loop. Judge scores train optimizer over time.

---

## agent/ — Agent Orchestration

### graph.py
Entry point. Calls optimizer to get topology, builds LangGraph,
runs it, returns result.
Key function: `run_task(task, task_id, budget) -> dict`

### state.py
`AgentState` TypedDict — shared state flowing through the graph:
task, plan, results, topology, budget, errors, logs, status.

---

## agent/nodes/ — Individual Agents

### planner.py
Breaks task into numbered steps. Standard model.
Key function: `plan_task(state) -> dict`

### executor.py
Runs one step using tools (code exec, web search, file ops, DB). Standard model.
Key function: `execute_step(state) -> dict`

### validator.py
Checks step result correctness. Returns confidence + reasoning divergence flag. Cheap model.
Key function: `validate_result(state) -> dict`

### judge.py
Final arbitration when agents disagree. Merges/selects best output. Frontier model, conditional.
Key function: `ensemble_judge(state) -> dict`

### escalation.py
Decision logic between Validator and Judge. Reasoning divergence, not vote count.
Key function: `check_escalation(state) -> str`

---

## agent/topologies/ — Graph Builders

### single.py
One agent, one pass. Trivial tasks.
Builder: `build_single_graph() -> StateGraph`

### supervisor.py
Supervisor routes to workers, workers return. Branching research.
Builder: `build_supervisor_graph() -> StateGraph`

### pipeline.py
Sequential stages. Content generation, ETL.
Builder: `build_pipeline_graph() -> StateGraph`

### fanout.py
Parallel workers + aggregator. Data analysis, bulk processing.
Builder: `build_fanout_graph() -> StateGraph`

### ensemble.py
Multiple agents + Judge. High-stakes cross-validation.
Builder: `build_ensemble_graph() -> StateGraph`

### builder.py
Picks right builder by name, compiles with checkpointer.
Key function: `compile_graph(topology) -> CompiledStateGraph`

---

## agent/tools/ — Tool Registry

### base.py
`BaseTool` ABC + `ToolResult` dataclass. All tools implement this.

### registry.py
Singleton registry. `register(tool)`, `execute(name, params)`, `list_tools()`.

### code_executor.py
Runs Python in sandboxed subprocess. Timeout: 30s.

### web_search.py
DuckDuckGo HTML scraping via httpx. Max 5 results.

### file_ops.py
Read/write/list files in `./workspace`. Path-sandboxed.

### db_query.py
Read-only SQL against SQLite. Blocks non-SELECT queries.

---

## api/ — FastAPI Service

### main.py
FastAPI app. Mounts routes, middleware (CORS), lifespan.

### routes/execute.py
`POST /execute` — accepts task + budget, runs graph, returns task_id.

### routes/tasks.py
`GET /tasks/{id}` — poll status/results.

### routes/audit.py
`GET /audit/{id}` — full decision audit trail.

### models/schemas.py
Pydantic request/response: `ExecuteRequest`, `TaskStatusResponse`, `AuditResponse`.

### websocket.py
Real-time event stream during execution.

---

## Infra Files

### Dockerfile
Multi-stage build: builder installs deps, runtime copies venv. Non-root user.

### docker-compose.yml
`api` (port 8000) + `redis` (sidecar, healthcheck).

### .github/workflows/deploy.yml
CI/CD: lint (ruff) → typecheck (mypy) → test (pytest) → build Docker → push to GHCR.

### requirements.txt
Production deps.

### requirements-dev.txt
Dev deps (pytest, ruff, mypy, respx).

### .env.example
Template for secrets.

### pyproject.toml
Project metadata, ruff/mypy/pytest config.
