# BAMAS — Technical Deep Dive

## 1. System Overview

BAMAS (Budget-Aware Multi-Agent System) is a **cost-controlled agent orchestration engine**. It accepts a plain-English task + budget, selects the optimal topology and model tiers, executes the task through a multi-agent LangGraph, and returns the result — all within the specified budget.

### Core Design Philosophy

Three layered decisions control cost-quality tradeoffs, from cheapest to most aggressive:

| Layer | Mechanism | Trigger |
|-------|-----------|---------|
| **Topology Selection** | Pick simplest topology that can handle the task | Per-request (optimizer) |
| **Model Tier Downgrade** | Swap frontier→standard→cheap per agent role | Budget 70-90% spent |
| **Topology Degradation** | Collapse ensemble→fanout→supervisor→pipeline→single | Budget >90% spent |

---

## 2. End-to-End Request Flow

```
Client                 FastAPI                Optimizer               LangGraph               LLM
  │                       │                       │                       │                   │
  │  POST /execute        │                       │                       │                   │
  │──────────────────────>│                       │                       │                   │
  │  {task_id, pending}   │                       │                       │                   │
  │<──────────────────────│                       │                       │                   │
  │                       │  Background task      │                       │                   │
  │                       │──────────────────────>│                       │                   │
  │                       │                       │  LLM semantic classify│                   │
  │                       │                       │────────────────────────────────────────────>│
  │                       │                       │<────────────────────────────────────────────│
  │                       │                       │  {topology, tiers}    │                   │
  │                       │                       │                       │                   │
  │                       │                       │  (or rule-based       │                   │
  │                       │                       │   keyword fallback)   │                   │
  │                       │                       │                       │                   │
  │                       │                       │  Query RL policy      │                   │
  │                       │                       │─────> Redis ─────────>│                   │
  │                       │                       │  (f ≥10 → override)   │                   │
  │                       │                       │                       │                   │
  │                       │                       │  Check budget band    │                   │
  │                       │                       │─────> BudgetTracker   │                   │
  │                       │                       │  degrade if >90%      │                   │
  │                       │                       │                       │                   │
  │                       │  compile_graph(topo)  │                       │                   │
  │                       │──────────────────────>│                       │                   │
  │                       │                       │  Planner              │                   │
  │                       │                       │────────────────────────────────────────────>│
  │                       │                       │<────────────────────────────────────────────│
  │                       │                       │  steps[1..n]          │                   │
  │                       │                       │                       │                   │
  │  WebSocket /ws/{id}   │   loop steps:         │                       │                   │
  │<══════════════════════│   Executor (ReAct)    │                       │                   │
  │  event: step_started  │────────────────────────────────────────────────────────────>│
  │  event: step_completed│<────────────────────────────────────────────────────────────│
  │  event: tool_call     │   Validator           │                       │                   │
  │  event: validation    │────────────────────────────────────────────────────────────>│
  │                       │<────────────────────────────────────────────────────────────│
  │                       │   {confidence, diverged} │                   │                   │
  │                       │                       │                       │                   │
  │                       │   if escalate:        │                       │                   │
  │                       │   Judge               │                       │                   │
  │                       │────────────────────────────────────────────────────────────>│
  │                       │<────────────────────────────────────────────────────────────│
  │                       │                       │                       │                   │
  │                       │  Finalizer            │                       │                   │
  │                       │──────────────────────>│                       │                   │
  │                       │  result + audit       │                       │                   │
  │  GET /tasks/{id}      │                       │                       │                   │
  │──────────────────────>│                       │                       │                   │
  │<──────────────────────│                       │                       │                   │
  │  {status, result}     │                       │                       │                   │
```

### 2.1 Step-by-Step (15,000 ft → code)

---

## 3. API Layer

### 3.1 Entry Point: `api/main.py`

```python
app = FastAPI(title="Multi-Agent Task Executor", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
app.include_router(execute.router)   # POST /execute
app.include_router(tasks.router)     # GET /tasks/{task_id}
app.include_router(audit.router)     # GET /audit/{task_id}
app.include_router(websocket.router) # WebSocket /ws/{task_id}
app.mount("/static", ...)            # static frontend files

@app.get("/")        → serves index.html
@app.get("/health")  → {"status": "ok"}
```

### 3.2 POST /execute: `api/routes/execute.py`

**Request** (`ExecuteRequest`):
```json
{"task": "...", "budget_usd": 1.0, "topology": "pipeline" | null}
```

**Flow** (lines 40-54):
1. Generate `task_id` (UUID4)
2. Create `BudgetTracker(max_cost_usd=req.budget_usd)` — starts with 0 cost
3. Store `TaskStatusResponse(status="pending")` in `_tasks` dict (in-memory)
4. Launch background task via `BackgroundTasks.add_task(_run_background, ...)`
5. Return `{task_id, status: "pending"}` immediately

**Background execution** (`_run_background`, lines 17-37):
1. Update `_tasks[task_id]` to `status="running"`
2. Call `run_task(task, budget, task_id, topology_override)`
3. On success → store result in `_tasks`
4. On exception → store failure status

### 3.3 GET /tasks/{task_id}: `api/routes/tasks.py`

Reads from `_tasks` dict (line 9-13). Returns 404 if not found.

### 3.4 GET /audit/{task_id}: `api/routes/audit.py`

Calls `get_audit_trail().get_task_audit(task_id)` — reads from in-memory `AuditTrail` singleton. Returns 404 if no entries.

### 3.5 WebSocket /ws/{task_id}: `api/websocket.py`

**Flow** (lines 8-29):
1. Accept WebSocket connection
2. Get Redis connection
3. Create `EventBroadcaster(redis)`
4. Send historical events via `get_history(task_id)`
5. Subscribe to live events via `subscribe(task_id)` — async generator that listens on Redis pub/sub channel `events:{task_id}`
6. Each event is JSON-serialized and sent to the WebSocket client

### 3.6 Request Schemas: `api/models/schemas.py`

```python
class ExecuteRequest(BaseModel):
    task: str                                    # required
    budget_usd: float = 1.0                      # 0.01-100.0
    topology: str | None = None                  # override: single/pipeline/...

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str                                  # pending/running/completed/failed
    final_result: str | None
    judge_output: str | None
    budget_spent_pct: float
    topology: str
    logs: list[str]

class AuditResponse(BaseModel):
    task_id: str
    events: list[dict]                           # timestamped audit entries
```

---

## 4. Core Engine

### 4.1 Configuration: `core/config.py`

`Settings` (Pydantic BaseSettings) reads from `.env`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `openai` | `openai` / `mistral` / `ollama` |
| `OPENAI_API_KEY` | None | OpenAI API key |
| `MISTRAL_API_KEY` | None | Mistral API key |
| `MISTRAL_MODEL` | `mistral-large-latest` | |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | |
| `tier_cheap_model` | `mistral-tiny` | Model for cheap tier |
| `tier_standard_model` | `mistral-small-latest` | Model for standard tier |
| `tier_frontier_model` | `mistral-large-latest` | Model for frontier tier |
| `tier_cost_per_1k_tokens` | `{cheap:0.0002, standard:0.001, frontier:0.008}` | Cost tracking |
| `BUDGET_MAX_COST_USD` | 1.00 | Default max budget |
| `BUDGET_MAX_TOKENS` | 100000 | Token limit |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `API_HOST/PORT` | 0.0.0.0:8000 | Server binding |

### 4.2 LLM Factory: `core/llm.py`

**Provider pattern** (lines 14-63):
```python
TIER_MODEL_MAP: dict[Provider, dict[ModelTier, str]] = {
    "openai": {"cheap": ..., "standard": ..., "frontier": ...},
    "mistral": {"cheap": ..., "standard": ..., "frontier": ...},
    "ollama": {"cheap": "llama3.2:3b", "standard": "llama3.2:7b", "frontier": "qwen2.5:14b"},
}

def create_llm(tier: ModelTier, temperature=0.0) -> BaseChatModel:
    provider = _PROVIDERS[settings.llm_provider]
    model_name = TIER_MODEL_MAP[settings.llm_provider][tier]
    return provider.get_chat_model(model=model_name, temperature=temperature)
```

Three provider classes (`OpenAIProvider`, `MistralProvider`, `OllamaProvider`) each implement `get_chat_model()` returning the appropriate LangChain chat model.

**Cost estimation** (lines 66-84):
```python
def estimate_tokens(response) -> int:    # reads usage_metadata or falls back to len(content)/4
def estimate_cost(response, tier) -> float:  # (tokens / 1000) * tier_cost_per_1k
```

### 4.3 Budget Tracker: `core/budget.py`

`BudgetTracker` is a dataclass with:

```python
@dataclass
class BudgetTracker:
    max_cost_usd: float          # from settings or request
    max_tokens: int
    consumed_cost: float = 0.0
    consumed_tokens: int = 0

    @property
    def spent_pct(self) -> float:                  # consumed_cost / max_cost_usd * 100
    @property
    def remaining_pct(self) -> float:              # 100 - spent_pct
    def get_band(self) -> BudgetBand:              # HEALTHY / TIER_DOWNGRADE / STRUCTURAL_DEGRADE / CRITICAL
    def can_afford_tier(self, tier: ModelTier) -> bool
    def get_allowed_tiers(self) -> list[ModelTier]  # based on current band
    def record_usage(self, tokens, cost) -> None   # increment consumed counters
    def should_skip_judge(self) -> bool             # True if STRUCTURAL or CRITICAL
```

**Budget Band thresholds** (lines 30-39):
| Spent % | Band | Behavior |
|---------|------|----------|
| < 70% | `HEALTHY` | Full flexibility, all tiers allowed |
| 70-90% | `TIER_DOWNGRADE` | Only cheap + standard tiers |
| 90-100% | `STRUCTURAL_DEGRADE` | Only cheap tier, topology will degrade |
| > 100% | `CRITICAL` | Only cheap tier, Judge skipped |

**Tier allowance per band** (lines 51-59):
```python
HEALTHY:        ["cheap", "standard", "frontier"]
TIER_DOWNGRADE: ["cheap", "standard"]
STRUCTURAL:     ["cheap"]
CRITICAL:       ["cheap"]
```

**Usage tracking** (line 61-63): called from executor, planner, validator, judge nodes after each LLM call:
```python
def record_usage(self, tokens, cost):
    self.consumed_tokens += tokens
    self.consumed_cost += cost
```

### 4.4 Cost-Tier Optimizer: `core/optimizer.py`

The optimizer has **three layers** in a fallback chain:

#### Layer 1: LLM Semantic Classification (primary)

Uses `self._structured_llm` — a LangChain chat model with `with_structured_output(OptimizerDecision)` — to parse the LLM response into a typed Pydantic model.

```python
OPTIMIZER_PROMPT = """You are a cost-tier optimizer for a multi-agent system...
Topology selection rules:
- single: trivial Q&A, one-liner answers, simple math
- pipeline: code generation, content creation, writing
- supervisor: research tasks, explanations, comparisons
- fanout: data analysis, parallel subtasks
- ensemble: high-stakes decisions, critical validation

Budget constraints (spent %):
- <70%: full flexibility
- 70-90%: downgrade frontier→standard, standard→cheap
- >90%: only cheap model, simplest topology

Task: {task}
Budget spent: {spent_pct:.1f}%
"""

async def optimize(self, task, budget, task_id):
    # 1. Try LLM
    try:
        prompt = OPTIMIZER_PROMPT.format(task=task, spent_pct=budget.spent_pct)
        llm_decision = await asyncio.wait_for(
            self._structured_llm.ainvoke(prompt), timeout=10.0
        )
        if llm_decision.topology in VALID_TOPOLOGIES:
            # success — use LLM result
    except Exception:
        llm_decision = None

    # 2. Fallback: rule-based
    if llm_decision is None:
        rule_topo = rule_based_select_topology(task)
        llm_decision = self._make_fallback_decision(task, rule_topo)

    # 3. RL refinement
    rl_topology = await rl.select_topology(task, ...)
    if rl_topology and (rule_topo == "single" or rl.total_tasks >= 10):
        chosen = rl_topology   # RL overrides
    else:
        chosen = llm_decision.topology

    # 4. Return with LLM's model tiers
    return OptimizerDecision(
        topology=chosen,
        model_tiers=llm_decision.model_tiers,  # from LLM, not hardcoded
        rationale=...,
        alternatives_considered=llm_decision.alternatives_considered,
    )
```

**Timeout behavior**: If the LLM call exceeds 10 seconds, `asyncio.TimeoutError` is caught and falls through to rule-based.

#### Layer 2: Rule-Based Keyword Fallback

`rule_based_select_topology(task)` (lines 60-79) — keyword matching with priority order:

```python
ensemble_kw  = ["verify", "audit", "validate", "critical", "security", "proof"]
fanout_kw    = ["analyze", "data", "parallel", "bulk", "multiple datasets", "compare all"]
supervisor_kw = ["explain", "research", "compare", "why", "how does", "describe", "summarize", "review"]
pipeline_kw  = ["function", "implement", "code", "class", "script"]
# else → "single"
```

Priority: ensemble > fanout > supervisor > pipeline > single. First match wins.

#### Layer 3: RL Policy Refinement

Called after LLM/rule-based selection. The RL policy (`RLPolicy`) uses contextual Thompson Sampling. See section 7 for details.

### 4.5 OptimizerDecision Schema (lines 12-25)

```python
class OptimizerDecision(BaseModel):
    topology: str                               # one of VALID_TOPOLOGIES
    model_tiers: dict[str, str]                 # e.g., {"planner": "standard", ...}
    rationale: str                              # why this choice
    alternatives_considered: list[dict]          # other topologies and rejection reasons
```

### 4.6 Topology Degrader: `core/degrader.py`

Degradation runs **pre-execution** (before graph.ainvoke). Triggered by budget band.

```python
TOPOLOGY_DEGRADATION_CHAIN = ["ensemble", "fanout", "supervisor", "pipeline", "single"]

def degrade_topology(budget, current_topology, task_id):
    band = budget.get_band()

    if band == HEALTHY:
        return current_topology                  # no change

    if band == TIER_DOWNGRADE:
        audit.record("tier_downgrade")           # model swap only
        return current_topology                  # topology unchanged

    if band == STRUCTURAL_DEGRADE:
        idx = chain.index(current_topology)
        degraded = chain[min(idx+1, len-1)]      # shift one step down
        audit.record("structural_degrade", f"{current} → {degraded}")
        return degraded

    if band == CRITICAL:
        audit.record("critical")
        return "single"                          # simplest topology, skip Judge
```

**Effect by current topology at STRUCTURAL_DEGRADE band:**
| From | To |
|------|-----|
| ensemble | fanout |
| fanout | supervisor |
| supervisor | pipeline |
| pipeline | single |
| single | single |

### 4.7 Escalation Engine: `core/escalation.py`

Simple threshold-based check (not topology-dependent):

```python
ESCALATION_THRESHOLD_CONFIDENCE = 0.85

def should_escalate(validator_confidence, reasoning_diverged, budget):
    if budget.should_skip_judge():
        return False               # budget < 90% → cheap mode, no judge
    if validator_confidence >= 0.85:
        return False               # high confidence → no escalation
    if not reasoning_diverged:
        return False               # no divergence → accept output
    return True                    # escalate to Judge
```

**Condition**: escalate only when ALL three hold: confidence < 0.85 AND reasoning diverged AND budget allows.

### 4.8 Audit Trail: `core/audit.py`

Singleton `AuditTrail` with an in-memory list of entries:

```python
class AuditTrail:
    def __init__(self):
        self._entries: list[dict] = []

    def record(self, task_id, event_type, detail):
        self._entries.append({
            "task_id": task_id,
            "event_type": event_type,    # topology_decision / budget_band_crossed / structural_degradation / task_completed
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "detail": detail,
        })

    def get_task_audit(self, task_id) -> list[dict]
    def to_json(self, task_id) -> str
```

**Specialized recorders**:
- `record_topology_decision(task_id, topology, model_tiers, budget, rationale, alternatives)`
- `record_budget_band(task_id, band, remaining_budget, action)`
- `record_degradation(task_id, from_topology, to_topology, reason)`

### 4.9 Event System: `core/events.py` + `core/node_events.py`

`EventBroadcaster` wraps Redis pub/sub:

```python
class EventBroadcaster:
    async def publish(self, task_id, event_type, data):
        event = {"event_type": event_type, "timestamp": ..., "data": data}
        key = f"events:{task_id}:log"
        await self.redis.lpush(key, event_json)     # push to history list
        await self.redis.ltrim(key, 0, 99)          # keep last 100
        await self.redis.expire(key, 3600)           # TTL 1 hour
        await self.redis.publish(f"events:{task_id}", event_json)  # pub/sub

    async def get_history(self, task_id) → list[dict]
    async def subscribe(self, task_id) → AsyncGenerator[dict, None]  # listen on channel
```

`emit_event()` in `node_events.py` is the convenience wrapper called from agent nodes:

```python
async def emit_event(task_id, event_type, data):
    redis = await get_redis()
    broadcaster = EventBroadcaster(redis)
    await broadcaster.publish(task_id, event_type, data)
```

**Event types emitted**:
| Event | Emitter | Payload |
|-------|---------|---------|
| `planner_started` | planner.py:83 | `{task}` |
| `planner_completed` | planner.py:136 | `{step_count}` |
| `step_started` | executor.py:85 | `{step_id, description}` |
| `step_completed` | executor.py:139 | `{step_id, result_preview}` |
| `tool_call` | executor.py:196 | `{tool, args}` |
| `tool_result` | executor.py:205 | `{tool, success, output_preview}` |
| `validation_completed` | validator.py:101 | `{confidence, diverged}` |
| `judge_completed` | judge.py:67 | `{result_preview}` |
| `escalation_check` | escalation.py:16 | `{confidence, diverged, escalated}` |
| `topology_selected` | graph.py:26 | `{topology, rationale}` |
| `task_completed` | graph.py:71 | `{status, final_result, budget_spent_pct, topology}` |

### 4.10 Redis Client: `core/redis_client.py`

```python
_redis_client: Redis | None = None

async def get_redis() -> Redis | None:
    if _redis_client is None:
        _redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis_client

async def close_redis() -> None:
    if _redis_client:
        await _redis_client.close()
```

Graceful degradation: if Redis is unavailable, returns None and events are silently skipped.

---

## 5. Agent Layer (LangGraph)

### 5.1 AgentState: `agent/state.py`

```python
class AgentState(TypedDict, total=False):
    task: str                                        # original task text
    task_id: str                                     # UUID
    decision: OptimizerDecision                      # from optimizer
    steps: list[PlanStep]                            # planned steps
    current_step_index: int                          # which step executing
    step_results: Annotated[dict, merge_step_results] # merged via reducer
    final_result: str | None
    judge_output: str | None
    budget: BudgetTracker
    escalation_triggered: bool
    validator_confidence: float | None
    reasoning_diverged: bool
    errors: list[str]                                # merged via reducer
    logs: Annotated[list[str], merge_logs]            # merged via reducer
    status: str                                      # pending/planning/executing/validating/...
    retry_count: int
```

**Reducers** (LangGraph pattern for state merging across nodes):
- `merge_step_results`: dict.update (accumulates across fanout/ensemble)
- `merge_logs`: list concatenation
- `merge_errors`: list concatenation

### 5.2 Graph Entry Point: `agent/graph.py`

`run_task()` is the main entry point called from API (lines 15-95):

```python
async def run_task(task, budget, task_id=None, topology_override=None):
    task_id = task_id or str(uuid.uuid4())

    if topology_override:
        degraded_topology = topology_override          # user override, no optimization
        decision = CostTierOptimizer._make_fallback_decision(task, topology_override)
    else:
        optimizer = CostTierOptimizer()
        decision = await optimizer.optimize(task=task, budget=budget, task_id=task_id)
        degraded_topology = degrade_topology(budget, decision.topology, task_id)  # pre-exec degrade

    graph = compile_graph(degraded_topology)           # build LangGraph

    initial_state = AgentState(
        task=task, task_id=task_id, decision=decision, budget=budget,
        steps=[], step_results={}, final_result=None, judge_output=None,
        errors=[], retry_count=0, logs=[f"Topology: {degraded_topology}", ...],
        status="pending",
    )

    result = await graph.ainvoke(initial_state, config={"configurable": {"thread_id": task_id}})

    # Record audit
    audit.record(task_id, "task_completed", {
        "topology": degraded_topology, "status": result.get("status"),
        "budget_spent_pct": budget.spent_pct, ...
    })

    # Emit completion event
    await emit_event(task_id, "task_completed", {...})

    # Train RL if not an override
    if not topology_override:
        rl = RLPolicy(redis)
        quality = 1.0 if result.get("status") == "completed" else 0.0
        cost_eff = max(0.0, 1.0 - (budget.spent_pct / 100.0))
        await rl.reward(topology=decision.topology, quality=quality, cost_efficiency=cost_eff)

    return {"task_id": ..., "status": ..., "final_result": ..., "budget_spent_pct": ..., ...}
```

**Key detail**: `degrade_topology()` is called **before** `compile_graph()`, not mid-execution. The topology is fixed for the entire graph run.

### 5.3 Graph Builder: `agent/topologies/builder.py`

```python
_TOPOLOGY_BUILDERS = {
    "single": build_single_graph,
    "supervisor": build_supervisor_graph,
    "pipeline": build_pipeline_graph,
    "fanout": build_fanout_graph,
    "ensemble": build_ensemble_graph,
}
checkpointer = MemorySaver()

def compile_graph(topology: str) -> StateGraph:
    builder_fn = _TOPOLOGY_BUILDERS.get(topology, build_single_graph)
    graph = builder_fn()
    return graph.compile(checkpointer=checkpointer)
```

Uses `MemorySaver` checkpointer (LangGraph's in-memory checkpoint for state persistence across graph steps).

---

## 6. Agent Nodes

### 6.1 Planner: `agent/nodes/planner.py`

**Input**: `task` string
**Output**: `steps: list[PlanStep]` + `current_step_index: 0` + `status: "planning"`

**Flow** (lines 82-144):
1. `analyze_task_complexity(task)` → returns 1, 2, or 3
2. Detects complexity via keyword heuristics:
   - 1: trivial (hi/hello), simple math (<15 words), or <2 action verbs
   - 3: multi-phase phrases ("and then", "first", "finally") or ≥3 action verbs or >40 words
   - 2: ≥1 action verb or >8 words
3. Calls LLM with `PLANNER_SYSTEM` prompt, requesting exactly `step_count` steps
4. Parses JSON response (with markdown code fence stripping fallback)
5. Falls back to numbered line parsing if JSON decode fails
6. Pads missing steps with generic "Continue and complete" entries
7. Records token/cost usage on budget tracker
8. Emits `planner_started` and `planner_completed` events

```python
PLANNER_SYSTEM = """You are a task planner. Break the given task into clear, numbered steps.
Return ONLY a JSON array of steps. Each step has:
- step_id (int, starting at 1)
- description (string, clear action to perform)
Produce EXACTLY the number of steps specified in the step_count parameter.
"""
```

### 6.2 Executor: `agent/nodes/executor.py`

**Input**: current step from `AgentState.steps[current_step_index]`
**Output**: step result in `step_results[step_id]`

**Flow** (lines 75-166):
1. Detect task type via keyword matching (lines 54-66):
   - `creative` → story/poem/narrative keywords
   - `data` → data/dataset/statistics keywords
   - `code` → write/code/function/keywords
   - `math` → calculate/solve/equation keywords
   - `research` → explain/research/compare keywords
   - `general` → fallback
2. Select task-specific system prompt (lines 12-45) — e.g., code executor gets "You are an expert software engineer..."
3. Build context from previous step results
4. Execute **ReAct loop** (lines 169-217) — up to 5 tool-calling iterations:
   ```
   llm → tool_call → execute_tool → ToolMessage → llm → tool_call → ...
   ```
5. Tools are bound via `llm.bind_tools(langchain_tools)` so the LLM can call them natively
6. After each LLM call, records token/cost on budget tracker
7. On failure, increments `retry_count` and stores error
8. Emits `step_started`, `step_completed`, `tool_call`, `tool_result` events

**ReAct loop detail** (lines 169-217):
```python
async def _react_loop(llm, messages, tier, state):
    for _iteration in range(MAX_TOOL_ITERATIONS):  # max 5
        response = await llm.ainvoke(messages + tool_messages)
        budget.record_usage(tokens=estimate_tokens(response), cost=estimate_cost(response, tier))

        if not response.tool_calls:
            return response.content  # done, no more tools needed

        for tool_call in response.tool_calls:
            result = await asyncio.to_thread(registry.execute, tool_name, **tool_args)
            tool_messages.append(ToolMessage(content=result_text, tool_call_id=tool_id))

    return last_content + "\n\n[Tool loop limit reached]"
```

### 6.3 Validator: `agent/nodes/validator.py`

**Input**: last completed step result from `step_results`
**Output**: `validator_confidence: float` + `reasoning_diverged: bool`

**Flow** (lines 54-113):
1. Select task-type-specific validation prompt (lines 10-51) — code validator checks correctness/edge cases/security, math validator checks calculations, etc.
2. Call LLM with validator prompt (uses `cheap` tier by default)
3. Parse JSON response: `{"confidence": 0.0-1.0, "reasoning_diverged": true/false, "issues": [...], "assessment": "..."}`
4. On parse failure, defaults to `confidence=0.5, diverged=False`
5. Records token/cost
6. Emits `validation_completed` event

### 6.4 Escalation Node: `agent/nodes/escalation.py`

```python
async def check_escalation(state):
    escalate = should_escalate(
        validator_confidence=state.get("validator_confidence", 1.0),
        reasoning_diverged=state.get("reasoning_diverged", False),
        budget=state["budget"],
    )
    return "judge" if escalate else "continue"
```

This is a **routing node** — it returns a string that LangGraph uses as the edge destination name.

### 6.5 Judge: `agent/nodes/judge.py`

**Input**: executor output(s) + validator assessment
**Output**: `judge_output: str` + `final_result: str`

**Flow** (lines 18-76):
1. Collects step results — for fanout/ensemble, collects all parallel agent outputs (`agent_a`, `agent_b`, `agent_c`); for single/pipeline, uses the last step's result
2. Builds prompt with task, executor outputs, validator confidence, and divergence flag
3. Calls LLM with `JUDGE_SYSTEM` prompt at `frontier` tier (highest quality)
4. The judge can: accept the executor's output, improve it based on validator's concerns, or rewrite from scratch
5. Records token/cost
6. Emits `judge_completed` event

```python
JUDGE_SYSTEM = """You are a judge agent. You receive a task, executor output, and validator assessment.
Your job is to produce the BEST possible final result.
You can: 1. Accept output if good enough, 2. Improve based on validator concerns, 3. Rewrite from scratch.
Return the BEST final result as plain text. Be concise and accurate."""
```

### 6.6 Finalizer: `agent/nodes/finalizer.py`

**Input**: `step_results` dict
**Output**: `final_result: str` + `status: str`

Pure logic (no LLM call):
1. If single result → use it directly
2. If multiple results → pick longest, filter out results that are substrings of the best, concatenate unique non-overlapping results
3. Status = "failed" if errors exist, else "completed"

---

## 7. Topology Variants

### 7.1 Single: `agent/topologies/single.py`

```
START → planner → executor → validator ──→ (conditional) ──→ finalizer → END
                                        │                     ↑
                                        ├─ retry (≤2x) → executor
                                        ├─ escalate → judge → finalizer
                                        └─ continue → finalizer
```

**Conditional router** (`_route_after_validation`, lines 14-32):
```python
if errors:
    return "executor" if retry_count < 2 else "finalizer"
if steps remaining:
    return "executor"
if escalate:
    return "judge"
return "finalizer"
```

### 7.2 Pipeline: `agent/topologies/pipeline.py`

```
START → planner → executor → validator ──→ (conditional) ──→ judge → finalizer → END
                                        │
                                        ├─ retry (≤2x) → executor
                                        └─ execute → executor
                                           end → judge
```

Always routes through Judge after all steps complete (no escalation check).

### 7.3 Supervisor: `agent/topologies/supervisor.py`

```
START → planner → supervisor ──→ executor → validator → supervisor (loop)
                        │                                      │
                        └── (if all done) → judge → finalizer → END
```

**Supervisor node** (lines 21-65): LLM-based dispatcher that assigns the next step to execute. Takes the step list, determines which step to run next, and returns the index. When all steps are done, routes to Judge.

### 7.4 Fanout: `agent/topologies/fanout.py`

```
START → planner → dispatcher → parallel_workers → aggregator → judge → END
```

**Dispatcher** (lines 18-37): Divides steps among 3 workers (chunk_size = len(steps) / 3).
**Parallel workers** (lines 40-103): 3 async workers run concurrently via `asyncio.gather`. Each worker executes its assigned steps with the executor prompt.
**Aggregator** (lines 106-115): Concatenates all step results in step_id order.

### 7.5 Ensemble: `agent/topologies/ensemble.py`

```
START → planner → agent_a (analytic) ──→ judge → END
                → agent_b (creative) ───→ judge
                → agent_c (domain expert) → judge
```

**3 agents** run in parallel (LangGraph fan-out), each with a different system prompt:
- Agent A: "analytical rigor and precision" (standard tier)
- Agent B: "creative problem-solving" (standard tier)
- Agent C: "domain expertise" (frontier tier — always)

Results are stored as `agent_a`, `agent_b`, `agent_c` in `step_results`.

---

## 8. Tool System

### 8.1 Architecture

```python
ToolRegistry (singleton)         # agent/tools/registry.py
  └── register(BaseTool)         # register tool by name
  └── execute(name, **kwargs)    # run tool, return ToolResult
  └── get_langchain_tools()      # return list of StructuredTool for LLM binding

BaseTool (ABC)                   # agent/tools/base.py
  ├── name                       # tool identifier string
  ├── description                # prompt description for LLM
  ├── parameters                 # JSON Schema for arguments
  └── execute(**kwargs)          # implementation (subclass responsibility)
      └── returns ToolResult(success, output, error, metadata)
```

### 8.2 CodeExecutor: `agent/tools/code_executor.py`
- Writes code to temp `.py` file
- Runs via `subprocess.run(["python", tmp_path], timeout=30)`
- Returns stdout on success, stderr on failure
- Auto-cleans temp file

### 8.3 WebSearchTool: `agent/tools/web_search.py`
- POSTs to `https://html.duckduckgo.com/html/` with query
- Parses HTML with regex for title/snippet/URL
- Returns top 5 results

### 8.4 File Operations: `agent/tools/file_ops.py`
- Sandboxed to `./workspace/` directory (path traversal protection via `_safe_path`)
- `FileReadTool`, `FileWriteTool`, `FileListTool`
- Write auto-creates parent directories

### 8.5 DBQueryTool: `agent/tools/db_query.py`
- SQLite read-only queries (blocks DELETE, DROP, INSERT, UPDATE, ALTER, CREATE, TRUNCATE)
- Returns results as list of dicts with row_count metadata

---

## 9. RL Policy: `core/rl_policy.py`

### 9.1 Algorithm: Contextual Thompson Sampling

Thompson Sampling is a Bayesian bandit algorithm. Each arm (topology) maintains a Beta distribution `Beta(alpha, beta)`. At selection time, a random sample is drawn from each arm's distribution, and the arm with the highest sample is chosen.

The Beta distribution starts as `Beta(1, 1)` (uniform). After each task:
- **Success** (reward > 0.5) → increment `alpha` (shifts distribution right ≈ higher probability of selection)
- **Failure** (reward ≤ 0.5) → increment `beta` (shifts distribution left ≈ lower probability)

### 9.2 Context Features (lines 59-66)

Four boolean features extracted from task text via keyword matching:

```python
def _extract_features(self, task):
    return {
        "is_code":     any(kw in task_lower for kw in CODE_KEYWORDS),
        "is_research": any(kw in task_lower for kw in RESEARCH_KEYWORDS),
        "is_data":     any(kw in task_lower for kw in DATA_KEYWORDS),
        "is_verify":   any(kw in task_lower for kw in VERIFY_KEYWORDS),
    }
```

### 9.3 Context Weights (lines 16-21)

When a context feature is active, the arm's Thompson sample is multiplied by a per-arm weight:

```python
CONTEXT_WEIGHTS = {
    "is_code":     {"pipeline": 2.0, "single": 0.5, "supervisor": 0.7, "fanout": 0.8, "ensemble": 0.6},
    "is_research": {"supervisor": 2.0, "pipeline": 0.7, "single": 0.5, "fanout": 0.8, "ensemble": 0.6},
    "is_data":     {"fanout": 2.0, "pipeline": 0.8, "single": 0.5, "supervisor": 0.7, "ensemble": 0.6},
    "is_verify":   {"ensemble": 2.0, "pipeline": 0.6, "single": 0.5, "supervisor": 0.7, "fanout": 0.8},
}
```

Higher weight = more likely to be selected for that task type. The weights are multiplied across active features.

### 9.4 Selection (lines 85-93)

```python
async def select_topology(self, task, budget_band):
    await self.load()                        # load from Redis
    if self.total_tasks < MIN_TASKS_TO_LEARN:  # 5 tasks
        return None                          # cold start → no RL override
    features = self._extract_features(task)
    weights = self._compute_context_weights(features)  # multiply per active features
    return self._thompson_sample(weights)    # sample Beta distributions, apply weights, pick max
```

**The RL policy only returns a value after 5 tasks.** Before that, `select_topology` returns `None` and the optimizer uses LLM/rule-based selection without RL override.

### 9.5 Reward (lines 95-112)

Called from `agent/graph.py:85` after each task completes:

```python
quality = 1.0 if status == "completed" else 0.0
cost_eff = max(0.0, 1.0 - (budget.spent_pct / 100.0))
await rl.reward(topology=decision.topology, quality=quality, cost_efficiency=cost_eff)
```

**Reward calculation** (line 96):
```python
combined = quality * 0.7 + cost_efficiency * 0.3
```

If `combined > 0.5` → increment alpha (success). Else → increment beta (failure). Both Redis and `rl_policy.json` file are updated.

### 9.6 Persistence

Two-layer persistence:
1. **Redis**: keys `rl_policy:arm:{topology}` (hash with alpha/beta) and `rl_policy:total_tasks` (string)
2. **File fallback**: `rl_policy.json` in project root — loaded when Redis has 0 tasks, saved after every reward

---

## 10. Infrastructure

### 10.1 Docker: `Dockerfile`

Multi-stage build:
1. **Builder stage**: installs Python deps to `/install` prefix
2. **Runtime stage**: `python:3.12-slim`, copies deps from builder, creates `appuser`, runs as non-root

### 10.2 Docker Compose: `docker-compose.yml`

Two services:
- `api`: builds from Dockerfile, port 8000, depends on `redis` (health check)
- `redis`: `redis:7-alpine`, port 6379, with healthcheck

### 10.3 CI/CD: `.github/workflows/deploy.yml`

GitHub Actions workflow triggered on push. Builds Docker image, deploys.

### 10.4 Frontend: `static/`

Vanilla JS + CSS (no build step):
- `index.html` — task submission form, status polling, WebSocket event log, audit trail viewer
- `app.js` — polls `/tasks/{id}` every 3s, connects WebSocket for live events, renders topology/status/budget
- `style.css` — basic styling

---

## 11. Configuration

**`.env`** file:
```env
LLM_PROVIDER=mistral                    # openai | mistral | ollama
MISTRAL_API_KEY=your-key
MISTRAL_MODEL=mistral-large-latest
BUDGET_MAX_COST_USD=1.00
BUDGET_MAX_TOKENS=100000
LLM_REQUEST_TIMEOUT=30
REDIS_URL=redis://localhost:6379/0
API_HOST=0.0.0.0
API_PORT=8000
JWT_SECRET=change-me-in-production
```

---

## 12. Testing

### 12.1 E2E Test: `test_e2e.py`

Submits a fibonacci task, polls every 3s for up to 90s, prints result and logs.

### 12.2 Stress Test: `tests/stress_test.py`

Four test suites:
1. **Topology sweep** — runs same task with each topology (single, pipeline, supervisor)
2. **Budget sweep** — runs same task with varying budgets ($0.05, $0.10, $0.50)
3. **Task complexity** — runs 4 different tasks (simple, code, research, complex)
4. **Concurrent** — runs all 4 tasks simultaneously via ThreadPoolExecutor

**Usage**:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
python tests/stress_test.py
```

### 12.3 Unit Tests: `tests/unit/`

| Test file | What it tests |
|-----------|---------------|
| `test_budget.py` | BudgetTracker bands, tier allowance, usage recording |
| `test_degrader.py` | Topology degradation chain for all bands |
| `test_escalation.py` | should_escalate logic (confidence, divergence, budget) |
| `test_events.py` | EventBroadcaster publish/subscribe, history |
| `test_optimizer.py` | Rule-based topology selection, fallback logic |
| `test_rl_policy.py` | Feature extraction, context weights, Thompson sampling |
| `test_audit.py` | Audit recording and retrieval |
| `test_llm_helpers.py` | Token/cost estimation |
| `test_executor_tools.py` | Tool registry, base tool execution |
| `test_finalizer.py` | Result deduplication and combining |
| `test_planner_complexity.py` | Task complexity analysis (1/2/3) |
| `test_core_fixes.py` | Edge cases and regression tests |

### 12.4 Integration Tests: `tests/integration/test_api.py`

Tests the full API pipeline with running server.

---

## 13. Key Code Paths Quick Reference

| What | File | Line |
|------|------|------|
| Server startup | `api/main.py` | 11 |
| POST /execute handler | `api/routes/execute.py` | 40 |
| Background task runner | `api/routes/execute.py` | 17 |
| Task status endpoint | `api/routes/tasks.py` | 9 |
| Audit endpoint | `api/routes/audit.py` | 9 |
| WebSocket endpoint | `api/websocket.py` | 8 |
| Main entry point | `agent/graph.py` | 15 |
| RL reward call | `agent/graph.py` | 85 |
| Optimizer (LLM + rules + RL) | `core/optimizer.py` | 96 |
| Rule-based fallback | `core/optimizer.py` | 60 |
| Budget bands | `core/budget.py` | 30 |
| Topology degradation | `core/degrader.py` | 6 |
| Escalation logic | `core/escalation.py` | 6 |
| Thompson sampling | `core/rl_policy.py` | 68 |
| RL reward | `core/rl_policy.py` | 95 |
| Event publish | `core/events.py` | 14 |
| Redis connection | `core/redis_client.py` | 12 |
| AgentState schema | `agent/state.py` | 25 |
| Planner node | `agent/nodes/planner.py` | 82 |
| Executor ReAct loop | `agent/nodes/executor.py` | 169 |
| Validator confidence | `agent/nodes/validator.py` | 54 |
| Judge arbitration | `agent/nodes/judge.py` | 18 |
| Finalizer dedup | `agent/nodes/finalizer.py` | 10 |
| Graph builder | `agent/topologies/builder.py` | 22 |
| Single topology | `agent/topologies/single.py` | 35 |
| Pipeline topology | `agent/topologies/pipeline.py` | 30 |
| Supervisor topology | `agent/topologies/supervisor.py` | 76 |
| Fanout topology | `agent/topologies/fanout.py` | 118 |
| Ensemble topology | `agent/topologies/ensemble.py` | 40 |
| Tool registry | `agent/tools/registry.py` | 8 |
| Code executor tool | `agent/tools/code_executor.py` | 30 |
| Web search tool | `agent/tools/web_search.py` | 26 |
| File ops tools | `agent/tools/file_ops.py` | 20 |
| DB query tool | `agent/tools/db_query.py` | 33 |
| Settings/config | `core/config.py` | 4 |
| LLM factory | `core/llm.py` | 60 |
