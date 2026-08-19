# BAMAS Middleware: Budget-Aware Proxy for Coding AI Agents

## Vision

BAMAS becomes a middleware layer that sits between users and existing coding AI agents (Cursor, Codex, OpenCode, Aider, etc.). It routes tasks to the cheapest capable agent, enforces budget constraints, and degrades gracefully when budget tightens. The ILP solver selects the optimal agent pool, the budget gate prevents overruns, and topology degradation ensures tasks complete within budget.

**Value Proposition:** Teams spend $500-2000/month on AI coding tools. BAMAS reduces that to $200-500/month while maintaining quality.

---

## Architecture

```
User/App
  │
  ▼
┌─────────────────────────────────────────────────────┐
│                   BAMAS API                         │
│  POST /tasks          GET /tasks/:id                │
│  POST /budgets        GET /budgets/:id              │
│  WebSocket /stream    GET /analytics                │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│              Task Classifier                        │
│  - code_generation  - code_review                   │
│  - refactoring      - debugging                     │
│  - documentation    - testing                       │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│           ILP Router (Existing)                     │
│  - solve_ilp(task_type, budget, available_agents)   │
│  - Returns: cheapest agent that can handle task     │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│           Budget Gate (Existing)                    │
│  - Circuit breaker at 100%                          │
│  - Judge skip at 80%                                │
│  - Executor loop at 75%                             │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│          Agent Adapter Layer (NEW)                  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │ OpenCode│ │  Aider  │ │ Cursor  │ │  Codex  │  │
│  │ Adapter │ │ Adapter │ │ Adapter │ │ Adapter │  │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘  │
└───────┼───────────┼───────────┼───────────┼────────┘
        │           │           │           │
        ▼           ▼           ▼           ▼
   OpenCode      Aider       Cursor      Codex API
      CLI          CLI        API          API
```

---

## Core Components

### Already Implemented

| Component | File | Purpose |
|-----------|------|---------|
| ILP Solver | `core/ilp_solver.py` | Selects optimal agent/LLM pool within budget |
| Budget Gate | `agent/nodes/budget_gate.py` | Enforces budget, fires circuit breaker |
| Cost Model | `core/llm.py`, `core/config.py` | Paper Eq. 1: c_i = T_in * P_in + T_out * P_out |
| Topology Degradation | `core/budget.py`, `core/degrader.py` | Graceful fallback under budget pressure |
| Feedback Topology | `agent/topologies/feedback.py` | Iterative refinement with cost awareness |
| RL Policy | `core/rl_policy.py` | Thompson Sampling for topology selection |

### New Components Needed

| Component | Purpose | Effort | Priority |
|-----------|---------|--------|----------|
| Agent Adapter Interface | Standardize communication with each agent | 3-5 days | P0 |
| Task Classifier | Classify task type for routing | 2-3 days | P0 |
| Agent Registry | Store agent capabilities, pricing, endpoints | 1-2 days | P0 |
| Budget Dashboard | Cost breakdown per task/project | 3-5 days | P1 |
| Webhook/Streaming | Push results back to user in real-time | 2-3 days | P1 |
| Agent Health Monitor | Track agent availability and latency | 1-2 days | P2 |
| Cost Analytics | Historical cost analysis and optimization | 2-3 days | P2 |

---

## Agent Adapters

### Adapter Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class AgentTask:
    task_id: str
    task_type: str          # code_generation, code_review, etc.
    prompt: str
    context: dict           # files, language, project info
    budget_usd: float
    timeout_seconds: int

@dataclass
class AgentResult:
    task_id: str
    agent: str
    output: str
    cost_usd: float
    tokens_used: int
    latency_ms: int
    success: bool
    error: str | None

class AgentAdapter(ABC):
    @abstractmethod
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute a task using this agent."""
        pass

    @abstractmethod
    def estimate_cost(self, task: AgentTask) -> float:
        """Estimate cost before execution."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if agent is available."""
        pass

    @abstractmethod
    def get_capabilities(self) -> dict:
        """Return supported task types and limits."""
        pass
```

### OpenCode Adapter

```python
class OpenCodeAdapter(AgentAdapter):
    """Adapter for OpenCode CLI agent."""

    def __init__(self, binary_path: str = "opencode"):
        self.binary = binary_path

    async def execute(self, task: AgentTask) -> AgentResult:
        # 1. Write task to temp file
        # 2. Run: opencode --file <task_file> --json
        # 3. Parse JSON output
        # 4. Return AgentResult
        pass

    def estimate_cost(self, task: AgentTask) -> float:
        # OpenCode uses Claude/GPT under the hood
        # Estimate based on prompt length + expected output
        pass
```

### Aider Adapter

```python
class AiderAdapter(AgentAdapter):
    """Adapter for Aider CLI agent."""

    def __init__(self, binary_path: str = "aider"):
        self.binary = binary_path

    async def execute(self, task: AgentTask) -> AgentResult:
        # 1. Initialize aider in project directory
        # 2. Run: aider --file <task_file> --no-git --json
        # 3. Parse output
        # 4. Return AgentResult
        pass
```

### Cursor Adapter

```python
class CursorAdapter(AgentAdapter):
    """Adapter for Cursor API (if available)."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.cursor.sh"

    async def execute(self, task: AgentTask) -> AgentResult:
        # 1. POST to Cursor API
        # 2. Poll for completion
        # 3. Return AgentResult
        pass
```

---

## API Design

### Endpoints

```
POST   /api/v1/tasks              Create and execute a task
GET    /api/v1/tasks/:id          Get task status and result
DELETE /api/v1/tasks/:id          Cancel a task

POST   /api/v1/budgets            Create a budget
GET    /api/v1/budgets/:id        Get budget status
PATCH  /api/v1/budgets/:id        Update budget limits

GET    /api/v1/agents             List available agents
GET    /api/v1/agents/:id         Get agent capabilities

GET    /api/v1/analytics/costs    Cost breakdown by task/agent
GET    /api/v1/analytics/performance  Agent performance metrics

WebSocket /ws/tasks/:id          Real-time task updates
```

### Request/Response Examples

**Create Task:**
```json
POST /api/v1/tasks
{
  "task_type": "code_generation",
  "prompt": "Write a Python function to parse CSV files",
  "context": {
    "language": "python",
    "project_root": "/path/to/project",
    "files": ["src/utils.py", "tests/test_utils.py"]
  },
  "budget_usd": 0.50,
  "timeout_seconds": 120,
  "preferred_agents": ["opencode", "aider"],
  "fallback_agents": ["codex"]
}
```

**Task Created:**
```json
{
  "task_id": "task_abc123",
  "status": "queued",
  "estimated_cost_usd": 0.12,
  "estimated_tokens": 3000,
  "selected_agent": "opencode",
  "selected_tier": "standard",
  "ws_url": "ws://localhost:8000/ws/tasks/task_abc123"
}
```

**Task Complete:**
```json
{
  "task_id": "task_abc123",
  "status": "completed",
  "agent": "opencode",
  "output": "def parse_csv(file_path: str) -> list[dict]:\n    ...",
  "cost_usd": 0.089,
  "tokens_used": 2234,
  "latency_ms": 4500,
  "quality_score": 0.92,
  "budget_remaining_usd": 0.411
}
```

---

## Budget Management

### Budget Lifecycle

```
User creates budget ($10/day)
  │
  ▼
Budget Gate tracks all task costs
  │
  ▼
┌─────────────────────────────────────────┐
│ Budget Band        │ Action             │
├─────────────────────────────────────────┤
│ HEALTHY (<70%)     │ Full flexibility   │
│ TIER_DOWNGRADE     │ Frontier → Standard│
│ STRUCTURAL_DEGRADE │ Complex → Simple   │
│ CRITICAL (>90%)    │ Cheap only         │
│ CIRCUIT_BREAKER    │ Hard stop at 100%  │
└─────────────────────────────────────────┘
```

### Cost Allocation

```
Task Budget: $0.50
  │
  ├─ ILP Allocation:
  │   ├─ Executor: $0.25 (standard tier)
  │   ├─ Judge:    $0.15 (standard tier)
  │   ├─ Planner:  $0.10 (cheap tier)
  │   └─ Reserve:  $0.00 (budget gate syncs)
  │
  └─ Runtime:
      ├─ Task starts → budget_gate checks before each LLM call
      ├─ 70% spent → executor loop breaks early
      ├─ 80% spent → judge skipped
      └─ 100% spent → circuit breaker fires
```

---

## Task Classification

### Task Types

| Type | Description | Typical Agent | Typical Cost |
|------|-------------|---------------|--------------|
| `code_generation` | Write new code | OpenCode, Aider | $0.05-0.20 |
| `code_review` | Review existing code | OpenCode, Codex | $0.03-0.10 |
| `refactoring` | Restructure code | Aider, Cursor | $0.05-0.15 |
| `debugging` | Find and fix bugs | OpenCode, Cursor | $0.03-0.12 |
| `documentation` | Write docs/comments | Aider, OpenCode | $0.02-0.08 |
| `testing` | Write tests | Aider, Codex | $0.03-0.10 |
| `explanation` | Explain code | Any | $0.01-0.05 |

### Classification Strategy

1. **Rule-based (fast):** Keywords in prompt → task type
   - "write", "create", "implement" → code_generation
   - "review", "check", "audit" → code_review
   - "refactor", "restructure", "clean" → refactoring
   - "bug", "error", "fix", "broken" → debugging

2. **LLM-based (accurate):** Small model classifies ambiguous prompts
   - Only used when rule-based confidence < 80%
   - Uses cheapest available model

---

## Agent Registry

### Registry Schema

```python
@dataclass
class AgentInfo:
    agent_id: str              # "opencode", "aider", "cursor"
    display_name: str          # "OpenCode"
    version: str               # "0.1.0"
    api_type: str              # "cli", "http", "websocket"
    capabilities: list[str]    # ["code_generation", "code_review", ...]
    pricing: dict              # {"input_per_1k": 0.003, "output_per_1k": 0.015}
    max_tokens: int            # 128000
    latency_p50_ms: int        # 2000
    latency_p99_ms: int        # 10000
    reliability: float         # 0.98 (98% success rate)
    health_endpoint: str | None
    enabled: bool
```

### Default Registry

| Agent | API | Capabilities | Input Cost/1K | Output Cost/1K |
|-------|-----|--------------|---------------|----------------|
| OpenCode | CLI | code_gen, review, refactor, debug, doc, test | $0.003 | $0.015 |
| Aider | CLI | code_gen, review, refactor, debug, doc, test | $0.003 | $0.015 |
| Cursor | HTTP | code_gen, review, refactor, debug | $0.010 | $0.030 |
| Codex | HTTP | code_gen, review, test | $0.010 | $0.030 |

---

## Implementation Plan

### Phase 1: Core Middleware (Week 1-2)
- [ ] Create `middleware/` directory
- [ ] Implement `AgentAdapter` ABC
- [ ] Implement `AgentRegistry`
- [ ] Implement `TaskClassifier`
- [ ] Wire ILP solver to agent selection
- [ ] Add API endpoints

### Phase 2: Agent Adapters (Week 3-4)
- [ ] OpenCode adapter (CLI, stdio)
- [ ] Aider adapter (CLI, stdio)
- [ ] Adapter test harness
- [ ] Health monitoring

### Phase 3: Budget Dashboard (Week 5)
- [ ] Cost breakdown API
- [ ] Real-time budget tracking
- [ ] WebSocket streaming for budget updates

### Phase 4: Advanced Features (Week 6-8)
- [ ] Cursor adapter (HTTP API)
- [ ] Codex adapter (HTTP API)
- [ ] Cost analytics dashboard
- [ ] Agent performance comparison
- [ ] Automatic agent selection learning

---

## Pricing Model

### Tiers

| Tier | Price | Tasks/Month | Features |
|------|-------|-------------|----------|
| Free | $0 | 100 | Basic routing, single agent |
| Pro | $20/month | 1,000 | ILP optimization, all agents, budget dashboard |
| Team | $50/month | Unlimited | Analytics, custom agents, priority support |
| Enterprise | Custom | Unlimited | On-prem, SLA, custom integrations |

### Cost Savings Example

| Before (Direct) | After (BAMAS) | Savings |
|-----------------|---------------|---------|
| $1,500/month (Cursor Pro + Codex) | $600/month | 60% |
| 1,319 tasks × $1.14 avg | 1,319 tasks × $0.46 avg | 60% |
| 0% OOB tasks | 0% OOB tasks | Same quality |

---

## Directory Structure

```
middleware/
├── __init__.py
├── adapters/
│   ├── __init__.py
│   ├── base.py              # AgentAdapter ABC
│   ├── opencode.py          # OpenCode adapter
│   ├── aider.py             # Aider adapter
│   ├── cursor.py            # Cursor adapter
│   └── codex.py             # Codex adapter
├── classifier/
│   ├── __init__.py
│   └── task_classifier.py   # Rule + LLM classification
├── registry/
│   ├── __init__.py
│   └── agent_registry.py    # Agent capabilities + pricing
├── budget/
│   ├── __init__.py
│   └── budget_manager.py    # Budget lifecycle management
├── api/
│   ├── __init__.py
│   ├── routes/
│   │   ├── tasks.py         # Task CRUD
│   │   ├── budgets.py       # Budget CRUD
│   │   ├── agents.py        # Agent info
│   │   └── analytics.py     # Cost analytics
│   └── websocket.py         # Real-time streaming
├── models/
│   ├── __init__.py
│   └── schemas.py           # Pydantic models
└── tests/
    ├── test_adapters.py
    ├── test_classifier.py
    ├── test_budget.py
    └── test_api.py
```

---

## Migration Path

### Current State
```
BAMAS (standalone agent system)
├── ILP solver ✅
├── Budget gate ✅
├── Topologies ✅ (single, pipeline, supervisor, fanout, ensemble, feedback)
├── Cost model ✅
└── RL policy ✅ (Thompson Sampling)
```

### Target State
```
BAMAS (middleware)
├── Core (existing)
│   ├── ILP solver → agent selection
│   ├── Budget gate → cost enforcement
│   ├── Cost model → agent pricing
│   └── Topology degradation → agent fallback
├── New
│   ├── Agent adapters (OpenCode, Aider, Cursor, Codex)
│   ├── Task classifier
│   ├── Agent registry
│   └── Budget dashboard
└── API (FastAPI)
    ├── /tasks (create, status, cancel)
    ├── /budgets (create, update, status)
    ├── /agents (list, capabilities)
    └── /ws (real-time updates)
```

---

## Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Cost reduction vs direct | > 50% | Track before/after per user |
| Task completion rate | > 95% | Successful tasks / total tasks |
| Average latency | < 10s | p50 latency across all agents |
| OOB tasks | 0% | Tasks exceeding budget |
| Agent availability | > 99% | Health check success rate |
| User retention | > 80% | Monthly active users |

---

## Next Steps

1. **Create `middleware/` directory structure**
2. **Implement `AgentAdapter` ABC** (base interface)
3. **Implement `OpenCodeAdapter`** (first adapter)
4. **Wire ILP solver to agent selection**
5. **Add API endpoints for task creation**
6. **Test with OpenCode end-to-end**
7. **Add budget dashboard**
8. **Implement remaining adapters**
