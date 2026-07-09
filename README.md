# BAMAS — Budget-Aware Multi-Agent System

A budget-aware multi-agent system that accepts plain-English tasks via API, uses a cost-tier optimizer (adapted from [BAMAS](https://arxiv.org/abs/2504.11428)) to select topology and model tiers, then orchestrates specialized agents with reasoning-divergence escalation and **mid-execution topology degradation** under budget pressure.

> Based on the peer-reviewed AAAI-26 paper: Yang, L. et al. (2026). *BAMAS: Structuring Budget-Aware Multi-Agent Systems.* AAAI-40, pp. 29802–29810.

## Architecture

```
POST /execute {task, budget_usd}
        │
        ▼
┌─────────────────────┐
│ Cost-Tier Optimizer │ ← LLM semantic + rule fallback + RL
│  selects topology   │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Budget Governor    │ ← pre-execution collapse
│  degrades topology  │
└────────┬────────────┘
         ▼
┌─────────────────────────────────────┐
│  Agent Team (topology-dependent)    │
│  Planner → Executor → Validator    │
│  → Budget Gate → Judge → Finalizer │
└────────┬────────────────────────────┘
         ▼
   Result + Audit Trail + WebSocket Events
```

## Features

- **5 topology modes** — single, pipeline, supervisor, fanout, ensemble
- **Mid-execution budget degradation** — interrupt → project → rebuild → resume
- **Cost-tier optimizer** — LLM semantic classification + rule-based fallback + Thompson Sampling RL
- **Reasoning-divergence escalation** — Judge invoked when Validator confidence drops
- **4-band budget governor** — HEALTHY → TIER_DOWNGRADE → STRUCTURAL_DEGRADE → CRITICAL
- **Real-time WebSocket events** — live execution log with heartbeat
- **SQLite audit trail** — persistent task history
- **JWT authentication** — dev-mode bypass for local development
- **CLI tool** — `bamas-cli` for budget burn risk analysis

## Quick Start

### Prerequisites

- Python 3.12+
- Redis (for RL policy and WebSocket events)
- Mistral API key (or OpenAI/Ollama)

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for testing
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env:
#   LLM_PROVIDER=mistral
#   MISTRAL_API_KEY=your-key-here
```

### 3. Start Redis

```bash
redis-server
```

### 4. Start the server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
```

### 5. Open the frontend

Navigate to [http://localhost:8001](http://localhost:8001)

### 6. Run a task

```bash
curl -X POST http://localhost:8001/execute \
  -H "Content-Type: application/json" \
  -d '{"task": "What is 2+2?", "budget_usd": 0.10}'
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Frontend UI |
| `GET` | `/health` | Health check |
| `POST` | `/execute` | Submit a task |
| `GET` | `/tasks/{task_id}` | Get task status and result |
| `GET` | `/audit/{task_id}` | Get audit trail |
| `POST` | `/estimate` | Dry-run cost estimation |
| `WS` | `/ws/{task_id}` | WebSocket real-time events |

### POST /execute

```json
{
  "task": "Write a Python function to compute fibonacci numbers",
  "budget_usd": 0.50,
  "topology": "pipeline"  // optional: single, pipeline, supervisor, fanout, ensemble
}
```

### POST /estimate

```json
{
  "task": "Write a Python function to compute fibonacci numbers",
  "budget_usd": 0.50
}
```

Returns: topology, model tiers, estimated cost, budget headroom.

## Topology Selection

| Topology | Use Case | Example |
|----------|----------|---------|
| `single` | Trivial Q&A, one-liner | "What is 2+2?" |
| `pipeline` | Multi-step sequential | "First do X, then Y" |
| `supervisor` | Complex with dispatch | "Research and explain" |
| `fanout` | Parallel workers | "List 5 things" |
| `ensemble` | Critical validation | "Audit this code" |

## Budget Bands

| Band | Spent % | Action |
|------|---------|--------|
| HEALTHY | <70% | Full flexibility |
| TIER_DOWNGRADE | 70-90% | Downgrade model tiers |
| STRUCTURAL_DEGRADE | 90-100% | Collapse topology |
| CRITICAL | >100% | Single topology, skip Judge |

## Project Structure

```
multi_agent/
├── core/                    # Core infrastructure
│   ├── config.py           # Settings (Mistral/OpenAI/Ollama)
│   ├── llm.py              # Multi-provider LLM factory
│   ├── budget.py           # BudgetTracker with 4 bands
│   ├── optimizer.py        # Cost-tier optimizer + RL
│   ├── rl_policy.py        # Thompson Sampling RL policy
│   ├── projections.py      # State projection for topology degrade
│   ├── learning.py         # Judge-score feedback loop
│   ├── stats.py            # Performance tracking
│   ├── audit.py            # SQLite audit trail
│   └── events.py           # Redis event broadcaster
├── agent/                   # Agent layer
│   ├── state.py            # AgentState TypedDict
│   ├── graph.py            # run_task() entry point
│   ├── orchestrator.py     # Mid-execution degradation loop
│   ├── nodes/              # Agent nodes
│   │   ├── planner.py      # Plans task steps
│   │   ├── executor.py     # ReAct loop with tools
│   │   ├── validator.py    # Confidence scoring
│   │   ├── judge.py        # Ensemble arbitration
│   │   ├── budget_gate.py  # Mid-execution interrupt
│   │   ├── entry_router.py # Resume routing
│   │   └── finalizer.py    # Result selection
│   ├── topologies/         # Graph topologies
│   │   ├── single.py       # Single agent loop
│   │   ├── pipeline.py     # Sequential steps
│   │   ├── supervisor.py   # Supervisor dispatch
│   │   ├── fanout.py       # Parallel workers
│   │   ├── ensemble.py     # 3 parallel agents
│   │   └── builder.py      # compile_graph()
│   └── tools/              # Tool implementations
│       ├── code_executor.py
│       ├── web_search.py
│       ├── file_ops.py
│       └── db_query.py
├── api/                     # FastAPI layer
│   ├── main.py             # App + CORS + static files
│   ├── models/schemas.py   # Pydantic models
│   ├── middleware/auth.py   # JWT authentication
│   └── routes/             # API routes
├── static/                  # Frontend
│   ├── index.html
│   ├── style.css
│   └── app.js
├── cli/                     # CLI tool
│   └── bamas_cli.py
├── tests/
│   ├── unit/               # 268 unit tests
│   ├── integration/        # Integration tests
│   └── stress_test.py      # Stress test suite
├── docs/
│   ├── plan.md             # Architecture plan
│   └── file-spec.md        # File specifications
├── pyproject.toml
├── requirements.txt
└── .env.example
```

## Running Tests

```bash
# Unit tests (268 tests, no server needed)
pytest tests/unit/ -v

# Integration tests (requires Mistral API key)
pytest tests/integration/ -v

# Stress test (requires running server)
python tests/stress_test.py
```

## Tech Stack

- **LLM**: Mistral (mistral-tiny, mistral-small-latest, mistral-large-latest)
- **Framework**: LangGraph + LangChain
- **API**: FastAPI + Uvicorn
- **Frontend**: Vanilla JS + CSS (no build step)
- **State**: LangGraph checkpointing + Redis pub/sub
- **Database**: SQLite (audit trail)
- **RL**: Thompson Sampling with 5 topology arms

## License

MIT
