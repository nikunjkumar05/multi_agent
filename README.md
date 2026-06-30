# BAMASE — Budget-Aware Multi-Agent Task Executor

This repository contains the implementation for the AAAI-26 paper "BAMAS: Structuring Budget-Aware Multi-Agent Systems". Yang, L., Luo, J., Liu, X., Lou, Y., & Chen, Z. (2026). BAMAS: Structuring Budget-Aware Multi-Agent Systems.

A budget-aware multi-agent system that accepts plain-English tasks via API, uses a cost-tier optimizer (adapted from [BAMAS](https://arxiv.org/abs/2504.11428)) to select topology and model tiers, then orchestrates four specialized agents with reasoning-divergence escalation and pre-execution topology degradation under budget pressure.

## Architecture

```
POST /execute {task, budget_usd}
        │
        ▼
┌─────────────────────┐
│ Cost-Tier Optimizer │ ← LLM + rule-based fallback
│  selects topology   │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Agent Team         │
│  Planner → Executor │
│  → Validator        │
└────────┬────────────┘
         ▼
┌─────────────────────┐ 
│ Escalation Engine   │ ← triggers on reasoning     
│ → Judge (if needed) │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Budget Governor    │ ← collapses topology at 90% spent
│  degrades topology  │
└────────┬────────────┘
         ▼
   Result + Audit Trail
```

## Features

- **Cost-tier optimizer** — LLM-based topology selection with rule-based fallback (BAMAS adaptation)
- **5 topology modes** — single, pipeline, supervisor, fanout, ensemble
- **Reasoning-divergence escalation** — Judge invoked when Validator confidence drops
- **Pre-execution topology collapse** — degrades ensemble → fanout → supervisor → pipeline → single before graph starts
- **Budget bands** — HEALTHY (<70%), TIER_DOWNGRADE (70-90%), STRUCTURAL_DEGRADE (90-100%), CRITICAL (>100%)
- **Mistral LLM** — configurable via `.env`, supports OpenAI/Ollama fallbacks
- **Real-time frontend** — status polling, audit trail display
- **Stress test suite** — topology sweep, budget sweep, concurrent testing

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your Mistral API key
```

### 3. Start the server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open the frontend

Navigate to [http://localhost:8000](http://localhost:8000)

### 5. Run a task

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{"task": "Write a Python function to compute fibonacci numbers", "budget_usd": 0.50}'
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Frontend UI |
| `GET` | `/health` | Health check |
| `POST` | `/execute` | Submit a task |
| `GET` | `/tasks/{task_id}` | Get task status |
| `GET` | `/audit/{task_id}` | Get audit trail |

### POST /execute

```json
{
  "task": "Write a Python function to compute fibonacci numbers",
  "budget_usd": 0.50,
  "topology": "pipeline"  // optional: single, pipeline, supervisor, fanout, ensemble
}
```

## Project Structure

```
multi_agent/
├── core/                    # Core infrastructure
│   ├── config.py           # Settings (Mistral/OpenAI/Ollama)
│   ├── llm.py              # Multi-provider LLM factory
│   ├── budget.py           # BudgetTracker with 4 bands
│   ├── optimizer.py        # Cost-tier optimizer + rule-based fallback
│   ├── degrader.py         # Mid-execution topology collapse
│   ├── escalation.py       # Reasoning-divergence escalation
│   └── audit.py            # Audit trail singleton
├── agent/                   # Agent layer
│   ├── state.py            # AgentState TypedDict
│   ├── graph.py            # run_task() entry point
│   ├── nodes/              # Agent nodes
│   │   ├── planner.py      # Plans task steps (1-3 max)
│   │   ├── executor.py     # Executes steps with context
│   │   ├── validator.py    # Validates with confidence scoring
│   │   ├── judge.py        # Ensemble arbitration
│   │   ├── escalation.py   # Escalation decision wrapper
│   │   └── finalizer.py    # Smart result selection
│   ├── topologies/         # Graph topologies
│   │   ├── single.py       # planner → executor → validator (loop)
│   │   ├── pipeline.py     # Sequential with judge
│   │   ├── supervisor.py   # Supervisor dispatches to workers
│   │   ├── fanout.py       # 3 parallel workers
│   │   ├── ensemble.py     # 2 prompt variants + 1 different model
│   │   └── builder.py      # compile_graph() selector
│   └── tools/              # Tool implementations
│       ├── code_executor.py
│       ├── web_search.py
│       ├── file_ops.py
│       └── db_query.py
├── api/                     # FastAPI layer
│   ├── main.py             # App + CORS + static files
│   ├── models/schemas.py   # Pydantic models
│   └── routes/             # API routes
│       ├── execute.py
│       ├── tasks.py
│       └── audit.py
├── static/                  # Frontend
│   ├── index.html
│   ├── style.css
│   └── app.js
├── tests/
│   └── stress_test.py      # Stress test suite
├── docs/
│   ├── plan.md             # Full architecture plan
│   └── file-spec.md        # File specifications
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```

## Topology Selection

The optimizer selects topology based on task complexity:

| Topology | Use Case | Example |
|----------|----------|---------|
| `single` | Trivial Q&A, one-liner | "What is 2+2?" |
| `pipeline` | Code generation, writing | "Write a fibonacci function" |
| `supervisor` | Research, explanations | "Explain TCP vs UDP" |
| `fanout` | Data analysis, parallel | "Analyze these datasets" |
| `ensemble` | Critical validation | "Audit this code" |

Override via API: `{"task": "...", "topology": "pipeline"}`

## Budget Bands

| Band | Spent % | Action |
|------|---------|--------|
| HEALTHY | <70% | Full flexibility |
| TIER_DOWNGRADE | 70-90% | frontier→standard, standard→cheap |
| STRUCTURAL_DEGRADE | 90-100% | Collapse topology (ensemble→fanout→...) |
| CRITICAL | >100% | Only cheap model, skip Judge |

## Stress Test

```bash
# Start server first
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Run stress test
python tests/stress_test.py
```

Tests: topology sweep, budget sweep, task complexity, concurrent execution.

## Tech Stack

- **LLM**: Mistral (mistral-tiny, mistral-small-latest, mistral-large-latest)
- **Framework**: LangGraph + LangChain
- **API**: FastAPI + Uvicorn
- **Frontend**: Vanilla JS + CSS (no build step)
- **State**: In-memory dict + Redis pub/sub events
- **Optimizer**: LLM semantic classification + rule-based fallback + RL refinement

## Technical Deep Dive

See [`learn.md`](learn.md) for a comprehensive technical reference covering every module, data flow, code path, and implementation detail.

## License

MIT
