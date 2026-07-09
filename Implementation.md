# BAMAS — Implementation Guide
> *Novelty analysis, codebase improvement suggestions, and phased build roadmap*

---

## Table of Contents

1. [Why BAMAS Is Genuinely Novel](#1-why-bamas-is-genuinely-novel)
2. [Competitive Landscape (Web Research)](#2-competitive-landscape-web-research)
3. [Critical Gaps: plan.md vs. Current Code](#3-critical-gaps-planmd-vs-current-code)
4. [Suggested Changes to plan.md](#4-suggested-changes-to-planmd)
5. [Suggested Changes to commercialization_roadmap.md](#5-suggested-changes-to-commercialization_roadmapmd)
6. [Concrete Code Improvements](#6-concrete-code-improvements)
7. [Phased Implementation Roadmap](#7-phased-implementation-roadmap)
8. [Summary: Top 5 Next Priorities](#summary-top-5-next-priorities)
9. [Budget Hardening (P1–P7)](#9-budget-hardening-p1p7)
10. [Future Research Directions](#10-future-research-directions)

---

## 1. Why BAMAS Is Genuinely Novel

### 1.1 What No Other System Does

BAMAS combines **three mechanisms** that individually exist elsewhere but are never combined into one unified runtime:

| Mechanism | BAMAS | Competitors |
|---|---|---|
| LLM-guided topology selection (semantic) | ✅ Primary path | ❌ None |
| Contextual Multi-Armed Bandit (Thompson Sampling) per topology arm | ✅ RL refinement | Partial (TREACLE uses RL for model cascade only) |
| Pre-execution topology **structural degradation** under budget pressure | ✅ Novel ★ | ❌ None — all others swap models, not topology shape |
| Reasoning-divergence escalation (not vote count) | ✅ Novel ★ | ❌ None |
| 4-band budget governor with automatic downgrade chain | ✅ Novel ★ | ❌ None |

The **topology degradation chain** (`ensemble→fanout→supervisor→pipeline→single`) is the single most unique engineering contribution. No published framework does pre-run structural graph collapse as a budget response. LangSight, AgentPrune, and BudgetMLAgent all stop at model-tier swapping.

### 1.2 What BAMAS Adapts from the AAAI-26 Paper

The original BAMAS paper (Yang et al., 2026, AAAI-40) uses **Integer Linear Programming (ILP)** for model selection. This implementation **replaces ILP with LLM semantic classification** as the primary optimizer, keeping RL (Thompson Sampling) for topology refinement. This is a valid and practical departure because:

- ILP requires offline profiling of model performance per task type — expensive to maintain.
- LLM classification is zero-shot generalisable, costs ~1 LLM call, and falls back to rules.
- The RL layer provides the adaptive optimisation the paper intended ILP to give.

**This deviation should be documented as a conscious design decision**, not an omission.

### 1.3 What the Market Is Saying (2026 Data)

- AI agent software spend hits **$206.5 billion in 2026** (Gartner, May 2026) — up 139% YoY.
- The AI agents market grows from **$7.6B (2025) → $182.9B (2033)** at 49.6% CAGR (Grand View Research).
- **Production cost blowouts are the #1 pain point** — one team's prototype costing $20/month hit $9,800 in month two (Edgeless Lab, 2026). This is BAMAS's exact wedge.
- Only 14–23% of enterprises have reached production-scale agent deployment. The market needs cost governance tooling *now* to unlock the remaining 77–86%.
- LangGraph's durable execution (checkpoints, retries, fan-out) **amplifies cost failures** — this is documented at runcycles.io (March 2026), which directly validates BAMAS's approach.

---

## 2. Competitive Landscape (Web Research)

### 2.1 Direct Competitors

| System | Cost Strategy | Topology Selection | Runtime Budget Enforcement |
|---|---|---|---|
| **BAMAS (this repo)** | LLM + RL + topology degradation | 5 topologies, RL-selected | 4-band governor, pre-execution |
| AgentBalance (HKUST, 2025) | Backbone-then-topology, latency-aware | Adaptive topology synthesis | At design time only |
| BudgetMLAgent (TCS, 2024) | LLM cascade + profiling + expert calls | Fixed (cascade) | None |
| TREACLE (NeurIPS 2024) | RL for model + prompt selection | No topology concept | Per-query budget |
| AgentPrune (2024) | Communication graph pruning | Fixed topology, prunes messages | None |
| LangSight (2026) | Budget cap via monkey-patch | None | Per-session cap (hard stop) |
| AgentOps Control Plane | Cost attribution + quality scoring | None | None |

### 2.2 White Space BAMAS Owns

1. **Mid-execution budget enforcement with topology degradation** — nobody does this in real time.
2. **Reasoning-divergence as an escalation signal** — all others use vote-count or static thresholds.
3. **Multi-topology orchestrator as a drop-in middleware** — not just a proxy or a wrapper.
4. **RL policy that learns which topology works for which task type** — persistent, improves over time.

### 2.3 Risks to Novelty

- **AgentBalance** (Dec 2025) is the closest academic competitor. It does backbone-then-topology selection but lacks runtime budget governance.
- **LangGraph Platform** (LangChain Inc.) is building managed deployment with cost controls — potential to commoditise the proxy layer. BAMAS must differentiate at the topology-intelligence and RL layers.
- **OpenAI Realtime API** and native model routing (GPT-4o mini fallback) reduce the need for external orchestration at the cheap/standard tier boundary.

---

## 3. Critical Gaps: plan.md vs. Current Code

These are inconsistencies or gaps found by cross-referencing `docs/plan.md`, `feature_comparison.md`, and the actual source files.

### 3.1 RL Cold-Start Threshold Mismatch

`docs/plan.md` says:
> *"Only overrides after ≥10 trained tasks."* (Section: RL Policy Details)

`core/rl_policy.py` initialises from the plan description with a threshold of 5 for selection and uses separate logic for overriding. The documentation is inconsistent. **Fix**: make `RL_MIN_TASKS_FOR_OVERRIDE` a configurable setting in `core/config.py` and document the actual value.

### 3.2 ~~Pre-Execution Only — Not Mid-Execution~~ `[CLOSED]`

> **Resolution**: Implemented in P4–P7 (commit `19f1b6c`). Budget gates now fire at every synchronization barrier — post-planner, post-executor, post-supervisor, and post-judge — across all 5 topologies. Mid-execution degradation via `langgraph.types.interrupt()` is fully operational.

### 3.3 stats.py and learning.py (V3) Are Mentioned but Missing

`docs/file-spec.md` describes two V3 modules:
- `core/stats.py` — tracks topology × budget × task performance.
- `core/learning.py` — self-optimisation loop using Judge scores.

Neither file exists. The RL policy (`core/rl_policy.py`) partially covers `stats.py` functionality but lacks the Judge-score feedback loop. These should be scaffolded now, even if empty, so the architecture is complete.

### 3.4 ~~JWT Config Present, Not Integrated~~ `[CLOSED]`

> **Resolution**: JWT auth wired to all routes (`api/routes/auth.py`), WebSocket uses `?token=` query param (`api/websocket.py`). `DEFAULT_JWT_SECRET` extracted to `core/config.py` as single source of truth.

### 3.5 ~~WebSocket Heartbeat Missing~~ `[CLOSED]`

> **Resolution**: 20s heartbeat/ping implemented in `api/websocket.py`. Prevents reverse proxy idle timeout disconnections.

### 3.6 ~~Audit Trail in Memory Only~~ `[CLOSED]`

> **Resolution**: SQLite persistence via `aiosqlite` with in-memory fast path. `core/audit.py` implements `AuditTrail` class with DB fallback. Indexed by `task_id`. Survives process restarts.

---

## 4. Suggested Changes to plan.md

These are additions/corrections to `docs/plan.md`:

### 4.1 Add ILP → LLM Replacement Note

Add a section after **Cost-tier Optimizer — Flow**:

```
### Design Departure from AAAI-26 Paper
The original BAMAS paper uses Integer Linear Programming (ILP) for model selection.
This implementation replaces ILP with LLM semantic classification as the primary
optimizer. Rationale: ILP requires per-model performance profiling per task type,
which is expensive to maintain in production. LLM classification is zero-shot
generalizable and falls back gracefully to rules. The RL layer provides the adaptive
optimization the ILP was intended to supply.
```

### 4.2 Update RL Cold-Start Threshold

Change the RL Policy Details section to:

```
- **Cold start**: Returns None for first {RL_MIN_TASKS_FOR_SELECTION} tasks (configurable, default 5)
- **Override threshold**: Only overrides LLM decision after ≥{RL_MIN_TASKS_FOR_OVERRIDE} trained tasks (configurable, default 10)
```

### 4.3 Add Mid-Execution Checkpointing to Layer 5

Update the **Budget Governor ★** section to include a planned mid-execution mode:

```
| Spent | Band | Action |
|-------|------|--------|
| <70% | HEALTHY | Full topology, all tiers |
| 70-90% | TIER_DOWNGRADE | Downgrade model tiers only |
| 90-100% | STRUCTURAL_DEGRADE | Collapse topology pre-execution (current); pause and migrate mid-execution (planned) |
| >100% | CRITICAL | Single topology, cheap model, skip Judge |
```

### 4.4 Add stats.py and learning.py to Directory Structure

```
├── core/
│   ├── ...
│   ├── stats.py            # (V2) topology × budget × task performance tracker
│   └── learning.py         # (V2) Judge-score feedback loop → optimizer improvement
```

### 4.5 Add Related Work Section

```
## Related Work
| System | Paper | Key Difference |
|--------|-------|----------------|
| AgentBalance | arXiv 2512.11426, 2025 | Backbone-then-topology; no runtime budget enforcement |
| BudgetMLAgent | TCS Research, 2024 | LLM cascade; no topology abstraction |
| TREACLE | NeurIPS 2024 | RL model+prompt selection; no multi-topology orchestration |
| AgentPrune | arXiv 2410.02506 | Message pruning; no budget degradation |
```

---

## 5. Suggested Changes to commercialization_roadmap.md

### 5.1 Add Market Size Data (Strengthens the Case)

Insert before Section 1:

```
## Market Context (2026)
- AI agent software spend: $206.5B in 2026, up 139% YoY (Gartner)
- AI agents market: $7.6B (2025) → $182.9B by 2033 at 49.6% CAGR (Grand View Research)
- Production cost blowouts are the #1 documented pain point: prototype→production
  cost spikes of 5–15× are reported across the industry (Edgeless Lab, 2026)
- LangGraph's durable execution (retries, fan-out, checkpoints) amplifies cost
  failures — documented at runcycles.io (March 2026). BAMAS's budget governor
  directly addresses this pattern.
```

### 5.2 Reframe the Agentic Guardrail as a "Cost Control Layer"

The tagline *"We guarantee your AI agents will never exceed their budget"* is strong but slightly overpromises (CRITICAL band still runs a degraded single agent). A more defensible framing:

> *"BAMAS is the cost-intelligence layer for production multi-agent systems: it automatically selects the cheapest topology that meets your quality threshold, and degrades gracefully before budget overruns occur."*

### 5.3 Reprioritise GTM — CLI Tool to Step 1

The CLI tool (`bamas-cli`) is listed as Step 2 in GTM. It should be Step 1 because:
- It is a zero-infrastructure lead magnet.
- It produces shareable artefacts (Budget Burn Risk Reports) that go viral in dev communities.
- It makes the core algorithm discoverable before the full backend is needed.

### 5.4 Add a 4th Business Model: "Agentic Policy SDK"

```
| 4. Policy SDK for LangGraph | Package the budget governor + escalation engine as
  `pip install bamas-policy`. Developers add it as a governance node inside any
  LangGraph workflow. | LangGraph / LangChain ecosystem developers |
  Open-core: free for <$100/mo API spend per tenant, $49/mo commercial license above.
```

### 5.5 Add the Academic Citation as Social Proof

The AAAI-26 paper citation should appear prominently in the README and commercialization doc:

> *"Based on the peer-reviewed AAAI-26 paper: Yang, L. et al. (2026). BAMAS: Structuring Budget-Aware Multi-Agent Systems. AAAI-40, pp. 29802–29810."*

This is a rare and strong differentiator — most competing repos have no peer-reviewed backing.

---

## 6. Concrete Code Improvements

### 6.1 ~~Mid-Execution Budget Enforcement (Highest Priority)~~ `✅ IMPLEMENTED`

> **Status**: Implemented in P4–P7 (commit `19f1b6c`). See Section 9 for full details.
>
> Budget gates now fire at every synchronization barrier across all topologies:
> - **Post-planner**: Catches budget exhaustion before any execution starts
> - **Post-executor**: Existing gate, now returns delta-only values (P1 fix)
> - **Post-supervisor**: Catches budget exhaustion between supervisor dispatches
> - **Post-judge**: Catches budget exhaustion after judge LLM call
>
> Circuit breaker at 110% provides last-resort hard stop.
> Pre-LLM checks in executor/planner/supervisor skip calls entirely when budget exhausted.

---

### 6.2 ~~Persistent Audit Trail (core/audit.py)~~ `✅ IMPLEMENTED`

> **Status**: Implemented. `core/audit.py` uses SQLite via `aiosqlite` with in-memory fast path. DB initialized on startup. Indexed by `task_id`. Survives process restarts.

---

### 6.3 ~~WebSocket Heartbeat (api/websocket.py)~~ `✅ IMPLEMENTED`

> **Status**: Implemented. 20s heartbeat/ping in `api/websocket.py`. Prevents reverse proxy idle timeout.

---

### 6.4 Scaffold stats.py and learning.py (core/)

**`core/stats.py`** — Tracks topology × budget × task performance for the RL reward signal:

```python
"""
Tracks task outcomes by topology, budget band, and task type.
Feeds the RL policy's reward function with richer historical data.
"""
from dataclasses import dataclass, field
from collections import defaultdict
from statistics import mean

@dataclass
class TaskOutcome:
    topology: str
    budget_band: str
    task_type: str  # code / research / data / verify / general
    quality_score: float  # 0.0–1.0 from Judge or Validator
    cost_usd: float
    cost_efficiency: float  # budget_remaining / budget_total

class PerformanceStats:
    """In-memory (V1) performance stats. Persisted to Redis in V2."""
    
    def __init__(self):
        self._outcomes: list[TaskOutcome] = []
        self._by_topology: dict[str, list[TaskOutcome]] = defaultdict(list)
    
    def record(self, outcome: TaskOutcome):
        self._outcomes.append(outcome)
        self._by_topology[outcome.topology].append(outcome)
    
    def best_topology_for(self, task_type: str, budget_band: str) -> str | None:
        """Returns the topology with highest mean quality for a given context."""
        candidates = {
            topo: [o.quality_score for o in outcomes
                   if o.task_type == task_type and o.budget_band == budget_band]
            for topo, outcomes in self._by_topology.items()
        }
        scored = {t: mean(scores) for t, scores in candidates.items() if len(scores) >= 3}
        return max(scored, key=scored.get) if scored else None

stats = PerformanceStats()
```

**`core/learning.py`** — Closes the feedback loop between Judge output and the optimizer:

```python
"""
Self-optimisation loop. Judge scores (quality_score) are fed back into
both the RL policy reward and the performance stats tracker.
"""
from core.stats import stats, TaskOutcome
from core.rl_policy import RLPolicy

async def record_task_result(
    rl_policy: RLPolicy,
    topology: str,
    budget_band: str,
    task_type: str,
    quality_score: float,
    cost_usd: float,
    budget_total: float,
):
    """
    Called at task completion (in agent/graph.py after finalizer).
    Updates both the RL policy and the performance stats tracker.
    """
    cost_efficiency = max(0.0, 1.0 - (cost_usd / budget_total)) if budget_total > 0 else 0.0
    reward = quality_score * 0.7 + cost_efficiency * 0.3

    # Update RL policy
    arm_index = rl_policy.topology_to_arm(topology)
    if arm_index is not None:
        rl_policy.update(arm_index, reward)

    # Update stats tracker
    stats.record(TaskOutcome(
        topology=topology,
        budget_band=budget_band,
        task_type=task_type,
        quality_score=quality_score,
        cost_usd=cost_usd,
        cost_efficiency=cost_efficiency,
    ))
```

---

### 6.5 ~~Make RL Thresholds Configurable (core/config.py)~~ `✅ IMPLEMENTED`

> **Status**: Implemented. `core/config.py` has `RL_MIN_TASKS_FOR_SELECTION`, `RL_MIN_TASKS_FOR_OVERRIDE`, `RL_QUALITY_WEIGHT`, `RL_COST_EFFICIENCY_WEIGHT` as configurable settings.

---

### 6.6 ~~JWT Authentication (Minimal, Wire It Up)~~ `✅ IMPLEMENTED`

> **Status**: Implemented. JWT auth via `api/routes/auth.py` (Bearer token), `api/websocket.py` (?token= query param). `DEFAULT_JWT_SECRET` in `core/config.py`. Dev-mode bypass when default secret is in use.

---

### 6.7 ~~Dry-Run / Cost Estimation Endpoint~~ `✅ IMPLEMENTED`

> **Status**: Implemented as `POST /estimate` in `api/routes/estimate.py`. Returns topology, model tiers, estimated cost, budget headroom, and risk level (LOW/MEDIUM/HIGH). Powers the frontend cost estimation banner.

---

### 6.8 bamas-cli Scaffold

Create `cli/bamas_cli.py`:

```python
#!/usr/bin/env python3
"""
bamas-cli — Budget Burn Risk Analyser for LangGraph agents.
Usage: python -m cli.bamas_cli --task "..." --budget 1.00
"""
import argparse
import asyncio
import httpx

async def dry_run(task: str, budget: float, server: str = "http://localhost:8000"):
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{server}/estimate", json={"task": task, "budget_usd": budget})
        data = resp.json()
    
    print("\n=== BAMAS Budget Burn Risk Report ===")
    print(f"  Task         : {task[:80]}")
    print(f"  Budget       : ${budget:.2f}")
    print(f"  Topology     : {data['topology']}")
    print(f"  Model tiers  : {data['model_tiers']}")
    print(f"  Est. cost    : ${data['estimated_cost_usd']:.4f}")
    print(f"  Budget left  : {data['budget_headroom_pct']:.1f}%")
    print(f"  Rationale    : {data['rationale']}")
    
    risk = "LOW" if data["budget_headroom_pct"] > 30 else "MEDIUM" if data["budget_headroom_pct"] > 10 else "HIGH"
    print(f"\n  Risk Level   : {risk}")
    print("=====================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BAMAS Budget Burn Risk Analyser")
    parser.add_argument("--task", required=True, help="Task description")
    parser.add_argument("--budget", type=float, required=True, help="Budget in USD")
    parser.add_argument("--server", default="http://localhost:8000", help="BAMAS server URL")
    args = parser.parse_args()
    asyncio.run(dry_run(args.task, args.budget, args.server))
```

---

## 7. Phased Implementation Roadmap

```mermaid
gantt
    title BAMAS Implementation Phases
    dateFormat  YYYY-MM-DD
    section Phase 1 — Robustness
    Mid-execution budget checkpoint     :p1a, 2026-07-01, 21d
    Persistent audit trail (SQLite)     :p1b, 2026-07-01, 7d
    WebSocket heartbeat                 :p1c, 2026-07-08, 5d
    stats.py + learning.py scaffold     :p1d, 2026-07-10, 7d
    RL config externalisation           :p1e, 2026-07-05, 3d
    POST /estimate endpoint             :p1f, 2026-07-15, 5d
    section Phase 2 — Developer Tools
    bamas-cli dry-run tool              :p2a, 2026-07-22, 7d
    JWT auth middleware (wired)         :p2b, 2026-07-22, 5d
    plan.md + README update             :p2c, 2026-07-28, 3d
    GitHub release + PyPI packaging     :p2d, 2026-08-01, 10d
    section Phase 3 — Enterprise
    PostgreSQL audit persistence        :p3a, 2026-08-11, 10d
    Multi-tenancy (user isolation)      :p3b, 2026-08-15, 21d
    Encrypted credential vault          :p3c, 2026-09-01, 14d
    Next.js observability dashboard     :p3d, 2026-09-01, 30d
    section Phase 4 — Commercial
    SaaS billing / metering             :p4a, 2026-10-01, 30d
    LangGraph plugin packaging          :p4b, 2026-10-01, 21d
    Enterprise self-hosted license      :p4c, 2026-11-01, 21d
```

### Phase 1 — Robustness (Weeks 1–4)

**Goal**: Make the existing system production-safe with zero external dependencies added.

| Task | File(s) | Effort | Impact | Status |
|---|---|---|---|---|
| ~~Mid-execution budget checkpoint~~ | `agent/nodes/budget_gate.py`, topology files, `agent/state.py` | 3d | ★★★★★ | ✅ **Done** (P4–P7, commit `19f1b6c`) |
| ~~Persistent audit trail (SQLite)~~ | `core/audit.py` | 1d | ★★★★ | ✅ **Done** |
| ~~WebSocket heartbeat~~ | `api/websocket.py` | 0.5d | ★★★ | ✅ **Done** |
| ~~Scaffold stats.py + learning.py~~ | `core/stats.py`, `core/learning.py` | 1d | ★★★ | ✅ **Done** |
| ~~RL config externalisation~~ | `core/config.py`, `core/rl_policy.py` | 0.5d | ★★ | ✅ **Done** |
| ~~POST /estimate endpoint~~ | `api/routes/estimate.py`, `api/main.py` | 1d | ★★★★ | ✅ **Done** |
| Budget hardening (P1–P7) | 12 files | 2d | ★★★★★ | ✅ **Done** (commit `19f1b6c`) |

### Phase 2 — Developer Tools (Weeks 5–7)

**Goal**: Create lead magnets and make the system installable.

| Task | File(s) | Effort | Impact |
|---|---|---|---|
| `bamas-cli` dry-run tool | `cli/bamas_cli.py`, `pyproject.toml` | 1.5d | ★★★★★ |
| Wire JWT auth | `api/middleware/auth.py`, `api/routes/*.py` | 1d | ★★★ |
| Plan.md + README improvements | `docs/plan.md`, `README.md` | 1d | ★★★ |
| PyPI packaging (`pip install bamas`) | `pyproject.toml`, `MANIFEST.in` | 2d | ★★★★ |

### Phase 3 — Enterprise (Weeks 8–14)

**Goal**: Add the infrastructure needed for paying enterprise customers.

| Task | Depends On | Effort |
|---|---|---|
| PostgreSQL audit persistence | Phase 1 audit work | 2d |
| Multi-tenancy (org/user isolation in Redis + audit) | JWT from Phase 2 | 5d |
| Encrypted credential vault (per-user API keys) | Multi-tenancy | 3d |
| Next.js observability dashboard | POST /estimate + WebSocket | 2–4 weeks |

### Phase 4 — Commercial (Weeks 15+)

**Goal**: Monetise.

| Task | Description |
|---|---|
| Billing/metering infrastructure | Stripe integration for pay-as-you-go proxied spend |
| LangGraph plugin | `pip install bamas-policy` governance node |
| Enterprise self-hosted license | Docker image + license key enforcement |
| Technical blog post | Medium/HN deep dive referencing AAAI-26 paper |

---

## 9. Budget Hardening (P1–P7)

*Implemented: 2026-07-09 | Commit: `19f1b6c` | 12 files, +235/-38 lines*

### 9.1 Overview

Seven phases of production-grade budget enforcement, addressing bugs found by deep audit and code review. These changes transform the budget system from "mostly works" to "production-hardened with multiple safety nets."

### 9.2 Changes by Phase

| Phase | Fix | Files | Impact |
|-------|-----|-------|--------|
| **P1** | Double-counting: return delta, not `prev+delta` | 5 node files | Stops silent cost inflation at source |
| **P2** | Fanout cost tracking: aggregate worker deltas | `fanout.py` | Recovers lost cost data |
| **P3** | Ensemble race: dispatcher pre-allocates per-agent caps | `ensemble.py`, `state.py` | Prevents 3x race overrun |
| **P4** | Missing gates: post-planner, post-supervisor, post-judge | 4 topology files | Catches budget breaches at every sync point |
| **P5** | BudgetTracker sync: gate writes live values to tracker | `budget_gate.py` | Makes band detection work |
| **P6** | Pre-LLM checks: skip LLM call if budget exhausted | `budget.py`, 3 node files | Prevents wasting tokens on nodes that would be degraded |
| **P7** | Circuit breaker: 110% hard cap forces emergency stop | `budget_gate.py` | Last-resort safety net |

### 9.3 Detailed Technical Changes

#### P1: Double-Counting Fix

**Bug**: `consumed_tokens` and `consumed_cost` use `Annotated[type, operator.add]` as reducer. Sequential nodes returned `prev + delta` instead of `delta`, causing the reducer to compute `prev + (prev + delta)` — inflating costs on every node call.

**Fix**: All nodes return only their local delta:
- `executor.py`: `exec_tokens`/`exec_cost` (was `acc_tokens`/`acc_cost`)
- `validator.py`: `val_tokens`/`val_cost`
- `judge.py`: `judge_tokens`/`judge_cost`
- `supervisor.py`: `sup_tokens`/`sup_cost`
- `planner.py`: `local_tokens`/`local_cost`

#### P2: Fanout Cost Tracking

**Bug**: `parallel_workers_node` collected `step_results` and `logs` from workers but dropped `consumed_tokens`/`consumed_cost`.

**Fix**: Added `merged_tokens`/`merged_cost` accumulators that sum each worker's deltas.

#### P3: Ensemble Race Condition

**Bug**: Three parallel agents read the same stale `consumed_cost` and all proceeded — combined cost could exceed budget 3x before the gate fired.

**Fix**: New `ensemble_dispatcher` node divides remaining budget equally among agents. Each agent checks its pre-allocated cap before calling LLM. Global threshold lowered from 100% → 80%.

#### P4: Missing Budget Gates

**Bug**: No gate after planner, supervisor, or judge in most topologies.

**Fix**: Added `budget_gate_post_planner`, `budget_gate_post_supervisor`, `budget_gate_post_judge` nodes:

| Topology | Gates Added |
|----------|-------------|
| Single | post-planner, post-judge |
| Pipeline | post-planner, post-judge |
| Supervisor | post-planner, post-supervisor, post-judge |
| Fanout | post-planner |
| Ensemble | (already had full coverage) |

#### P5: BudgetTracker Sync

**Bug**: `BudgetTracker.consumed_cost` was never synced with `state.consumed_cost`. The `budget_checkpoint` node in `budget_interrupt.py` always returned HEALTHY.

**Fix**: `budget_gate_node` now writes `budget.consumed_cost = acc_cost` and `budget.consumed_tokens = acc_tokens` at every gate evaluation, before returning `{"budget": budget}`.

#### P6: Pre-LLM Budget Checks

**Bug**: Executor, planner, and supervisor called LLM without checking budget first. Tokens were wasted on nodes that would be degraded by the gate.

**Fix**: New `should_skip_llm(state, threshold=0.9)` utility in `core/budget.py`. Called before every LLM invocation:
- Executor: skips step, returns "[Step skipped - budget exhausted]"
- Planner: returns single-step fallback with 0 cost
- Supervisor: returns `{"status": "completed"}`, stops execution

#### P7: Hard Circuit Breaker

**Bug**: If all gates failed, no last-resort safety net existed.

**Fix**: `HARD_CAP_MULTIPLIER = 1.1` constant. If `consumed_cost >= budget.max_cost_usd * 1.1`, `budget_gate_node` emits `budget_circuit_breaker` event, records in audit trail, and calls `interrupt()` to force-stop.

### 9.4 Code Review Fixes

Applied during review of P1–P7:

| Issue | Severity | Fix |
|-------|----------|-----|
| Executor truncated-step return outside `if` block | Critical | Restored correct indentation (12 spaces) |
| Ensemble dispatcher stored `per_agent` instead of `acc_cost + per_agent` | Critical | Agents would never run — fixed cap calculation |
| `acc_tokens`/`acc_cost` naming ambiguous | Suggestion | Renamed to `total_tokens`/`total_cost` in executor |
| Inline band computation hard to maintain | Suggestion | Extracted `_spent_band()` helper |
| Supervisor skip event missing step count | Suggestion | Added `pending_steps` to event payload |
| Missing trailing newlines | Nitpick | Added to `budget.py`, `budget_gate.py` |

### 9.5 Budget Enforcement Layers (Post P1–P7)

The system now has **12 distinct budget enforcement mechanisms**:

```
Layer 1: Pre-execution degradation (core/degrader.py)
Layer 2: Pre-LLM budget check (should_skip_llm in executor/planner/supervisor)
Layer 3: Per-step budget caps (planner → executor)
Layer 4: Post-planner budget gate
Layer 5: Post-executor budget gate
Layer 6: Post-supervisor budget gate (supervisor topology)
Layer 7: Post-judge budget gate
Layer 8: Ensemble per-agent budget caps (dispatcher)
Layer 9: Ensemble global 80% threshold
Layer 10: BudgetTracker sync (band detection)
Layer 11: Circuit breaker at 110%
Layer 12: Orchestrator degradation chain (ensemble→fanout→supervisor→pipeline→single)
```

### 9.6 What This Means for Production

- **Cost transparency**: Every LLM call is tracked with delta-only values. No silent inflation.
- **Race condition safety**: Parallel agents (ensemble) cannot collectively exceed budget.
- **Zero-waste execution**: Nodes skip LLM calls entirely when budget is exhausted.
- **Multiple safety nets**: Even if one gate misses, the circuit breaker catches overruns at 110%.
- **Correct band detection**: `BudgetTracker` stays in sync with live state, so `budget_interrupt.py` fires accurately.

---

## 10. Future Research Directions

### 10.1 Online ILP (Restore Paper Fidelity)

The original AAAI-26 paper uses ILP for model selection. Implement a lightweight online ILP solver using `scipy.optimize.milp` (already in `requirements.txt` as `scipy`) that uses **observed cost-per-token data** from `core/stats.py` as ILP coefficients. This would make BAMAS fully paper-faithful and publishable as an extension.

### 10.2 Cross-Task RL Transfer

Current Thompson Sampling resets per deployment. Add **task embedding similarity**: when a new task arrives, find the 3 most similar past tasks (cosine similarity of task embeddings stored in Redis), and initialise the Thompson Sampling priors from their outcomes. This is a direct research contribution.

### 10.3 Topology-Aware Escalation

`docs/plan.md` notes:
> *"Current implementation is threshold-based. Not topology-aware."*

For supervisor topology, the escalation check should compare divergence between *worker sub-results*, not just executor vs. validator. This requires routing divergence signals through the supervisor dispatch loop.

### 10.4 Adversarial Budget Injection Resistance

As documented in AgentPrune (2024), multi-agent systems are vulnerable to adversarial messages inflating token counts. BAMAS's budget governor could be extended with a **token-budget pre-commitment** step: estimate max tokens per step from the plan, reject execution if a step would exceed its allocation.

### 10.5 Benchmark on SWE-Bench / GAIA

The AAAI-26 paper evaluates on 3 internal tasks. A public evaluation on **SWE-Bench** (code), **GAIA** (general), and **HumanEval** (code generation) would make BAMAS's cost savings claims independently verifiable and dramatically increase academic citations and GitHub visibility.

---

## Summary: Top 5 Next Priorities

| # | Action | File | Impact | Time |
|---|---|---|---|---|
| 1 | PostgreSQL audit persistence | `core/audit.py` | Unlocks production durability, multi-tenant audit | 2d |
| 2 | Prometheus metrics + Grafana dashboards | `api/main.py`, new `metrics/` | Production monitoring, SLA tracking | 3d |
| 3 | Rate limiting on API endpoints | `api/middleware/` | Security, fairness, abuse prevention | 1d |
| 4 | OpenTelemetry distributed tracing | `agent/nodes/`, `core/` | Debugging at scale, cross-service visibility | 2d |
| 5 | Kubernetes manifests + health probes | `deploy/` | Deployment maturity, horizontal scaling | 2d |

---

*Generated: 2026-07-09 | Based on AAAI-26 paper + repo analysis + web research*
*Last updated: Budget hardening P1–P7 complete (commit `19f1b6c`)*
