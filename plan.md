# BAMAS Alignment Plan: Current Implementation vs Paper

**Goal:** Align the current codebase with the BAMAS paper (AAAI-26) by Yang et al.

---

## Paper Summary

BAMAS has 3 core components:
1. **Budget-Constrained LLM Provisioning** — ILP solver selects optimal LLM pool within budget
2. **Agent Collaboration Topology Selection** — Offline REINFORCE policy selects topology
3. **Agent Instantiation** — Assigns LLMs to topology roles

---

## Gap Analysis

### Component 1: Budget-Constrained LLM Provisioning (ILP)

| Aspect | Paper | Current | Gap |
|--------|-------|---------|-----|
| Algorithm | Integer Linear Programming (ILP) | LLM semantic classification + keyword fallback | **CRITICAL** — paper's core innovation missing |
| LLM selection | Binary decision vars x_ij per LLM tier | Fixed tier mapping in config | No optimization over LLM pool |
| Decision weights | W_i = 1 + sum(W_j * floor(B/c_j)) | None | Lexicographic optimality not guaranteed |
| Budget constraint | sum(c_i * x_i) <= B | BudgetTracker checks after execution | Cost is estimated, not optimized |
| Minimum agents | >= 2 LLMs selected | Single topology exists | Enforced only in ensemble |
| Cost model | c_i = T_in * P_in + T_out * P_out | estimate_cost() uses rough heuristics | Inaccurate cost predictions |

**Paper's ILP Formulation (Eq. 3):**
```
maximize  sum(W_i * x_ij)
subject to  sum(c_i * x_ij) <= B
            sum(x_ij) >= 2
```

### Component 2: Topology Selection (RL)

| Aspect | Paper | Current | Gap |
|--------|-------|---------|-----|
| Algorithm | Offline REINFORCE | Contextual Thompson Sampling | Different algorithm (TS is bandit, not full RL) |
| Training | Pre-collected dataset D, batch training | Online reward updates | No offline training pipeline |
| Action space | Linear, Star, Feedback, Planner-Driven | single, pipeline, supervisor, fanout, ensemble | Different topology set |
| Reward | R_final = w_perf * R_perf + w_cost * R_cost | quality_score (binary) + cost_efficiency | Missing overflow penalty, missing success bonus |
| Loss | -E[log pi(t|T,B) * R] - beta * H(pi) | N/A (no gradient-based training) | No proper loss function |
| Policy model | Neural network (pi_theta) | Beta distributions (Thompson Sampling) | Simpler but less expressive |

### Component 3: Agent Instantiation

| Aspect | Paper | Current | Gap |
|--------|-------|---------|-----|
| Role assignment | Highest-weight LLM -> most critical role | Hardcoded tier assignments | No weight-based assignment |
| Feedback topology | Generate-critique-revise loop | Not implemented | Paper's most-selected topology for math (40-70%) |
| Linear topology | Sequential reasoning | pipeline topology | Similar but not identical |
| Star topology | Parallel hypothesis generation | ensemble/fanout | Partial match |
| Planner-Driven | Central planner coordinates | supervisor topology | Partial match |

### Budget Adherence

| Metric | Paper | Current | Gap |
|--------|-------|---------|-----|
| OOB tasks (GSM8K) | 0 / 1,319 | Frequent 200-300% overruns | **CRITICAL** |
| OOB tasks (MBPP) | 1-5 / 500 | Not measured | No evaluation |
| Cost model accuracy | T_in=500, T_out=max from samples | Rough char/4 heuristic | Inaccurate |

---

## What Current Has That Paper Doesn't

| Feature | Value |
|---------|-------|
| Mid-execution topology degradation | Novel — paper only selects before execution |
| Budget gate with circuit breaker | Runtime budget enforcement |
| Tool-use executor (ReAct) | Paper's agents don't use tools |
| Validator node | Paper doesn't validate step outputs |
| FastAPI + Docker deployment | Production-ready infrastructure |
| WebSocket event streaming | Real-time monitoring |
| 28 test files | Comprehensive test coverage |

---

## Implementation Plan

### Phase 1: ILP Solver (Paper Component 1)
**Priority: CRITICAL | Effort: 2-3 days**

**Files to create:**
- `core/ilp_solver.py` — ILP formulation and solver

**Files to modify:**
- `core/optimizer.py` — Replace LLM topology selection with ILP
- `pyproject.toml` — Add `pulp` or use `scipy.optimize.linprog`

**Implementation steps:**
1. Add `pulp` dependency
2. Implement ILP solver:
   - Define LLM tiers with costs (T_in=500, T_out from sampling)
   - Compute decision weights W_i using Eq. 2
   - Formulate Eq. 3: maximize sum(W_i * x_ij) subject to budget + min 2 agents
   - Solve with PuLP CBC solver
3. Replace `CostTierOptimizer.optimize()` ILP path:
   - Keep LLM fallback for when ILP fails
   - Add ILP result to `OptimizerDecision`
4. Update cost model to use actual token pricing from provider

**Acceptance criteria:**
- ILP solver selects LLM pool within budget
- Decision weights ensure lexicographic optimality
- At least 2 LLMs selected
- Cost model matches paper's Eq. 1

### Phase 2: REINFORCE Topology Selection (Paper Component 2)
**Priority: HIGH | Effort: 5-7 days**

**Files to create:**
- `core/reinforce.py` — REINFORCE policy with neural network
- `core/training.py` — Offline training pipeline
- `core/reward.py` — Composite reward function (Eq. 5)
- `core/dataset.py` — Dataset collection for offline training

**Files to modify:**
- `core/optimizer.py` — Use REINFORCE policy instead of Thompson Sampling
- `core/rl_policy.py` — Migrate to REINFORCE or keep as fallback
- `agent/state.py` — Add trajectory storage fields

**Implementation steps:**
1. Implement composite reward function (Eq. 5):
   ```
   R_final = w_perf * R_perf + w_cost * R_cost
   R_perf = +C_succ if success, -C_fail otherwise
   R_cost = -C_overflow if over budget, +g(1 - C_actual/B) if success
   ```
2. Implement REINFORCE policy:
   - Neural network: task features -> topology probabilities
   - Entropy regularization: -beta * H(pi)
   - Loss: -E[log pi(t|T,B) * R] - beta * H(pi)
3. Implement offline training pipeline:
   - Collect trajectories (task, budget, topology, result, cost)
   - Store in dataset D
   - Batch training on collected data
4. Replace Thompson Sampling with REINFORCE in optimizer
5. Add topology mapping:
   - Linear -> pipeline
   - Star -> fanout
   - Feedback -> new topology (generate-critique-revise)
   - Planner-Driven -> supervisor

**Acceptance criteria:**
- REINFORCE policy selects topologies based on task + budget
- Entropy regularization prevents premature convergence
- Training pipeline works on collected trajectories
- Reward function matches paper's Eq. 5

### Phase 3: Feedback Topology (Paper Component 3)
**Priority: HIGH | Effort: 2-3 days**

**Files to create:**
- `agent/topologies/feedback.py` — Feedback topology graph

**Implementation steps:**
1. Implement Feedback topology:
   - Generate -> Critique -> Revise loop
   - Critic is highest-weight LLM
   - Executor produces initial output
   - Critic audits and provides feedback
   - Executor revises based on feedback
   - Loop until acceptance or budget limit
2. Add to topology registry in `builder.py`
3. Add projection for mid-execution degradation

**Acceptance criteria:**
- Feedback topology runs generate-critique-revise loop
- Critic doesn't perform task-side computation
- Loop terminates on acceptance or budget limit
- Integrates with budget gate and degradation system

### Phase 4: Cost Model Accuracy
**Priority: HIGH | Effort: 1-2 days**

**Files to modify:**
- `core/llm.py` — Update cost estimation
- `core/config.py` — Add token pricing config

**Implementation steps:**
1. Implement paper's cost model (Eq. 1):
   ```
   c_i = T_in * P_in + T_out * P_out
   ```
2. Set T_in = 500 (paper's recommendation)
3. Sample T_out from training data (paper's method)
4. Use actual provider pricing (not rough heuristics)
5. Add token pricing to config:
   ```python
   tier_token_pricing = {
       "cheap": {"input": 0.0001, "output": 0.0004},
       "standard": {"input": 0.001, "output": 0.003},
       "frontier": {"input": 0.008, "output": 0.024},
   }
   ```

**Acceptance criteria:**
- Cost predictions match actual API costs within 20%
- T_in and T_out match paper's methodology
- Token pricing is configurable per provider

### Phase 5: Evaluation Pipeline
**Priority: MEDIUM | Effort: 3-5 days**

**Files to create:**
- `eval/` — Evaluation directory
- `eval/gsm8k.py` — GSM8K dataset loader
- `eval/mbpp.py` — MBPP dataset loader
- `eval/math.py` — MATH dataset loader
- `eval/benchmark.py` — Benchmark runner
- `eval/metrics.py` — Accuracy and cost metrics

**Implementation steps:**
1. Implement dataset loaders for GSM8K, MBPP, MATH
2. Implement benchmark runner:
   - Run tasks with different budget levels
   - Measure accuracy and cost
   - Track OOB (out-of-budget) tasks
3. Generate tables matching paper's Table 1, 2, 4
4. Add evaluation commands to CLI

**Acceptance criteria:**
- Can run evaluation on GSM8K, MBPP, MATH
- Outputs tables matching paper format
- Tracks accuracy, average cost, OOB count
- Budget levels match paper (500, 875, 1250, 1625, 2000)

### Phase 6: Budget Adherence Fix
**Priority: CRITICAL | Effort: 1-2 days**

**Files to modify:**
- `agent/nodes/judge.py` — Already fixed (dynamic threshold)
- `agent/nodes/executor.py` — Already fixed (loop budget check)
- `agent/nodes/budget_gate.py` — Tighten thresholds
- `core/budget.py` — Improve cost tracking

**Implementation steps:**
1. Tighten budget gate thresholds:
   - Circuit breaker: 105% -> 100%
   - Judge skip: 90% -> 80%
   - Executor loop: 85% -> 75%
2. Fix stale BudgetTracker issue:
   - Sync BudgetTracker more frequently
   - Use consumed_cost from state, not BudgetTracker object
3. Add pre-execution cost estimation:
   - Estimate total cost before starting
   - Warn if estimated cost > budget
4. Add budget reservation:
   - Reserve budget for judge before running executor
   - Prevent executor from consuming judge's budget

**Acceptance criteria:**
- OOB tasks < 5% (matching paper's Table 3)
- Budget gate fires before overrun
- Cost tracking is accurate and timely

---

## Implementation Order

```
Phase 6 (Budget Fix)     — 1-2 days  — Fix current overruns
Phase 4 (Cost Model)     — 1-2 days  — Accurate cost predictions
Phase 1 (ILP Solver)     — 2-3 days  — Paper's Component 1
Phase 3 (Feedback Topo)  — 2-3 days  — Paper's missing topology
Phase 2 (REINFORCE)      — 5-7 days  — Paper's Component 2
Phase 5 (Evaluation)     — 3-5 days  — Validate against paper
```

**Total estimated effort: 14-22 days**

---

## Success Metrics

| Metric | Target | Paper Baseline |
|--------|--------|----------------|
| OOB tasks (GSM8K) | < 5% | 0/1319 (0%) |
| OOB tasks (MBPP) | < 5% | 1-5/500 (0.2-1%) |
| Cost reduction vs AutoGen | > 50% | 62% (GSM8K) |
| Accuracy vs AutoGen | Within 5% | 95.3% vs 95.4% |
| ILP solver time | < 1s | Not reported |
| Topology selection time | < 2s | Not reported |

---

## Dependencies to Add

| Package | Purpose | Phase |
|---------|---------|-------|
| `pulp` | ILP solver | Phase 1 |
| `torch` or `jax` | Neural network for REINFORCE | Phase 2 |
| `datasets` | Load GSM8K, MBPP, MATH | Phase 5 |

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| ILP solver too slow | High | Use PuLP CBC (fast for small problems) |
| REINFORCE training unstable | Medium | Start with Thompson Sampling as fallback |
| Feedback topology loops forever | Medium | Add max iterations + budget limit |
| Cost model still inaccurate | Medium | Calibrate with real API calls |
| Evaluation takes too long | Low | Run on subset first, extrapolate |
