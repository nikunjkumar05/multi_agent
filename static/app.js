const form = document.getElementById("taskForm");
const resultSection = document.getElementById("resultSection");
const auditSection = document.getElementById("auditSection");
const eventLogSection = document.getElementById("eventLogSection");
const costSection = document.getElementById("costSection");
const statusBadge = document.getElementById("statusBadge");
const topologyUsed = document.getElementById("topologyUsed");
const resultOutput = document.getElementById("resultOutput");
const auditLog = document.getElementById("auditLog");
const eventList = document.getElementById("eventList");
const warningBanner = document.getElementById("warningBanner");
const warningText = document.getElementById("warningText");
const warningConfirm = document.getElementById("warningConfirm");
const warningCancel = document.getElementById("warningCancel");
const submitBtn = document.getElementById("submitBtn");
const historyList = document.getElementById("historyList");
const refreshHistoryBtn = document.getElementById("refreshHistory");
const topologyDiagram = document.getElementById("topologyDiagram");
const costSummary = document.getElementById("costSummary");

let currentWs = null;
let currentTaskId = null;
let collectedEvents = [];
let historyRefreshInterval = null;
let pendingTask = null;

// ── Form submission ──────────────────────────────────────────────────────────

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  await submitTask(false);
});

async function submitTask(skipEstimate) {
  const task = document.getElementById("task").value.trim();
  const budget = parseFloat(document.getElementById("budget").value);
  const topology = document.getElementById("topology").value;

  if (!task) return;

  warningBanner.classList.add("hidden");

  if (!skipEstimate) {
    submitBtn.disabled = true;
    submitBtn.textContent = "Estimating...";
  }

  const body = { task, budget_usd: budget };
  if (topology) body.topology = topology;

  try {
    const res = await fetch("/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }

    // Check risk level
    if (data.risk_level === "HIGH" && !skipEstimate) {
      pendingTask = { data };
      showWarning(data);
      submitBtn.disabled = false;
      submitBtn.textContent = "Run Task";
      return;
    }

    startTask(data.task_id);
  } catch (err) {
    statusBadge.textContent = "FAILED";
    statusBadge.className = "badge failed";
    resultOutput.textContent = `Error: ${err.message}`;
    submitBtn.disabled = false;
    submitBtn.textContent = "Run Task";
  }
}

function showWarning(data) {
  const budget = parseFloat(document.getElementById("budget").value);
  const est = data.estimated_cost || 0;
  const ratio = budget > 0 ? (est / budget).toFixed(1) : "∞";

  warningBanner.classList.remove("hidden");
  warningText.innerHTML = `
    Estimated cost: <strong>$${est.toFixed(4)}</strong> vs budget: <strong>$${budget.toFixed(2)}</strong>
    <span class="warning-ratio">(${ratio}x budget)</span>
    <br>Topology: ${data.topology} — this task will likely exceed your budget.
  `;
}

warningConfirm.addEventListener("click", async () => {
  warningBanner.classList.add("hidden");
  await submitTask(true);
});

warningCancel.addEventListener("click", () => {
  warningBanner.classList.add("hidden");
  pendingTask = null;
  submitBtn.disabled = false;
  submitBtn.textContent = "Run Task";
});

function startTask(taskId) {
  currentTaskId = taskId;

  resultSection.classList.remove("hidden");
  auditSection.classList.add("hidden");
  eventLogSection.classList.remove("hidden");
  costSection.classList.add("hidden");
  statusBadge.textContent = "RUNNING";
  statusBadge.className = "badge running";
  topologyUsed.textContent = "";
  topologyDiagram.classList.add("hidden");
  resultOutput.innerHTML = "Executing task...";
  eventList.innerHTML = "";
  collectedEvents = [];

  submitBtn.disabled = false;
  submitBtn.textContent = "Run Task";

  connectWebSocket(taskId);
  pollTask(taskId);
  startHistoryRefresh();
}

// ── WebSocket ────────────────────────────────────────────────────────────────

function connectWebSocket(taskId) {
  if (currentWs) {
    currentWs.close();
  }

  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${location.host}/ws/${taskId}`;
  currentWs = new WebSocket(wsUrl);

  currentWs.onmessage = function (e) {
    const event = JSON.parse(e.data);
    appendEvent(event);
    collectedEvents.push(event);
  };

  currentWs.onclose = function () {
    appendEvent({
      event_type: "connection_closed",
      timestamp: new Date().toISOString(),
      data: {},
    });
  };

  currentWs.onerror = function () {
    appendEvent({
      event_type: "connection_error",
      timestamp: new Date().toISOString(),
      data: {},
    });
  };
}

// ── Event log ────────────────────────────────────────────────────────────────

function appendEvent(event) {
  const el = document.createElement("div");
  el.className = "event-entry";

  const time = new Date(event.timestamp).toLocaleTimeString();
  const typeColors = {
    topology_selected: "#00d4ff",
    topology_degraded: "#ff6b6b",
    planner_started: "#a78bfa",
    planner_completed: "#a78bfa",
    step_started: "#34d399",
    step_completed: "#34d399",
    validation_completed: "#fbbf24",
    escalation_check: "#f87171",
    budget_band_crossed: "#fb923c",
    budget_gate_pause: "#fb923c",
    budget_gate_emergency: "#ef4444",
    budget_gate_skip_judge: "#fbbf24",
    agent_completed: "#34d399",
    supervisor_decided: "#a78bfa",
    task_completed: "#22c55e",
    task_failed: "#ef4444",
    connection_closed: "#888",
    connection_error: "#ef4444",
    tool_call: "#60a5fa",
    tool_result: "#34d399",
    judge_completed: "#fbbf24",
  };

  const color = typeColors[event.event_type] || "#ccc";
  const formatted = formatEvent(event);

  const timeSpan = document.createElement("span");
  timeSpan.className = "event-time";
  timeSpan.textContent = time;

  const typeSpan = document.createElement("span");
  typeSpan.className = "event-type";
  typeSpan.style.color = color;
  typeSpan.textContent = event.event_type;

  const dataSpan = document.createElement("span");
  dataSpan.className = "event-data";
  dataSpan.textContent = formatted;

  el.appendChild(timeSpan);
  el.appendChild(typeSpan);
  el.appendChild(dataSpan);

  eventList.appendChild(el);
  eventList.scrollTop = eventList.scrollHeight;

  // Update topology diagram on relevant events
  if (event.event_type === "topology_selected") {
    renderTopology(event.data.topology);
  } else if (event.event_type === "topology_degraded") {
    renderTopology(event.data.to_topology, event.data.from_topology);
  }
}

function formatEvent(event) {
  const d = event.data || {};
  const fmtTokens = (n) => n != null ? `${n.toLocaleString()} tokens` : "";
  const fmtCost = (c) => c != null ? `$${c.toFixed(4)}` : "";
  const fmtBudget = (p) => p != null ? `${p}% budget` : "";

  switch (event.event_type) {
    case "ping":
      return ``;
    case "topology_selected":
      return `→ ${d.topology} (${d.rationale || ""})`;

    case "topology_degraded":
      return `${d.from_topology} → ${d.to_topology} (${d.reason || "budget"})`;

    case "planner_started":
      return `Planning...`;

    case "planner_completed":
      return `Planned ${d.step_count} steps [${fmtTokens(d.tokens_used)}, ${fmtCost(d.cost_usd)}, ${fmtBudget(d.budget_spent_pct)}]`;

    case "step_started":
      return `Step ${d.step_id}: ${(d.description || "").substring(0, 60)}${d.description && d.description.length > 60 ? "..." : ""}`;

    case "step_completed": {
      let msg = `Step ${d.step_id} done`;
      if (d.worker) msg += ` [${d.worker}]`;
      const parts = [];
      if (d.tokens_used) parts.push(fmtTokens(d.tokens_used));
      if (d.cost_usd) parts.push(fmtCost(d.cost_usd));
      if (d.budget_spent_pct !== undefined) parts.push(fmtBudget(d.budget_spent_pct));
      if (parts.length) msg += ` (${parts.join(", ")})`;
      if (d.result_preview) msg += ` — ${d.result_preview.substring(0, 80)}...`;
      return msg;
    }

    case "validation_completed": {
      const conf = Math.round((d.confidence || 0) * 100);
      let msg = `Validation: ${conf}% confidence`;
      if (d.diverged) msg += " (diverged)";
      const parts = [];
      if (d.tokens_used) parts.push(fmtTokens(d.tokens_used));
      if (d.cost_usd) parts.push(fmtCost(d.cost_usd));
      if (d.budget_spent_pct !== undefined) parts.push(fmtBudget(d.budget_spent_pct));
      if (parts.length) msg += ` [${parts.join(", ")}]`;
      return msg;
    }

    case "escalation_check":
      return `Escalation: ${Math.round((d.confidence || 0) * 100)} confidence, escalated=${d.escalated}`;

    case "budget_band_crossed":
      return `Budget: ${d.from_band} → ${d.to_band} (${d.spent_pct}% spent)`;

    case "budget_gate_pause":
      return `Pause: ${d.from_topology} → ${d.to_topology} (${d.band}, ${d.spent_pct}% spent, ${fmtTokens(d.consumed_tokens)})`;

    case "budget_gate_emergency":
      return `Emergency: ${d.from_topology} → ${d.to_topology} (${d.band}, ${d.spent_pct}% spent, ${fmtTokens(d.consumed_tokens)})`;

    case "budget_gate_skip_judge":
      return `Skip judge on ${d.topology} (${d.band}, ${d.spent_pct}% spent)`;

    case "agent_completed":
      return `Agent ${d.agent_key} (${d.role}) done [${fmtTokens(d.tokens_used)}, ${fmtCost(d.cost_usd)}, ${fmtBudget(d.budget_spent_pct)}]`;

    case "supervisor_decided":
      return `Supervisor → step ${d.next_step_id} [${fmtTokens(d.tokens_used)}, ${fmtCost(d.cost_usd)}, ${fmtBudget(d.budget_spent_pct)}]`;

    case "judge_completed":
      return `Judge done [${fmtTokens(d.tokens_used)}, ${fmtCost(d.cost_usd)}, ${fmtBudget(d.budget_spent_pct)}]`;

    case "task_completed": {
      const parts = [];
      if (d.tokens_used) parts.push(fmtTokens(d.tokens_used));
      if (d.cost_usd) parts.push(fmtCost(d.cost_usd));
      if (d.budget_spent_pct !== undefined) parts.push(fmtBudget(d.budget_spent_pct));
      if (d.degradation_count > 0) parts.push(`${d.degradation_count} degradations`);
      return `Complete (${d.status}, ${parts.join(", ")})`;
    }

    case "task_failed":
      return `Failed`;

    case "connection_closed":
      return "Connection closed";

    case "connection_error":
      return "Connection error";

    case "tool_call":
      return `Tool: ${d.tool}`;

    case "tool_result":
      return `Tool ${d.tool}: ${d.success ? "ok" : "failed"}`;

    default:
      return Object.entries(d).map(([k, v]) => `${k}: ${v}`).join(", ");
  }
}

// ── Task polling ─────────────────────────────────────────────────────────────

async function pollTask(taskId) {
  const maxAttempts = 180;
  for (let i = 0; i < maxAttempts; i++) {
    await sleep(1000);

    try {
      const res = await fetch(`/tasks/${taskId}`);
      const task = await res.json();

      if (task.topology && task.topology !== "pending") {
        topologyUsed.textContent = `topology: ${task.topology}`;
      }

      if (task.status === "completed") {
        statusBadge.textContent = "COMPLETE";
        statusBadge.className = "badge complete";
        const raw = task.final_result || "No result produced";
        resultOutput.innerHTML = DOMPurify.sanitize(marked.parse(raw));
        loadAudit(taskId);
        renderCostSummary(collectedEvents);
        submitBtn.disabled = false;
        submitBtn.textContent = "Run Task";
        stopHistoryRefresh();
        loadHistory();
        if (currentWs) currentWs.close();
        return;
      }

      if (task.status === "failed") {
        statusBadge.textContent = "FAILED";
        statusBadge.className = "badge failed";
        const logs = task.logs || [];
        resultOutput.textContent = logs[logs.length - 1] || "Unknown error";
        submitBtn.disabled = false;
        submitBtn.textContent = "Run Task";
        stopHistoryRefresh();
        loadHistory();
        if (currentWs) currentWs.close();
        return;
      }

      resultOutput.textContent = `Status: ${task.status}...`;
    } catch {
      // keep polling
    }
  }

  statusBadge.textContent = "TIMEOUT";
  statusBadge.className = "badge failed";
  resultOutput.textContent = "Task timed out after 3 minutes";
  submitBtn.disabled = false;
  submitBtn.textContent = "Run Task";
  stopHistoryRefresh();
  if (currentWs) currentWs.close();
}

// ── Audit trail ──────────────────────────────────────────────────────────────

async function loadAudit(taskId) {
  try {
    const res = await fetch(`/audit/${taskId}`);
    const data = await res.json();
    const events = data.events ?? [];

    if (events.length === 0) {
      auditSection.classList.add("hidden");
      return;
    }

    auditSection.classList.remove("hidden");
    auditLog.innerHTML = "";
    events.forEach((ev) => {
      const entry = document.createElement("div");
      entry.className = "audit-entry";

      const typeSpan = document.createElement("span");
      typeSpan.className = "event-type";
      typeSpan.textContent = ev.event_type;
      entry.appendChild(typeSpan);

      if (ev.detail?.topology) {
        const topoSpan = document.createElement("span");
        topoSpan.textContent = ` topology: ${ev.detail.topology}`;
        entry.appendChild(topoSpan);
      }
      if (ev.detail?.band) {
        const bandSpan = document.createElement("span");
        bandSpan.textContent = ` band: ${ev.detail.band}`;
        entry.appendChild(bandSpan);
      }

      const tsSpan = document.createElement("span");
      tsSpan.className = "timestamp";
      tsSpan.textContent = ev.timestamp;
      entry.appendChild(tsSpan);

      auditLog.appendChild(entry);
    });
  } catch {
    auditSection.classList.add("hidden");
  }
}

// ── Task history ─────────────────────────────────────────────────────────────

async function loadHistory() {
  try {
    const res = await fetch("/tasks?limit=50");
    const tasks = await res.json();

    if (tasks.length === 0) {
      historyList.innerHTML = '<p class="muted">No tasks yet</p>';
      return;
    }

    historyList.innerHTML = "";
    tasks.forEach((task) => {
      const item = document.createElement("div");
      item.className = "history-item";
      if (task.task_id === currentTaskId) {
        item.classList.add("active");
      }

      const statusClass = {
        running: "running",
        pending: "running",
        completed: "complete",
        failed: "failed",
      }[task.status] || "running";

      const shortId = task.task_id.substring(0, 8);
      const budgetPct = task.budget_spent_pct != null ? `${task.budget_spent_pct.toFixed(1)}%` : "—";
      const topology = task.topology || "—";

      item.innerHTML = `
        <div class="history-item-header">
          <span class="badge small ${statusClass}">${task.status}</span>
          <span class="history-id">${shortId}</span>
        </div>
        <div class="history-item-details">
          <span class="muted">${topology}</span>
          <span class="muted">${budgetPct} budget</span>
        </div>
      `;

      item.addEventListener("click", () => viewTask(task));
      historyList.appendChild(item);
    });
  } catch {
    historyList.innerHTML = '<p class="muted">Failed to load history</p>';
  }
}

function viewTask(task) {
  currentTaskId = task.task_id;
  resultSection.classList.remove("hidden");
  eventLogSection.classList.add("hidden");
  costSection.classList.add("hidden");
  auditSection.classList.add("hidden");

  const statusClass = {
    running: "running",
    pending: "running",
    completed: "complete",
    failed: "failed",
  }[task.status] || "running";

  statusBadge.textContent = task.status.toUpperCase();
  statusBadge.className = `badge ${statusClass}`;
  topologyUsed.textContent = task.topology ? `topology: ${task.topology}` : "";

  if (task.status === "completed" && task.final_result) {
    resultOutput.innerHTML = DOMPurify.sanitize(marked.parse(task.final_result));
    loadAudit(task.task_id);
  } else if (task.status === "failed") {
    const logs = task.logs || [];
    resultOutput.textContent = logs[logs.length - 1] || "Task failed";
  } else {
    resultOutput.textContent = `Status: ${task.status}`;
  }

  // Highlight in history
  document.querySelectorAll(".history-item").forEach((el) => el.classList.remove("active"));
  const items = historyList.querySelectorAll(".history-item");
  items.forEach((el) => {
    if (el.querySelector(".history-id")?.textContent === task.task_id.substring(0, 8)) {
      el.classList.add("active");
    }
  });
}

function startHistoryRefresh() {
  stopHistoryRefresh();
  historyRefreshInterval = setInterval(loadHistory, 5000);
}

function stopHistoryRefresh() {
  if (historyRefreshInterval) {
    clearInterval(historyRefreshInterval);
    historyRefreshInterval = null;
  }
}

refreshHistoryBtn.addEventListener("click", loadHistory);

// ── Topology visualization ───────────────────────────────────────────────────

function renderTopology(topology, degradedFrom) {
  topologyDiagram.classList.remove("hidden");

  const diagrams = {
    single: `
      <div class="topo-row">
        <div class="topo-box topo-executor">Executor</div>
      </div>
    `,
    pipeline: `
      <div class="topo-row">
        <div class="topo-box topo-planner">Planner</div>
        <div class="topo-arrow">→</div>
        <div class="topo-box topo-executor">Executor</div>
        <div class="topo-arrow">→</div>
        <div class="topo-box topo-validator">Validator</div>
        <div class="topo-arrow">→</div>
        <div class="topo-box topo-judge">Judge</div>
      </div>
    `,
    supervisor: `
      <div class="topo-row topo-supervisor-layout">
        <div class="topo-box topo-supervisor">Supervisor</div>
        <div class="topo-spokes">
          <div class="topo-spoke">
            <div class="topo-arrow-v">↕</div>
            <div class="topo-box topo-worker">Worker 1</div>
          </div>
          <div class="topo-spoke">
            <div class="topo-arrow-v">↕</div>
            <div class="topo-box topo-worker">Worker 2</div>
          </div>
          <div class="topo-spoke">
            <div class="topo-arrow-v">↕</div>
            <div class="topo-box topo-worker">Worker N</div>
          </div>
        </div>
      </div>
    `,
    fanout: `
      <div class="topo-row topo-fanout-layout">
        <div class="topo-box topo-fanout">Fan-out</div>
        <div class="topo-arrow">→</div>
        <div class="topo-parallel">
          <div class="topo-box topo-worker">Worker 1</div>
          <div class="topo-box topo-worker">Worker 2</div>
          <div class="topo-box topo-worker">Worker N</div>
        </div>
        <div class="topo-arrow">→</div>
        <div class="topo-box topo-aggregator">Aggregator</div>
      </div>
    `,
    ensemble: `
      <div class="topo-row topo-ensemble-layout">
        <div class="topo-parallel">
          <div class="topo-box topo-agent">Agent A</div>
          <div class="topo-box topo-agent">Agent B</div>
          <div class="topo-box topo-agent">Agent C</div>
        </div>
        <div class="topo-arrow">→</div>
        <div class="topo-box topo-judge">Judge</div>
      </div>
    `,
  };

  let html = `<div class="topo-label">${topology}</div>`;
  if (degradedFrom) {
    html = `<div class="topo-label topo-degraded">${degradedFrom} → ${topology}</div>`;
  }
  html += diagrams[topology] || diagrams.single;

  topologyDiagram.innerHTML = html;
}

// ── Cost breakdown ───────────────────────────────────────────────────────────

function renderCostSummary(events) {
  let totalTokens = 0;
  let totalCost = 0;
  let budgetPct = 0;
  const steps = [];

  for (const event of events) {
    const d = event.data || {};

    if (d.tokens_used) totalTokens = d.tokens_used;
    if (d.cost_usd) totalCost = d.cost_usd;
    if (d.budget_spent_pct !== undefined) budgetPct = d.budget_spent_pct;

    if (event.event_type === "step_completed" && d.step_id) {
      steps.push({
        label: `Step ${d.step_id}`,
        tokens: d.tokens_used || 0,
        cost: d.cost_usd || 0,
      });
    }

    if (event.event_type === "planner_completed") {
      steps.unshift({
        label: "Planning",
        tokens: d.tokens_used || 0,
        cost: d.cost_usd || 0,
      });
    }

    if (event.event_type === "validation_completed") {
      steps.push({
        label: "Validation",
        tokens: d.tokens_used || 0,
        cost: d.cost_usd || 0,
      });
    }

    if (event.event_type === "judge_completed") {
      steps.push({
        label: "Judge",
        tokens: d.tokens_used || 0,
        cost: d.cost_usd || 0,
      });
    }

    if (event.event_type === "agent_completed" && d.agent_key) {
      steps.push({
        label: `Agent ${d.agent_key}`,
        tokens: d.tokens_used || 0,
        cost: d.cost_usd || 0,
      });
    }
  }

  if (totalTokens === 0 && totalCost === 0) {
    costSection.classList.add("hidden");
    return;
  }

  costSection.classList.remove("hidden");

  const fmtTokens = (n) => n.toLocaleString();
  const fmtCost = (c) => `$${c.toFixed(4)}`;

  let html = `
    <div class="cost-totals">
      <div class="cost-stat">
        <span class="cost-stat-value">${fmtTokens(totalTokens)}</span>
        <span class="cost-stat-label">Total Tokens</span>
      </div>
      <div class="cost-stat">
        <span class="cost-stat-value">${fmtCost(totalCost)}</span>
        <span class="cost-stat-label">Total Cost</span>
      </div>
      <div class="cost-stat">
        <span class="cost-stat-value">${budgetPct.toFixed(1)}%</span>
        <span class="cost-stat-label">Budget Used</span>
      </div>
    </div>
  `;

  if (steps.length > 0) {
    const maxCost = Math.max(...steps.map((s) => s.cost), 0.0001);
    html += `<div class="cost-steps">`;
    for (const step of steps) {
      const pct = (step.cost / maxCost) * 100;
      html += `
        <div class="cost-step">
          <div class="cost-step-label">${step.label}</div>
          <div class="cost-step-bar-track">
            <div class="cost-step-bar" style="width: ${pct}%"></div>
          </div>
          <div class="cost-step-values">${fmtTokens(step.tokens)} tok · ${fmtCost(step.cost)}</div>
        </div>
      `;
    }
    html += `</div>`;
  }

  costSummary.innerHTML = html;
}

// ── Utility ──────────────────────────────────────────────────────────────────

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// ── Init ─────────────────────────────────────────────────────────────────────

loadHistory();
