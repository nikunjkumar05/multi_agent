const form = document.getElementById("taskForm");
const resultSection = document.getElementById("resultSection");
const auditSection = document.getElementById("auditSection");
const eventLogSection = document.getElementById("eventLogSection");
const statusBadge = document.getElementById("statusBadge");
const topologyUsed = document.getElementById("topologyUsed");
const resultOutput = document.getElementById("resultOutput");
const auditLog = document.getElementById("auditLog");
const eventList = document.getElementById("eventList");
const submitBtn = document.getElementById("submitBtn");

let currentWs = null;

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const task = document.getElementById("task").value.trim();
  const budget = parseFloat(document.getElementById("budget").value);
  const topology = document.getElementById("topology").value;

  if (!task) return;

  submitBtn.disabled = true;
  submitBtn.textContent = "Running...";
  resultSection.classList.remove("hidden");
  auditSection.classList.add("hidden");
  eventLogSection.classList.remove("hidden");
  statusBadge.textContent = "RUNNING";
  statusBadge.className = "badge running";
  topologyUsed.textContent = "";
  resultOutput.textContent = "Executing task...";
  eventList.innerHTML = "";

  const body = { task, budget_usd: budget };
  if (topology) body.topology = topology;

  try {
    const res = await fetch("/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    const taskId = data.task_id;

    connectWebSocket(taskId);
    pollTask(taskId);
  } catch (err) {
    statusBadge.textContent = "FAILED";
    statusBadge.className = "badge failed";
    resultOutput.textContent = `Error: ${err.message}`;
    submitBtn.disabled = false;
    submitBtn.textContent = "Run Task";
  }
});

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

function appendEvent(event) {
  const el = document.createElement("div");
  el.className = "event-entry";

  const time = new Date(event.timestamp).toLocaleTimeString();
  const typeColors = {
    topology_selected: "#00d4ff",
    planner_started: "#a78bfa",
    planner_completed: "#a78bfa",
    step_started: "#34d399",
    step_completed: "#34d399",
    validation_completed: "#fbbf24",
    escalation_check: "#f87171",
    budget_band_crossed: "#fb923c",
    task_completed: "#22c55e",
    task_failed: "#ef4444",
    connection_closed: "#888",
    connection_error: "#ef4444",
    tool_call: "#60a5fa",
    tool_result: "#34d399",
    judge_completed: "#fbbf24",
  };

  const color = typeColors[event.event_type] || "#ccc";
  const dataStr = Object.keys(event.data || {}).length > 0
    ? ` — ${JSON.stringify(event.data)}`
    : "";

  const timeSpan = document.createElement("span");
  timeSpan.className = "event-time";
  timeSpan.textContent = time;

  const typeSpan = document.createElement("span");
  typeSpan.className = "event-type";
  typeSpan.style.color = color;
  typeSpan.textContent = event.event_type;

  const dataSpan = document.createElement("span");
  dataSpan.className = "event-data";
  dataSpan.textContent = dataStr;

  el.appendChild(timeSpan);
  el.appendChild(typeSpan);
  el.appendChild(dataSpan);

  eventList.appendChild(el);
  eventList.scrollTop = eventList.scrollHeight;
}

async function pollTask(taskId) {
  const maxAttempts = 60;
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
        resultOutput.textContent = task.final_result || "No result produced";
        loadAudit(taskId);
        submitBtn.disabled = false;
        submitBtn.textContent = "Run Task";
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
  resultOutput.textContent = "Task timed out after 60 seconds";
  submitBtn.disabled = false;
  submitBtn.textContent = "Run Task";
  if (currentWs) currentWs.close();
}

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

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}
