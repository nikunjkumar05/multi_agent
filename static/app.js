const form = document.getElementById("taskForm");
const resultSection = document.getElementById("resultSection");
const auditSection = document.getElementById("auditSection");
const statusBadge = document.getElementById("statusBadge");
const topologyUsed = document.getElementById("topologyUsed");
const resultOutput = document.getElementById("resultOutput");
const auditLog = document.getElementById("auditLog");
const submitBtn = document.getElementById("submitBtn");

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
  statusBadge.textContent = "RUNNING";
  statusBadge.className = "badge running";
  topologyUsed.textContent = "";
  resultOutput.textContent = "Executing task...";

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

    pollTask(taskId);
  } catch (err) {
    statusBadge.textContent = "FAILED";
    statusBadge.className = "badge failed";
    resultOutput.textContent = `Error: ${err.message}`;
    submitBtn.disabled = false;
    submitBtn.textContent = "Run Task";
  }
});

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
        return;
      }

      if (task.status === "failed") {
        statusBadge.textContent = "FAILED";
        statusBadge.className = "badge failed";
        const logs = task.logs || [];
        resultOutput.textContent = logs[logs.length - 1] || "Unknown error";
        submitBtn.disabled = false;
        submitBtn.textContent = "Run Task";
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
    auditLog.innerHTML = events
      .map(
        (ev) => `
        <div class="audit-entry">
          <span class="event-type">${ev.event_type}</span>
          ${ev.topology ? `topology: ${ev.topology}` : ""}
          ${ev.band ? `band: ${ev.band}` : ""}
          ${ev.degradation ? `degraded: ${ev.degradation.original} → ${ev.degradation.collapsed_to}` : ""}
          <span class="timestamp">${ev.timestamp}</span>
        </div>`
      )
      .join("");
  } catch {
    auditSection.classList.add("hidden");
  }
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}
