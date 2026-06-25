import httpx
import time
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "http://localhost:8000"

TASKS = {
    "simple": "What is 2+2?",
    "code": "Write a Python function to compute fibonacci numbers",
    "research": "Explain the difference between TCP and UDP protocols",
    "complex": "Write a web scraper for news headlines with error handling and retry logic",
}

TOPOLOGIES = ["single", "pipeline", "supervisor"]
BUDGETS = [0.05, 0.10, 0.50]


def submit_and_wait(task: str, budget: float, topology: str | None = None, timeout: int = 90) -> dict:
    body = {"task": task, "budget_usd": budget}
    if topology:
        body["topology"] = topology

    r = httpx.post(f"{BASE}/execute", json=body, timeout=10)
    task_id = r.json()["task_id"]

    for _ in range(timeout // 3):
        time.sleep(3)
        r = httpx.get(f"{BASE}/tasks/{task_id}", timeout=5)
        data = r.json()
        if data["status"] in ("completed", "failed"):
            return data

    return {"task_id": task_id, "status": "timeout", "topology": topology or "auto"}


def test_topology_sweep():
    print("\n=== TOPOLOGY SWEEP ===")
    print(f"{'Topology':<12} {'Task':<15} {'Status':<10} {'Steps':<6}")
    print("-" * 50)

    results = []
    for topo in TOPOLOGIES:
        data = submit_and_wait(TASKS["code"], 0.50, topology=topo, timeout=120)
        logs = data.get("logs", [])
        step_count = sum(1 for l in logs if l.startswith("Completed step"))
        status = data["status"]
        print(f"{topo:<12} {'fibonacci':<15} {status:<10} {step_count:<6}")
        results.append({"topology": topo, "status": status, "steps": step_count, "topology_used": data.get("topology")})
    return results


def test_budget_sweep():
    print("\n=== BUDGET SWEEP ===")
    print(f"{'Budget':<10} {'Status':<10} {'Spent%':<10}")
    print("-" * 35)

    results = []
    for budget in BUDGETS:
        data = submit_and_wait(TASKS["simple"], budget, timeout=90)
        spent = data.get("budget_spent_pct", 0)
        print(f"${budget:<9} {data['status']:<10} {spent:<10.1f}")
        results.append({"budget": budget, "status": data["status"], "spent_pct": spent})
    return results


def test_task_complexity():
    print("\n=== TASK COMPLEXITY ===")
    print(f"{'Task':<12} {'Status':<10} {'Topology':<12} {'Steps':<6}")
    print("-" * 45)

    results = []
    for name, task in TASKS.items():
        data = submit_and_wait(task, 0.50, timeout=120)
        logs = data.get("logs", [])
        step_count = sum(1 for l in logs if l.startswith("Completed step"))
        print(f"{name:<12} {data['status']:<10} {data.get('topology', '?'):<12} {step_count:<6}")
        results.append({"task": name, "status": data["status"], "topology": data.get("topology"), "steps": step_count})
    return results


def test_concurrent():
    print("\n=== CONCURRENT (5 tasks) ===")
    print(f"{'Task':<12} {'Status':<10} {'Topology':<12}")
    print("-" * 38)

    results = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(submit_and_wait, task, 0.30, None, 120): name
            for name, task in TASKS.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            data = future.result()
            print(f"{name:<12} {data['status']:<10} {data.get('topology', '?'):<12}")
            results.append({"task": name, "status": data["status"], "topology": data.get("topology")})
    return results


def test_health():
    try:
        r = httpx.get(f"{BASE}/health", timeout=5)
        return r.json().get("status") == "ok"
    except Exception:
        return False


if __name__ == "__main__":
    if not test_health():
        print("ERROR: Server not running at", BASE)
        print("Start with: uvicorn api.main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    print(f"BAMAS Stress Test — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Server: {BASE}")

    all_results = {}

    all_results["topology_sweep"] = test_topology_sweep()
    all_results["budget_sweep"] = test_budget_sweep()
    all_results["task_complexity"] = test_task_complexity()
    all_results["concurrent"] = test_concurrent()

    print("\n=== SUMMARY ===")
    total = sum(len(v) for v in all_results.values())
    passed = sum(1 for v in all_results.values() for r in v if r["status"] == "completed")
    failed = sum(1 for v in all_results.values() for r in v if r["status"] == "failed")
    timeout = sum(1 for v in all_results.values() for r in v if r["status"] == "timeout")
    print(f"Total: {total} | Passed: {passed} | Failed: {failed} | Timeout: {timeout}")

    with open("tests/stress_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("Results saved to tests/stress_results.json")
