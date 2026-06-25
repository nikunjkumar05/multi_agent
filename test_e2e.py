import httpx, time

r = httpx.post("http://localhost:8000/execute", json={"task": "Write a Python function to compute fibonacci numbers", "budget_usd": 0.50})
task_id = r.json()["task_id"]
print(f"Task: {task_id}")

for i in range(30):
    time.sleep(3)
    r = httpx.get(f"http://localhost:8000/tasks/{task_id}")
    data = r.json()
    status = data["status"]
    topology = data["topology"]
    print(f"[{i*3}s] status={status} topology={topology}")
    if status in ("completed", "failed"):
        print()
        print("=== RESULT ===")
        print(data.get("final_result") or "(none)")
        print()
        print("=== LOGS ===")
        for log in data.get("logs", []):
            print(f"  {log}")
        break
