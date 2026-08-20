import pytest
from fastapi.testclient import TestClient
from middleware.api.main import app

client = TestClient(app)

def test_health_check():
    """Test that the API is up and running."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_create_and_get_task():
    """Test the full lifecycle of creating a task and fetching its status."""
    
    # 1. Create a task (this uses the 'create_task' endpoint)
    payload = {
        "task_type": "code_generation",
        "prompt": "Write a python function to add two numbers",
        "budget_usd": 0.50,
        "timeout_seconds": 60
    }
    
    response = client.post("/api/v1/tasks/", json=payload)
    
    # Ensure it was accepted and queued
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "queued"
    
    # Extract the task ID
    task_id = data["task_id"]
    
    # 2. Get the task status
    get_response = client.get(f"/api/v1/tasks/{task_id}")
    assert get_response.status_code == 200
    
    get_data = get_response.json()
    assert get_data["task_id"] == task_id
    
    # Because the background task might have executed instantly in the TestClient, 
    # the status could be 'in_progress', 'completed', or 'failed', 
    # but it definitely shouldn't be a 404 error.
    assert get_data["status"] in ["queued", "in_progress", "completed", "failed"]

def test_get_nonexistent_task():
    """Test that requesting an invalid task ID returns a 404."""
    response = client.get("/api/v1/tasks/task_invalid999")
    assert response.status_code == 404

def test_cancel_task():
    """Test cancelling a task."""
    
    # Create a task first
    payload = {
        "task_type": "debugging",
        "prompt": "Fix this error: IndexError",
        "budget_usd": 0.20
    }
    response = client.post("/api/v1/tasks/", json=payload)
    task_id = response.json()["task_id"]
    
    # Cancel it
    cancel_response = client.delete(f"/api/v1/tasks/{task_id}")
    assert cancel_response.status_code == 200
    
    # Verify it is cancelled
    get_response = client.get(f"/api/v1/tasks/{task_id}")
    assert get_response.json()["status"] == "cancelled"
