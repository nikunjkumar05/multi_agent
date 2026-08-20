import uuid
import time
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from middleware.models.schemas import TaskCreate, TaskResponse, TaskStatus
from middleware.classifier.task_classifier import classify_task
from middleware.registry.agent_registry import AgentRegistry
from middleware.adapters.base import AgentTask
from middleware.adapters.opencode import OpenCodeAdapter
from middleware.adapters.aider import AiderAdapter
import logging

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])

# In a real app, this registry would be instantiated globally in main.py
# and passed down via dependency injection.
registry = AgentRegistry()
registry.register("opencode", OpenCodeAdapter())
registry.register("aider", AiderAdapter())

# In-memory store for tasks (replace with database in production)
tasks_db = {}

def execute_task_background(task_id: str, agent_id: str, task_data: AgentTask):
    """Background worker to execute the task using the selected adapter."""
    adapter = registry.get(agent_id)
    if not adapter:
        tasks_db[task_id]["status"] = TaskStatus.FAILED
        return

    tasks_db[task_id]["status"] = TaskStatus.IN_PROGRESS
    
    # Run the adapter synchronously or await if we had an async loop context.
    # Note: in FastAPI background tasks, if we define it as async, FastAPI will await it.
    pass

async def async_execute_task_background(task_id: str, agent_id: str, task_data: AgentTask):
    """Async background worker."""
    adapter = registry.get(agent_id)
    if not adapter:
        tasks_db[task_id]["status"] = TaskStatus.FAILED
        return

    tasks_db[task_id]["status"] = TaskStatus.IN_PROGRESS
    
    try:
        # Execute the task
        result = await adapter.execute(task_data)
        
        # Update the task status in our DB
        task_info = tasks_db[task_id]
        if result.success:
            task_info["status"] = TaskStatus.COMPLETED
            task_info["output"] = result.output
            task_info["cost_usd"] = result.cost_usd
            task_info["tokens_used"] = result.tokens_used
            task_info["latency_ms"] = result.latency_ms
            task_info["budget_remaining_usd"] = task_data.budget_usd - result.cost_usd
            registry.record_success(agent_id, result.cost_usd, result.latency_ms)
        else:
            task_info["status"] = TaskStatus.FAILED
            task_info["output"] = result.error
            registry.record_failure(agent_id)
            
    except Exception as e:
        log.error(f"Task execution failed: {e}")
        tasks_db[task_id]["status"] = TaskStatus.FAILED
        tasks_db[task_id]["output"] = str(e)
        registry.record_failure(agent_id)

@router.post("/", response_model=TaskResponse)
async def create_task(task_in: TaskCreate, background_tasks: BackgroundTasks, request: Request):
    """Create a new task, select the optimal agent, and queue it for execution."""
    
    # 1. Classify the task (override user input if needed, or validate)
    classification = classify_task(task_in.prompt, task_in.context)
    task_type = classification.task_type.value
    
    # 2. Select the optimal agent via the registry (acts as a proxy for the ILP solver for now)
    selected = registry.get_cheapest(task_type)
    if not selected:
        raise HTTPException(status_code=400, detail=f"No capable agents found for task type: {task_type}")
    
    agent_id, adapter = selected
    
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    
    # 3. Create the standard AgentTask
    agent_task = AgentTask(
        task_id=task_id,
        task_type=task_type,
        prompt=task_in.prompt,
        context=task_in.context,
        budget_usd=task_in.budget_usd,
        timeout_seconds=task_in.timeout_seconds,
        preferred_agents=task_in.preferred_agents or []
    )
    
    # 4. Estimate cost
    estimated_cost = adapter.estimate_cost(agent_task)
    
    # 5. Save to DB
    tasks_db[task_id] = {
        "task_id": task_id,
        "status": TaskStatus.QUEUED,
        "estimated_cost_usd": estimated_cost,
        "estimated_tokens": 1000, # Mocked
        "selected_agent": agent_id,
        "selected_tier": "standard",
        "ws_url": f"ws://{request.headers.get('host')}/ws/tasks/{task_id}"
    }
    
    # 6. Queue execution
    background_tasks.add_task(async_execute_task_background, task_id, agent_id, agent_task)
    
    return TaskResponse(**tasks_db[task_id])

@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """Get the current status and results of a task."""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskResponse(**tasks_db[task_id])

@router.delete("/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a queued or in-progress task."""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
        
    tasks_db[task_id]["status"] = TaskStatus.CANCELLED
    return {"message": "Task cancelled"}
