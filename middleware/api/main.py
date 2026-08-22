import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from middleware import persistence
from middleware.api.routes import budgets, tasks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

log = logging.getLogger("middleware")

# Dashboard UI assets (middleware/api/static)
_STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: restore persisted state. Shutdown: nothing to flush (write-through)."""
    from middleware.api.state import budget_manager, registry, tasks_db

    # 1. Restore budgets and tasks from SQLite (if a DB exists).
    task_records, budget_dicts = persistence.load_all()
    for b in budget_dicts:
        budget_manager.restore_budget(b)
    for t in task_records:
        tid = t.get("task_id")
        if not tid:
            continue
        # Tasks interrupted by a restart can't resume — mark them failed.
        if t.get("status") in ("queued", "in_progress"):
            t["status"] = "failed"
            t["error"] = "Interrupted by server restart"
        tasks_db[tid] = t

    summary = registry.get_summary()
    log.info(
        "BAMAS Middleware ready — agents: %d registered, %d healthy; "
        "restored %d budgets, %d tasks",
        summary["total_agents"], summary["healthy_agents"],
        len(budget_dicts), len(task_records),
    )

    if os.getenv("BAMAS_MIDDLEWARE_TEST_MODE") == "1":
        log.warning("TEST MODE: real agents disabled, MockAdapter only")

    yield


app = FastAPI(
    title="BAMAS Middleware API",
    description="Budget-Aware Proxy for Coding AI Agents",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS: localhost-only by default; override via BAMAS_CORS_ORIGINS (comma-separated).
# Never combine wildcard origins with allow_credentials=True.
_cors_origins = [
    o.strip() for o in os.getenv(
        "BAMAS_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://localhost:8080",
    ).split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PATCH"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


@app.middleware("http")
async def api_key_guard(request, call_next):
    """Require X-API-Key on /api/* when BAMAS_API_KEY is set.

    Unset key => local-only trust model (allow all); warned at startup.
    """
    expected = os.getenv("BAMAS_API_KEY")
    if (
        expected
        and request.url.path.startswith("/api/")
        and request.headers.get("X-API-Key") != expected
    ):
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing X-API-Key"})
    return await call_next(request)


@app.on_event("startup")
async def warn_if_unprotected():
    if not os.getenv("BAMAS_API_KEY"):
        log.warning("BAMAS_API_KEY unset — API is UNPROTECTED (local-only mode)")


# Include the routers
app.include_router(budgets.router)
app.include_router(tasks.router)


@app.get("/")
async def root():
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


# Dashboard UI (mounted LAST so /api/*, /docs, /openapi.json keep priority)
app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR), html=True), name="ui")
