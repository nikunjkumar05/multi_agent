import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api import websocket
from api.routes import audit, estimate, execute, rl, tasks


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialise persistent resources on startup."""
    from core.audit import audit_trail
    from core.redis_client import get_redis
    from core.rl_policy import RLPolicy

    await audit_trail.init_db()

    # Initialize RL policy SQLite schema
    redis = await get_redis()
    if redis:
        rl = RLPolicy(redis)
        await rl._ensure_db()

    yield
    # Shutdown hooks can be added here if needed


app = FastAPI(
    title="BAMAS — Budget-Aware Multi-Agent System",
    description="Budget-aware multi-agent system with cost-tier optimizer, RL topology selection, and topology degradation",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(execute.router)
app.include_router(tasks.router)
app.include_router(audit.router)
app.include_router(estimate.router)
app.include_router(rl.router)
app.include_router(websocket.router)

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
