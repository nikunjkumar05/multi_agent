import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from middleware.api.routes import budgets, tasks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

log = logging.getLogger("middleware")

app = FastAPI(
    title="BAMAS Middleware API",
    description="Budget-Aware Proxy for Coding AI Agents",
    version="1.0.0",
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
    allow_headers=["Authorization", "Content-Type"],
)

# Include the routers
app.include_router(budgets.router)
app.include_router(tasks.router)

@app.on_event("startup")
async def startup_banner():
    from middleware.api.state import registry

    summary = registry.get_summary()
    log.info(
        "BAMAS Middleware ready — agents: %d registered, %d healthy",
        summary["total_agents"], summary["healthy_agents"],
    )

@app.get("/")
async def root():
    return {
        "message": "Welcome to BAMAS Middleware API",
        "docs_url": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}
