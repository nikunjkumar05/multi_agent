import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from middleware.api.routes import tasks

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

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the routers
app.include_router(tasks.router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to BAMAS Middleware API",
        "docs_url": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}
