from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    llm_provider: Literal["openai", "anthropic", "ollama"] = "openai"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ollama_api_key: str | None = None

    tier_cheap_model: str = "gpt-3.5-turbo"
    tier_standard_model: str = "gpt-4"
    tier_frontier_model: str = "gpt-4o"

    budget_max_cost_usd: float = 1.00
    budget_max_tokens: int = 100_000

    redis_url: str = "redis://localhost:6379/0"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    jwt_secret: str = "change-me-in-production"

settings = Settings()
