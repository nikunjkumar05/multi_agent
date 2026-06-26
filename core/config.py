from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    llm_provider: Literal["openai", "mistral", "ollama"] = "openai"
    openai_api_key: str | None = None
    mistral_api_key: str | None = None
    ollama_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    mistral_model: str = "mistral-large-latest"

    tier_cheap_model: str = "mistral-tiny"
    tier_standard_model: str = "mistral-small-latest"
    tier_frontier_model: str = "mistral-large-latest"

    budget_max_cost_usd: float = 1.00
    budget_max_tokens: int = 100_000

    redis_url: str = "redis://localhost:6379/0"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    jwt_secret: str = "change-me-in-production"
    rl_model_key: str = "rl_policy"

settings = Settings()
