from typing import Literal

from pydantic_settings import BaseSettings

DEFAULT_JWT_SECRET = "change-me-in-production"


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    llm_provider: Literal["openai", "mistral", "ollama"] = "openai"
    openai_api_key: str | None = None
    mistral_api_key: str | None = None
    ollama_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    mistral_model: str = "mistral-large-latest"

    tier_cheap_model: str = "mistral-tiny"
    tier_standard_model: str = "mistral-small-latest"
    tier_frontier_model: str = "mistral-large-latest"

    tier_cost_per_1k_tokens: dict[str, float] = {
        "cheap": 0.0002,
        "standard": 0.001,
        "frontier": 0.008,
    }

    budget_max_cost_usd: float = 1.00
    budget_max_tokens: int = 100_000
    llm_request_timeout: int = 30

    redis_url: str = "redis://localhost:6379/0"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    jwt_secret: str = "change-me-in-production"
    rl_model_key: str = "rl_policy"

    # RL policy tuning (thresholds exposed so they can be overridden in .env)
    rl_min_tasks_for_selection: int = 5  # Tasks before RL starts suggesting topologies
    rl_min_tasks_for_override: int = 50  # Tasks before RL can override the LLM decision
    rl_quality_weight: float = 0.7  # Weight of quality score in the reward signal
    rl_cost_efficiency_weight: float = 0.3  # Weight of cost efficiency in the reward signal

    # Database
    database_url: str | None = None  # PostgreSQL DSN (e.g. postgresql://user:pass@host/db). Falls back to SQLite if unset.
    audit_db_path: str = "./workspace/audit.db"  # Used only when database_url is not set
    db_pool_min_size: int = 2
    db_pool_max_size: int = 10

    # Celery
    celery_broker_url: str | None = None  # Defaults to redis_url if unset


settings = Settings()
