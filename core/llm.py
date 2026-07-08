from abc import ABC, abstractmethod
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama

from core.config import settings

ModelTier = Literal["cheap", "standard", "frontier"]
Provider = Literal["openai", "mistral", "ollama"]

TIER_MODEL_MAP: dict[Provider, dict[ModelTier, str]] = {
    "openai": {
        "cheap": "gpt-4o-mini",
        "standard": "gpt-4o",
        "frontier": "o3",
    },
    "mistral": {
        "cheap": settings.tier_cheap_model,
        "standard": settings.tier_standard_model,
        "frontier": settings.tier_frontier_model,
    },
    "ollama": {
        "cheap": "llama3.2:3b",
        "standard": "llama3.2:7b",
        "frontier": "qwen2.5:14b",
    },
}


class BaseLLMProvider(ABC):
    @abstractmethod
    def get_chat_model(self, model: str, temperature: float = 0.0) -> BaseChatModel: ...


class OpenAIProvider(BaseLLMProvider):
    def get_chat_model(self, model: str, temperature: float = 0.0) -> ChatOpenAI:
        return ChatOpenAI(model=model, temperature=temperature, api_key=settings.openai_api_key, timeout=settings.llm_request_timeout)


class MistralProvider(BaseLLMProvider):
    def get_chat_model(self, model: str, temperature: float = 0.0) -> ChatMistralAI:
        return ChatMistralAI(model=model, temperature=temperature, api_key=settings.mistral_api_key, timeout=settings.llm_request_timeout)


class OllamaProvider(BaseLLMProvider):
    def get_chat_model(self, model: str, temperature: float = 0.0) -> ChatOllama:
        return ChatOllama(model=model, temperature=temperature, base_url=settings.ollama_base_url, timeout=settings.llm_request_timeout)


_PROVIDERS: dict[Provider, BaseLLMProvider] = {
    "openai": OpenAIProvider(),
    "mistral": MistralProvider(),
    "ollama": OllamaProvider(),
}


def create_llm(tier: ModelTier, temperature: float = 0.0) -> BaseChatModel:
    provider = _PROVIDERS[settings.llm_provider]
    model_name = TIER_MODEL_MAP[settings.llm_provider][tier]
    return provider.get_chat_model(model=model_name, temperature=temperature)


def estimate_tokens(response: Any) -> int:
    usage = getattr(response, "usage_metadata", None)
    if usage:
        if isinstance(usage, dict):
            total = usage.get("total_tokens")
            if total is not None:
                return int(total)
        elif hasattr(usage, "total_tokens"):
            return usage.total_tokens or 0
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return max(1, len(content) // 4)
    return 0


def estimate_cost(response: Any, tier: ModelTier) -> float:
    tokens = estimate_tokens(response)
    cost_per_1k = settings.tier_cost_per_1k_tokens.get(tier, 0.001)
    return (tokens / 1000.0) * cost_per_1k
