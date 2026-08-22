from abc import ABC, abstractmethod
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

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
    """Estimate cost using paper's Eq. 1: c_i = T_in * P_in + T_out * P_out.

    Uses actual usage metadata when available, falls back to estimates.
    """
    usage = getattr(response, "usage_metadata", None)
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    if usage:
        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens", 0) or 0
            output_tokens = usage.get("output_tokens", 0) or 0
            total_tokens = usage.get("total_tokens", 0) or 0
        else:
            raw_in = getattr(usage, "input_tokens", None)
            raw_out = getattr(usage, "output_tokens", None)
            raw_total = getattr(usage, "total_tokens", None)
            input_tokens = int(raw_in) if isinstance(raw_in, (int, float)) else 0
            output_tokens = int(raw_out) if isinstance(raw_out, (int, float)) else 0
            total_tokens = int(raw_total) if isinstance(raw_total, (int, float)) else 0

    # If we have total but not split, assume 80% input / 20% output
    if total_tokens > 0 and input_tokens == 0 and output_tokens == 0:
        input_tokens = int(total_tokens * 0.8)
        output_tokens = total_tokens - input_tokens

    # Fallback: estimate from content if no usage metadata
    if total_tokens == 0 and input_tokens == 0 and output_tokens == 0:
        content = getattr(response, "content", "")
        if isinstance(content, str):
            total_tokens = max(1, len(content) // 4)
        input_tokens = int(total_tokens * 0.8)
        output_tokens = total_tokens - input_tokens

    # Paper Eq. 1: c_i = T_in * P_in + T_out * P_out
    input_cost_per_1k = settings.tier_input_cost_per_1k.get(tier, 0.001)
    output_cost_per_1k = settings.tier_output_cost_per_1k.get(tier, 0.003)
    cost = (input_tokens / 1000.0) * input_cost_per_1k + (output_tokens / 1000.0) * output_cost_per_1k
    return cost


def estimate_cost_from_tokens(input_tokens: int, output_tokens: int, tier: ModelTier) -> float:
    """Estimate cost from explicit token counts using paper's Eq. 1."""
    input_cost_per_1k = settings.tier_input_cost_per_1k.get(tier, 0.001)
    output_cost_per_1k = settings.tier_output_cost_per_1k.get(tier, 0.003)
    return (input_tokens / 1000.0) * input_cost_per_1k + (output_tokens / 1000.0) * output_cost_per_1k
