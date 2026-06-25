from abc import ABC, abstractmethod
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama

from core.config import settings

ModelTier = Literal["cheap", "standard", "frontier"]
Provider = Literal["openai", "mistral", "ollama"]

TIER_MODEL_MAP: dict[Provider, dict[ModelTier, str]] = {
    "openai": {
        "cheap": settings.tier_cheap_model,
        "standard": settings.tier_standard_model,
        "frontier": settings.tier_frontier_model,
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
        return ChatOpenAI(model=model, temperature=temperature, api_key=settings.openai_api_key)


class MistralProvider(BaseLLMProvider):
    def get_chat_model(self, model: str, temperature: float = 0.0) -> ChatMistralAI:
        return ChatMistralAI(model=model, temperature=temperature, api_key=settings.mistral_api_key)


class OllamaProvider(BaseLLMProvider):
    def get_chat_model(self, model: str, temperature: float = 0.0) -> ChatOllama:
        return ChatOllama(model=model, temperature=temperature, base_url=settings.ollama_base_url)


_PROVIDERS: dict[Provider, BaseLLMProvider] = {
    "openai": OpenAIProvider(),
    "mistral": MistralProvider(),
    "ollama": OllamaProvider(),
}


def create_llm(tier: ModelTier, temperature: float = 0.0) -> BaseChatModel:
    provider = _PROVIDERS[settings.llm_provider]
    model_name = TIER_MODEL_MAP[settings.llm_provider][tier]
    return provider.get_chat_model(model=model_name, temperature=temperature)
