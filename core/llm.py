from abc import ABC, abstractmethod
from typing import Literal

from langchain_core.language_models import BaseLanguageModel
from langchain_openai import OpenAI
from langchain_anthropic import Anthropic
from langchain_ollama import Ollama

from core.config import settings
ModelTier = Literal["cheap", "standard", "frontier"]
Provider = Literal["openai", "anthropic", "ollama"]

TIER_MODEL_MAP : dict[ProviderName, dict[ModelTier, str]] = {
    "openai": {
        "cheap": settings.tier_cheap_model,
        "standard": settings.tier_standard_model,
        "frontier": settings.tier_frontier_model,
    },
    "anthropic": {
        "cheap": "claude-3-5-haiku-latest",
        "standard": "claude-3-5-sonnet-latest",
        "frontier": "claude-opus-4-20250514",
    },
    "ollama": {
        "cheap": "llama3.2:3b",
        "standard": "llama3.2:7b",
        "frontier": "qwen2.5:14b",
    },
}
class BaseLLMProvider(ABC):
    @abstractmethod
    def get_chat_model(self,model: str,temperature: float = 0.0) -> BaseChatModel:
        pass
class OpenAIProvider(BaseLLMProvider):
    def get_chat_model(self, model: str, temperature: float = 0.0) -> ChatOpenAI:
        return ChatOpenAI(model=model, temperature=temperature, api_key=settings.openai_api_key)

class AnthropicProvider(BaseLLMProvider):
    def get_chat_model(self, model: str, temperature: float = 0.0) -> ChatAnthropic:
        return ChatAnthropic(model=model, temperature=temperature, api_key=settings.anthropic_api_key)

class OllamaProvider(BaseLLMProvider):
    def get_chat_model(self, model: str, temperature: float = 0.0) -> ChatOllama:
        return ChatOllama(model=model, temperature=temperature, base_url=settings.ollama_base_url)

_PROVIDERS: dict[ProviderName, BaseLLMProvider] = {
    "openai": OpenAIProvider(),
    "anthropic": AnthropicProvider(),
    "ollama": OllamaProvider(),
}
def create_llm(tier: ModelTier, provider: Provider) -> BaseChatModel:
    provider = _PROVIDERS[settings.llm_provider]
    model_name = TIER_MODEL_MAP[settings.llm_provider][tier]
    return provider.get_chat_model(model=model_name)
