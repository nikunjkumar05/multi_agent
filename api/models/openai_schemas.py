"""OpenAI-compatible request/response models for the BAMAS proxy layer.

These models mirror the OpenAI Chat Completions API so that existing
client code (openai.Python, LangChain, curl, etc.) works unchanged
when pointing OPENAI_BASE_URL at BAMAS.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Request Models ─────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str = Field(..., description="System, user, assistant, or tool")
    content: str | None = Field(default=None)
    name: str | None = Field(default=None)


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model name (mapped to BAMAS tier internally)")
    messages: list[ChatMessage] = Field(..., min_length=1)
    stream: bool = Field(default=False)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    user: str | None = Field(default=None)

    # BAMAS-specific extensions (ignored by OpenAI clients, used by BAMAS proxy)
    budget_usd: float | None = Field(
        default=None,
        alias="x-bamas-budget-usd",
        description="BAMAS budget override. If absent, proxy uses default.",
    )

    model_config = {"extra": "allow"}


# ── Response Models ────────────────────────────────────────────────────


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-bamas-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage


# ── Streaming Models ───────────────────────────────────────────────────


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-bamas-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[StreamChoice]


# ── Model List (for /v1/models) ───────────────────────────────────────


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "bamas"


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]
