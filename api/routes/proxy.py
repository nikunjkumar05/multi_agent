"""
OpenAI-compatible proxy endpoints.

Point OPENAI_BASE_URL at BAMAS and existing code works unchanged.

    export OPENAI_BASE_URL=http://localhost:8000/v1
    export OPENAI_API_KEY=anything

Endpoints:
    POST /v1/chat/completions   — run task through BAMAS, return OpenAI-format response
    GET  /v1/models             — list available models
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from api.middleware.auth import require_auth
from api.models.openai_schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceMessage,
    DeltaMessage,
    ModelListResponse,
    ModelObject,
    StreamChoice,
    Usage,
)

log = logging.getLogger(__name__)

router = APIRouter()

_DEFAULT_BUDGET_USD = 0.10
_POLL_INTERVAL = 0.5  # seconds between status polls
_POLL_TIMEOUT = 120   # max seconds to wait for task completion

# ── Model → Tier Mapping ──────────────────────────────────────────────

_MODEL_TO_TIER: dict[str, str] = {
    "gpt-4o-mini": "cheap",
    "gpt-3.5-turbo": "cheap",
    "gpt-4o": "standard",
    "gpt-4-turbo": "standard",
    "gpt-4": "frontier",
    "o1": "frontier",
    "o1-mini": "standard",
    "o3": "frontier",
    "o3-mini": "standard",
    "mistral-tiny": "cheap",
    "mistral-small-latest": "standard",
    "mistral-large-latest": "frontier",
}

# ── Available Models ───────────────────────────────────────────────────

_AVAILABLE_MODELS = [
    ModelObject(id="gpt-4o-mini"),
    ModelObject(id="gpt-4o"),
    ModelObject(id="gpt-4-turbo"),
    ModelObject(id="gpt-4"),
    ModelObject(id="o1"),
    ModelObject(id="o3"),
    ModelObject(id="mistral-tiny"),
    ModelObject(id="mistral-small-latest"),
    ModelObject(id="mistral-large-latest"),
]


def _map_model_to_tier(model: str) -> str:
    """Map an OpenAI/compatible model name to a BAMAS cost tier."""
    model_lower = model.lower().strip()
    return _MODEL_TO_TIER.get(model_lower, "standard")


def _extract_prompt(messages: list[ChatMessage]) -> str:
    """Extract the task prompt from chat messages.

    Strategy: take the last user message content. If it contains prior
    system/assistant messages, prepend them as context.
    """
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # Find all user messages
    user_msgs = [m for m in messages if m.role == "user" and m.content]
    if not user_msgs:
        # Fall back to last message with content
        for m in reversed(messages):
            if m.content:
                return m.content
        raise HTTPException(status_code=400, detail="No content in messages")

    last_user = user_msgs[-1].content

    # If there's a system message, prepend it as context
    system_msgs = [m for m in messages if m.role == "system" and m.content]
    if system_msgs:
        return f"System: {system_msgs[-1].content}\n\nUser: {last_user}"

    # If there are prior user/assistant messages, include them
    prior = []
    for m in messages:
        if m.role in ("user", "assistant") and m.content:
            prior.append(f"{m.role.title()}: {m.content}")
    if len(prior) > 1:
        return "\n\n".join(prior)

    return last_user


# ── /v1/models ─────────────────────────────────────────────────────────


@router.get("/v1/models", dependencies=[Depends(require_auth)])
async def list_models() -> ModelListResponse:
    return ModelListResponse(data=_AVAILABLE_MODELS)


# ── /v1/chat/completions ──────────────────────────────────────────────


@router.post("/v1/chat/completions", dependencies=[Depends(require_auth)], response_model=None)
async def chat_completions(
    req: ChatCompletionRequest,
    x_bamas_budget_usd: float | None = Header(default=None, convert_underscores=False),
) -> ChatCompletionResponse | StreamingResponse:
    """OpenAI-compatible chat completions endpoint.

    Routes the request through the BAMAS multi-agent pipeline.
    Returns OpenAI-format response (streaming or non-streaming).
    """
    from agent.graph import run_task
    from api.routes.execute import _tasks, _run_background, _evict_if_needed
    from core.budget import BudgetTracker

    # Determine budget
    budget_usd = x_bamas_budget_usd or req.budget_usd or _DEFAULT_BUDGET_USD
    budget_usd = max(0.001, min(budget_usd, 100.0))  # clamp

    # Extract task prompt
    prompt = _extract_prompt(req.messages)

    # Budget tracker
    budget = BudgetTracker(max_cost_usd=budget_usd)

    task_id = f"proxy-{uuid.uuid4().hex[:12]}"

    if req.stream:
        return StreamingResponse(
            _stream_chat(task_id, prompt, budget, req),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming: run task and wait for completion
    _evict_if_needed()
    try:
        result = await run_task(
            task=prompt,
            budget=budget,
            task_id=task_id,
        )
    except Exception as e:
        log.exception("Proxy task %s failed", task_id)
        raise HTTPException(status_code=500, detail=f"BAMAS task failed: {e}")

    # Extract response content
    content = result.get("final_result") or result.get("judge_output") or ""
    if not content:
        content = f"[Task completed with status: {result.get('status', 'unknown')}]"

    # Build usage from budget tracker
    spent_pct = result.get("budget_spent_pct", 0.0)
    est_tokens = max(1, int(len(content) // 4))  # rough estimate
    usage = Usage(
        prompt_tokens=max(1, len(prompt) // 4),
        completion_tokens=est_tokens,
        total_tokens=max(1, len(prompt) // 4 + est_tokens),
    )

    return ChatCompletionResponse(
        model=req.model,
        choices=[
            Choice(
                message=ChoiceMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )


# ── Streaming SSE ──────────────────────────────────────────────────────


async def _stream_chat(
    task_id: str,
    prompt: str,
    budget: "BudgetTracker",
    req: ChatCompletionRequest,
) -> AsyncGenerator[str, None]:
    """Run task through BAMAS and stream results as OpenAI SSE chunks."""
    import asyncio
    from core.events import EventBroadcaster
    from core.redis_client import get_redis
    from agent.graph import run_task

    chunk_id = f"chatcmpl-bamas-{uuid.uuid4().hex[:12]}"

    # Yield the role chunk first
    yield _format_chunk(chunk_id, req.model, delta_role="assistant")

    # Start task in background
    task = asyncio.create_task(run_task(task=prompt, budget=budget, task_id=task_id))

    # Try to stream events from Redis PubSub
    redis = await get_redis()
    if redis is not None:
        broadcaster = EventBroadcaster(redis)
        pubsub = redis.pubsub()
        await pubsub.subscribe(f"events:{task_id}")

        accumulated = ""
        last_event = None

        try:
            while not task.done():
                # Check for new events (non-blocking with short timeout)
                try:
                    message = await asyncio.wait_for(pubsub.get_message(), timeout=0.5)
                    if message and message["type"] == "message":
                        event = json.loads(message["data"])
                        event_type = event.get("event_type", "")
                        data = event.get("data", {})

                        # Emit content chunks for completed steps
                        if event_type == "step_completed":
                            step_result = data.get("result_preview", "")
                            if step_result and step_result != accumulated:
                                delta = step_result[len(accumulated):]
                                if delta:
                                    yield _format_chunk(chunk_id, req.model, delta_content=delta)
                                    accumulated = step_result

                        elif event_type == "agent_completed":
                            agent_content = data.get("content_preview", "")
                            if agent_content and agent_content != accumulated:
                                delta = agent_content[len(accumulated):]
                                if delta:
                                    yield _format_chunk(chunk_id, req.model, delta_content=delta)
                                    accumulated = agent_content

                        last_event = event
                except asyncio.TimeoutError:
                    pass

            # Task finished — get result
            result = await task
            final_content = result.get("final_result") or result.get("judge_output") or ""

            # Stream any remaining content
            if final_content and final_content != accumulated:
                delta = final_content[len(accumulated):]
                if delta:
                    yield _format_chunk(chunk_id, req.model, delta_content=delta)

        except Exception as e:
            log.warning("Streaming error for %s: %s", task_id, e)
            # Send error as content
            error_msg = f"\n\n[Streaming error: {e}]"
            yield _format_chunk(chunk_id, req.model, delta_content=error_msg)

        finally:
            try:
                await pubsub.unsubscribe(f"events:{task_id}")
                await pubsub.close()
            except Exception:
                pass
    else:
        # No Redis — just wait for the task and send final result
        try:
            result = await task
            final_content = result.get("final_result") or result.get("judge_output") or ""
            if final_content:
                yield _format_chunk(chunk_id, req.model, delta_content=final_content)
        except Exception as e:
            yield _format_chunk(chunk_id, req.model, delta_content=f"\n[Error: {e}]")

    # Final chunk with finish_reason
    yield _format_chunk(chunk_id, req.model, finish_reason="stop")
    yield "data: [DONE]\n\n"


def _format_chunk(
    chunk_id: str,
    model: str,
    delta_role: str | None = None,
    delta_content: str | None = None,
    finish_reason: str | None = None,
) -> str:
    """Format a single SSE chunk in OpenAI format."""
    delta = DeltaMessage(role=delta_role, content=delta_content)
    choice = StreamChoice(delta=delta, finish_reason=finish_reason)
    chunk = ChatCompletionChunk(
        id=chunk_id,
        model=model,
        choices=[choice],
    )
    return f"data: {chunk.model_dump_json()}\n\n"
