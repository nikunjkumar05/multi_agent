"""Unit tests for the OpenAI proxy layer."""

import pytest
from api.models.openai_schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChatMessage,
    ModelListResponse,
)
from api.routes.proxy import (
    _extract_prompt,
    _map_model_to_tier,
    _format_chunk,
    _DEFAULT_BUDGET_USD,
)


# ── Model → Tier Mapping ──────────────────────────────────────────────

class TestMapModelToTier:
    def test_gpt4o_mini_is_cheap(self):
        assert _map_model_to_tier("gpt-4o-mini") == "cheap"

    def test_gpt35_turbo_is_cheap(self):
        assert _map_model_to_tier("gpt-3.5-turbo") == "cheap"

    def test_gpt4o_is_standard(self):
        assert _map_model_to_tier("gpt-4o") == "standard"

    def test_gpt4_turbo_is_standard(self):
        assert _map_model_to_tier("gpt-4-turbo") == "standard"

    def test_gpt4_is_frontier(self):
        assert _map_model_to_tier("gpt-4") == "frontier"

    def test_o1_is_frontier(self):
        assert _map_model_to_tier("o1") == "frontier"

    def test_o3_is_frontier(self):
        assert _map_model_to_tier("o3") == "frontier"

    def test_o3_mini_is_standard(self):
        assert _map_model_to_tier("o3-mini") == "standard"

    def test_unknown_model_defaults_to_standard(self):
        assert _map_model_to_tier("claude-3-opus") == "standard"

    def test_case_insensitive(self):
        assert _map_model_to_tier("GPT-4o") == "standard"
        assert _map_model_to_tier("gpt-4o") == "standard"

    def test_strips_whitespace(self):
        assert _map_model_to_tier("  gpt-4o  ") == "standard"

    def test_mistral_models(self):
        assert _map_model_to_tier("mistral-tiny") == "cheap"
        assert _map_model_to_tier("mistral-small-latest") == "standard"
        assert _map_model_to_tier("mistral-large-latest") == "frontier"


# ── Prompt Extraction ─────────────────────────────────────────────────

class TestExtractPrompt:
    def test_single_user_message(self):
        messages = [ChatMessage(role="user", content="Hello")]
        assert _extract_prompt(messages) == "Hello"

    def test_system_plus_user(self):
        messages = [
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="Hi"),
        ]
        result = _extract_prompt(messages)
        assert "You are helpful" in result
        assert "Hi" in result

    def test_multi_turn(self):
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there"),
            ChatMessage(role="user", content="How are you?"),
        ]
        result = _extract_prompt(messages)
        assert "Hello" in result
        assert "How are you?" in result

    def test_no_messages_raises(self):
        with pytest.raises(Exception):
            _extract_prompt([])

    def test_only_system_messages_fallback(self):
        messages = [ChatMessage(role="system", content="System msg")]
        result = _extract_prompt(messages)
        assert result == "System msg"

    def test_none_content_skipped(self):
        messages = [
            ChatMessage(role="user", content=None),
            ChatMessage(role="user", content="Actual message"),
        ]
        assert _extract_prompt(messages) == "Actual message"

    def test_long_conversation_takes_last_user(self):
        messages = [
            ChatMessage(role="user", content="First question"),
            ChatMessage(role="assistant", content="First answer"),
            ChatMessage(role="user", content="Second question"),
            ChatMessage(role="assistant", content="Second answer"),
            ChatMessage(role="user", content="Third question"),
        ]
        result = _extract_prompt(messages)
        assert "Third question" in result


# ── SSE Chunk Formatting ──────────────────────────────────────────────

class TestFormatChunk:
    def test_role_chunk(self):
        chunk = _format_chunk("cmpl-123", "gpt-4o", delta_role="assistant")
        assert "chatcmpl" in chunk or "cmpl" in chunk
        assert "assistant" in chunk
        assert chunk.startswith("data: ")
        assert chunk.endswith("\n\n")

    def test_content_chunk(self):
        chunk = _format_chunk("cmpl-123", "gpt-4o", delta_content="Hello")
        assert "Hello" in chunk
        assert "chat.completion.chunk" in chunk

    def test_finish_chunk(self):
        chunk = _format_chunk("cmpl-123", "gpt-4o", finish_reason="stop")
        assert "stop" in chunk

    def test_chunk_is_valid_json(self):
        import json
        chunk = _format_chunk("cmpl-123", "gpt-4o", delta_content="test")
        # Strip "data: " prefix and parse
        json_str = chunk.removeprefix("data: ").strip()
        parsed = json.loads(json_str)
        assert parsed["id"] == "cmpl-123"
        assert parsed["model"] == "gpt-4o"
        assert parsed["object"] == "chat.completion.chunk"


# ── Schema Validation ─────────────────────────────────────────────────

class TestSchemas:
    def test_chat_message_defaults(self):
        msg = ChatMessage(role="user", content="hi")
        assert msg.name is None

    def test_request_defaults(self):
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="hi")],
        )
        assert req.stream is False
        assert req.temperature is None
        assert req.max_tokens is None

    def test_request_allows_extra_fields(self):
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="hi")],
            top_p=0.9,
        )
        assert req.model == "gpt-4o"

    def test_response_has_id(self):
        from api.models.openai_schemas import Usage
        resp = ChatCompletionResponse(
            model="gpt-4o",
            choices=[],
            usage=Usage(),
        )
        assert resp.id.startswith("chatcmpl-bamas-")
        assert resp.object == "chat.completion"

    def test_chunk_has_correct_object(self):
        chunk = ChatCompletionChunk(
            model="gpt-4o",
            choices=[],
        )
        assert chunk.object == "chat.completion.chunk"

    def test_model_list(self):
        resp = ModelListResponse(data=[])
        assert resp.object == "list"
        assert resp.data == []


# ── Default Budget ────────────────────────────────────────────────────

class TestDefaultBudget:
    def test_default_budget_is_10_cents(self):
        assert _DEFAULT_BUDGET_USD == 0.10
