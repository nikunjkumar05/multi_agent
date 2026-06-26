import pytest
from unittest.mock import MagicMock
from core.llm import estimate_tokens, estimate_cost


class TestEstimateTokens:
    def test_with_usage_metadata(self):
        response = MagicMock()
        response.usage_metadata.total_tokens = 500
        assert estimate_tokens(response) == 500

    def test_with_zero_tokens(self):
        response = MagicMock()
        response.usage_metadata.total_tokens = 0
        assert estimate_tokens(response) == 0

    def test_without_usage_metadata(self):
        response = MagicMock()
        response.usage_metadata = None
        response.content = "hello world"
        assert estimate_tokens(response) == 2

    def test_fallback_string_content(self):
        response = MagicMock()
        del response.usage_metadata
        response.content = "a" * 100
        assert estimate_tokens(response) == 25

    def test_empty_content(self):
        response = MagicMock()
        del response.usage_metadata
        response.content = ""
        assert estimate_tokens(response) == 1


class TestEstimateCost:
    def test_cheap_tier(self):
        response = MagicMock()
        response.usage_metadata.total_tokens = 1000
        cost = estimate_cost(response, "cheap")
        assert cost == pytest.approx(0.0002, rel=1e-6)

    def test_standard_tier(self):
        response = MagicMock()
        response.usage_metadata.total_tokens = 1000
        cost = estimate_cost(response, "standard")
        assert cost == pytest.approx(0.001, rel=1e-6)

    def test_frontier_tier(self):
        response = MagicMock()
        response.usage_metadata.total_tokens = 1000
        cost = estimate_cost(response, "frontier")
        assert cost == pytest.approx(0.008, rel=1e-6)

    def test_half_thousand_tokens(self):
        response = MagicMock()
        response.usage_metadata.total_tokens = 500
        cost = estimate_cost(response, "standard")
        assert cost == pytest.approx(0.0005, rel=1e-6)
