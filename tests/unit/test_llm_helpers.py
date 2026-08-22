from unittest.mock import MagicMock

import pytest

from core.llm import estimate_cost, estimate_tokens


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
        # Paper Eq. 1: 800 input * $0.0001/1k + 200 output * $0.0004/1k = $0.00008 + $0.00008
        assert cost == pytest.approx(0.00016, rel=1e-6)

    def test_standard_tier(self):
        response = MagicMock()
        response.usage_metadata.total_tokens = 1000
        cost = estimate_cost(response, "standard")
        # 800 * $0.0003/1k + 200 * $0.001/1k = $0.00024 + $0.0002
        assert cost == pytest.approx(0.00044, rel=1e-6)

    def test_frontier_tier(self):
        response = MagicMock()
        response.usage_metadata.total_tokens = 1000
        cost = estimate_cost(response, "frontier")
        # 800 * $0.002/1k + 200 * $0.006/1k = $0.0016 + $0.0012
        assert cost == pytest.approx(0.0028, rel=1e-6)

    def test_half_thousand_tokens(self):
        response = MagicMock()
        response.usage_metadata.total_tokens = 500
        cost = estimate_cost(response, "standard")
        # 400 * $0.0003/1k + 100 * $0.001/1k = $0.00012 + $0.0001
        assert cost == pytest.approx(0.00022, rel=1e-6)
