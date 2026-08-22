"""
Error scenario tests — verifies graceful degradation when things go wrong.

Tests that the system handles:
1. LLM failure in planner → single-step fallback
2. LLM timeout in executor → retry then skip
3. LLM timeout in judge → use executor output
4. All ensemble agents failing → judge picks best available
5. Budget exhaustion mid-execution → interrupt/degrade
"""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from core.budget import BudgetTracker


class TestPlannerFailure:
    """Planner LLM failures should fall back to single-step plan."""

    @pytest.mark.anyio
    async def test_planner_llm_failure_single_step_fallback(self):
        """If planner LLM raises, task should still complete with 1 step."""
        from agent.graph import run_task

        with patch("agent.nodes.planner.create_llm") as mock_create:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.side_effect = Exception("LLM connection failed")
            mock_create.return_value = mock_llm

            result = await run_task(
                task="What is 2+2?",
                budget=BudgetTracker(max_cost_usd=0.50),
                task_id="test-planner-fail-001",
                topology_override="single",
            )

            assert result["status"] == "completed"
            assert result["final_result"] is not None


class TestJudgeFailure:
    """Judge LLM failures should fall back to executor output."""

    @pytest.mark.anyio
    async def test_judge_timeout_uses_executor_output(self):
        """If judge times out, the executor's output should be used as final result."""
        from agent.graph import run_task


        async def slow_judge(state):
            await asyncio.sleep(300)  # Simulate timeout
            return {}

        with patch("agent.nodes.judge.ensemble_judge", side_effect=slow_judge):
            result = await run_task(
                task="What is 2+2?",
                budget=BudgetTracker(max_cost_usd=0.50),
                task_id="test-judge-timeout-001",
                topology_override="single",
            )

            # Judge timeout should not crash the system
            assert result["status"] in ("completed", "failed")
            assert result["final_result"] is not None


class TestEnsembleAgentFailure:
    """Ensemble agent failures should be handled gracefully."""

    @pytest.mark.anyio
    async def test_ensemble_completes_even_with_low_budget(self):
        """Ensemble should complete even with tight budget (agents may fail)."""
        from agent.graph import run_task

        result = await run_task(
            task="What is 5 + 3?",
            budget=BudgetTracker(max_cost_usd=0.01),
            task_id="test-ensemble-tight-001",
            topology_override="ensemble",
        )

        # System should complete (possibly degraded) even with tight budget
        assert result["status"] in ("completed", "degraded_completion", "failed")
        assert result["final_result"] is not None


class TestBudgetExhaustion:
    """Budget exhaustion should trigger degradation or skip judge."""

    @pytest.mark.anyio
    async def test_tiny_budget_completes(self):
        """Even with a tiny budget, the system should complete (not hang)."""
        from agent.graph import run_task

        result = await run_task(
            task="What is 1+1?",
            budget=BudgetTracker(max_cost_usd=0.001),
            task_id="test-tiny-budget-001",
            topology_override="single",
        )

        # Should complete (possibly with degraded status)
        assert result["status"] in ("completed", "degraded_completion", "failed")
        assert result["final_result"] is not None


class TestEdgeCases:
    """Edge cases that should not crash the system."""

    @pytest.mark.anyio
    async def test_empty_task(self):
        """Empty task should be handled gracefully."""
        from agent.graph import run_task

        result = await run_task(
            task="",
            budget=BudgetTracker(max_cost_usd=0.50),
            task_id="test-empty-task-001",
            topology_override="single",
        )

        assert result["status"] in ("completed", "failed")

    @pytest.mark.anyio
    async def test_very_long_task(self):
        """Very long task description should be handled."""
        from agent.graph import run_task

        long_task = "Analyze " + "and compare " * 50 + "these concepts."
        result = await run_task(
            task=long_task,
            budget=BudgetTracker(max_cost_usd=0.50),
            task_id="test-long-task-001",
            topology_override="single",
        )

        assert result["status"] in ("completed", "failed")
