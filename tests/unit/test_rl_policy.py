import json

import pytest
from unittest.mock import AsyncMock

from core.rl_policy import RLPolicy, TOPOLOGIES, MIN_TASKS_TO_LEARN


class TestRLPolicyFeatures:
    def test_extract_features_code(self):
        mock_redis = AsyncMock()
        policy = RLPolicy(mock_redis)
        features = policy._extract_features("Write a Python function")
        assert features["is_code"] is True
        assert features["is_research"] is False

    def test_extract_features_research(self):
        mock_redis = AsyncMock()
        policy = RLPolicy(mock_redis)
        features = policy._extract_features("Explain how TCP works")
        assert features["is_research"] is True
        assert features["is_code"] is False

    def test_extract_features_data(self):
        mock_redis = AsyncMock()
        policy = RLPolicy(mock_redis)
        features = policy._extract_features("Analyze these datasets")
        assert features["is_data"] is True

    def test_extract_features_verify(self):
        mock_redis = AsyncMock()
        policy = RLPolicy(mock_redis)
        features = policy._extract_features("Audit this code for security")
        assert features["is_verify"] is True

    def test_extract_features_default(self):
        mock_redis = AsyncMock()
        policy = RLPolicy(mock_redis)
        features = policy._extract_features("What is 2+2?")
        assert features["is_code"] is False
        assert features["is_research"] is False
        assert features["is_data"] is False
        assert features["is_verify"] is False


class TestThompsonSampling:
    def test_returns_valid_topology(self):
        mock_redis = AsyncMock()
        policy = RLPolicy(mock_redis)
        policy.arms = {t: {"alpha": 1.0, "beta": 1.0} for t in TOPOLOGIES}
        result = policy._thompson_sample()
        assert result in TOPOLOGIES

    def test_biased_toward_high_alpha(self):
        mock_redis = AsyncMock()
        policy = RLPolicy(mock_redis)
        policy.arms = {
            "single":     {"alpha": 100.0, "beta": 1.0},
            "pipeline":   {"alpha": 1.0, "beta": 1.0},
            "supervisor": {"alpha": 1.0, "beta": 1.0},
            "fanout":     {"alpha": 1.0, "beta": 1.0},
            "ensemble":   {"alpha": 1.0, "beta": 1.0},
        }
        results = [policy._thompson_sample() for _ in range(100)]
        assert results.count("single") > 80

    def test_biased_toward_high_alpha_pipeline(self):
        mock_redis = AsyncMock()
        policy = RLPolicy(mock_redis)
        policy.arms = {
            "single":     {"alpha": 1.0, "beta": 1.0},
            "pipeline":   {"alpha": 50.0, "beta": 1.0},
            "supervisor": {"alpha": 1.0, "beta": 1.0},
            "fanout":     {"alpha": 1.0, "beta": 1.0},
            "ensemble":   {"alpha": 1.0, "beta": 1.0},
        }
        results = [policy._thompson_sample() for _ in range(100)]
        assert results.count("pipeline") > 60

    def test_context_weight_boosts_pipeline_for_code(self):
        mock_redis = AsyncMock()
        policy = RLPolicy(mock_redis)
        policy.arms = {t: {"alpha": 1.0, "beta": 1.0} for t in TOPOLOGIES}
        features = policy._extract_features("Write a code function")
        weights = policy._compute_context_weights(features)
        assert weights["pipeline"] > weights["supervisor"]
        assert weights["pipeline"] > weights["ensemble"]

    def test_context_weight_boosts_supervisor_for_research(self):
        mock_redis = AsyncMock()
        policy = RLPolicy(mock_redis)
        policy.arms = {t: {"alpha": 1.0, "beta": 1.0} for t in TOPOLOGIES}
        features = policy._extract_features("Explain and research this topic")
        weights = policy._compute_context_weights(features)
        assert weights["supervisor"] > weights["pipeline"]
        assert weights["supervisor"] > weights["fanout"]

    def test_context_weight_boosts_ensemble_for_verify(self):
        mock_redis = AsyncMock()
        policy = RLPolicy(mock_redis)
        policy.arms = {t: {"alpha": 1.0, "beta": 1.0} for t in TOPOLOGIES}
        features = policy._extract_features("Verify and audit the security")
        weights = policy._compute_context_weights(features)
        assert weights["ensemble"] > weights["pipeline"]

    def test_context_weight_multiplied_into_sample(self):
        mock_redis = AsyncMock()
        policy = RLPolicy(mock_redis)
        policy.arms = {t: {"alpha": 1.0, "beta": 1.0} for t in TOPOLOGIES}
        weights = {"pipeline": 100.0, "single": 0.01, "supervisor": 0.01, "fanout": 0.01, "ensemble": 0.01}
        results = [policy._thompson_sample(weights) for _ in range(100)]
        assert results.count("pipeline") > 95


class TestColdStart:
    @pytest.mark.asyncio
    async def test_returns_none_below_min_tasks(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value="2")
        policy = RLPolicy(mock_redis)
        result = await policy.select_topology("test task", "healthy")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_topology_above_min_tasks(self):
        default_arms = {t: {"alpha": 1.0, "beta": 1.0} for t in TOPOLOGIES}
        mock_redis = AsyncMock()

        async def fake_get(key):
            if key == "rl_policy:total_tasks":
                return "10"
            if key == "rl_policy:priors":
                return json.dumps(default_arms)
            return None

        mock_redis.get = AsyncMock(side_effect=fake_get)
        policy = RLPolicy(mock_redis)
        result = await policy.select_topology("Write a code function", "healthy")
        assert result in TOPOLOGIES


class TestRewardUpdate:
    @pytest.mark.asyncio
    async def test_good_reward_increases_alpha(self):
        default_arms = {t: {"alpha": 1.0, "beta": 1.0} for t in TOPOLOGIES}
        mock_redis = AsyncMock()

        async def fake_get(key):
            if key == "rl_policy:priors":
                return json.dumps(default_arms)
            return None

        mock_redis.get = AsyncMock(side_effect=fake_get)
        policy = RLPolicy(mock_redis)

        await policy.reward("pipeline", quality=1.0, cost_efficiency=1.0)

        assert policy.arms["pipeline"]["alpha"] > 1.0
        assert policy.arms["pipeline"]["beta"] == 1.0

    @pytest.mark.asyncio
    async def test_bad_reward_increases_beta(self):
        default_arms = {t: {"alpha": 1.0, "beta": 1.0} for t in TOPOLOGIES}
        mock_redis = AsyncMock()

        async def fake_get(key):
            if key == "rl_policy:priors":
                return json.dumps(default_arms)
            return None

        mock_redis.get = AsyncMock(side_effect=fake_get)
        policy = RLPolicy(mock_redis)

        await policy.reward("pipeline", quality=0.0, cost_efficiency=0.0)

        assert policy.arms["pipeline"]["alpha"] == 1.0
        assert policy.arms["pipeline"]["beta"] > 1.0

    @pytest.mark.asyncio
    async def test_saves_to_redis(self):
        default_arms = {t: {"alpha": 1.0, "beta": 1.0} for t in TOPOLOGIES}
        mock_redis = AsyncMock()

        async def fake_get(key):
            if key == "rl_policy:priors":
                return json.dumps(default_arms)
            return None

        mock_redis.get = AsyncMock(side_effect=fake_get)
        policy = RLPolicy(mock_redis)

        await policy.reward("pipeline", quality=0.8, cost_efficiency=0.9)

        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args[0][0] == "rl_policy:priors"

    @pytest.mark.asyncio
    async def test_increments_task_count(self):
        default_arms = {t: {"alpha": 1.0, "beta": 1.0} for t in TOPOLOGIES}
        mock_redis = AsyncMock()

        async def fake_get(key):
            if key == "rl_policy:priors":
                return json.dumps(default_arms)
            return None

        mock_redis.get = AsyncMock(side_effect=fake_get)
        policy = RLPolicy(mock_redis)

        await policy.reward("pipeline", quality=0.5, cost_efficiency=0.5)

        mock_redis.incr.assert_called_with("rl_policy:total_tasks")


class TestLoadSave:
    @pytest.mark.asyncio
    async def test_load_from_redis(self):
        mock_redis = AsyncMock()
        priors = {"single": {"alpha": 5.0, "beta": 2.0}, "pipeline": {"alpha": 3.0, "beta": 1.0},
                  "supervisor": {"alpha": 1.0, "beta": 1.0}, "fanout": {"alpha": 1.0, "beta": 1.0},
                  "ensemble": {"alpha": 1.0, "beta": 1.0}}
        mock_redis.get = AsyncMock(return_value=json.dumps(priors))
        policy = RLPolicy(mock_redis)

        await policy.load()

        assert policy.arms["single"]["alpha"] == 5.0
        assert policy.arms["pipeline"]["beta"] == 1.0

    @pytest.mark.asyncio
    async def test_load_defaults_when_empty(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        policy = RLPolicy(mock_redis)

        await policy.load()

        for topo in TOPOLOGIES:
            assert policy.arms[topo] == {"alpha": 1.0, "beta": 1.0}

    @pytest.mark.asyncio
    async def test_save_to_redis(self):
        mock_redis = AsyncMock()
        policy = RLPolicy(mock_redis)
        policy.arms = {t: {"alpha": 1.0, "beta": 1.0} for t in TOPOLOGIES}

        await policy.save()

        mock_redis.set.assert_called_once()
        saved_json = mock_redis.set.call_args[0][1]
        saved = json.loads(saved_json)
        assert "single" in saved
        assert "pipeline" in saved
