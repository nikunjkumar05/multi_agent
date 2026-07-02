import os

import pytest
from unittest.mock import AsyncMock, MagicMock

from core.rl_policy import RLPolicy, TOPOLOGIES


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
        pipe = MagicMock()
        pipe.execute = AsyncMock(return_value=[{}, {}, {}, {}, {}, b"2"])
        mock_redis.pipeline = AsyncMock(return_value=pipe)
        policy = RLPolicy(mock_redis)
        result = await policy.select_topology("test task", "healthy")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_topology_above_min_tasks(self):
        mock_redis = AsyncMock()
        default_arm = {"alpha": b"1.0", "beta": b"1.0"}
        pipe = MagicMock()
        pipe.execute = AsyncMock(return_value=[default_arm, default_arm, default_arm, default_arm, default_arm, b"10"])
        mock_redis.pipeline = AsyncMock(return_value=pipe)

        policy = RLPolicy(mock_redis)
        result = await policy.select_topology("Write a code function", "healthy")
        assert result in TOPOLOGIES


class TestRewardUpdate:
    @pytest.mark.asyncio
    async def test_good_reward_increases_alpha(self, tmp_path):
        mock_redis = AsyncMock()
        mock_redis.hincrbyfloat = AsyncMock()
        mock_redis.incr = AsyncMock()
        policy = RLPolicy(mock_redis, persist_path=str(tmp_path / "rl.json"))

        await policy.reward("pipeline", quality=1.0, cost_efficiency=1.0)

        mock_redis.hincrbyfloat.assert_called_once_with(
            "rl_policy:arm:pipeline", "alpha", 1.0
        )
        assert policy.arms["pipeline"]["alpha"] == 2.0

    @pytest.mark.asyncio
    async def test_bad_reward_increases_beta(self, tmp_path):
        mock_redis = AsyncMock()
        mock_redis.hincrbyfloat = AsyncMock()
        mock_redis.incr = AsyncMock()
        policy = RLPolicy(mock_redis, persist_path=str(tmp_path / "rl.json"))

        await policy.reward("pipeline", quality=0.0, cost_efficiency=0.0)

        mock_redis.hincrbyfloat.assert_called_once_with(
            "rl_policy:arm:pipeline", "beta", 1.0
        )
        assert policy.arms["pipeline"]["beta"] == 2.0

    @pytest.mark.asyncio
    async def test_increments_task_count(self, tmp_path):
        mock_redis = AsyncMock()
        mock_redis.hincrbyfloat = AsyncMock()
        mock_redis.incr = AsyncMock()
        policy = RLPolicy(mock_redis, persist_path=str(tmp_path / "rl.json"))

        await policy.reward("pipeline", quality=0.5, cost_efficiency=0.5)

        mock_redis.incr.assert_called_once_with("rl_policy:total_tasks")
        assert policy.total_tasks == 1


class TestLoadSave:
    @pytest.mark.asyncio
    async def test_load_from_redis(self):
        mock_redis = AsyncMock()
        arm_single = {"alpha": b"5.0", "beta": b"2.0"}
        arm_pipeline = {"alpha": b"3.0", "beta": b"1.0"}
        arm_default = {"alpha": b"1.0", "beta": b"1.0"}
        pipe = MagicMock()
        pipe.execute = AsyncMock(return_value=[
            arm_single, arm_pipeline, arm_default, arm_default, arm_default,
            b"42"
        ])
        mock_redis.pipeline = AsyncMock(return_value=pipe)
        policy = RLPolicy(mock_redis)

        await policy.load()

        assert policy.arms["single"]["alpha"] == 5.0
        assert policy.arms["pipeline"]["beta"] == 1.0
        assert policy.total_tasks == 42

    @pytest.mark.asyncio
    async def test_load_defaults_when_empty(self, tmp_path):
        mock_redis = AsyncMock()
        pipe = MagicMock()
        pipe.execute = AsyncMock(return_value=[{}, {}, {}, {}, {}, None])
        mock_redis.pipeline = AsyncMock(return_value=pipe)
        policy = RLPolicy(mock_redis, persist_path=str(tmp_path / "missing.json"))

        await policy.load()

        for topo in TOPOLOGIES:
            assert policy.arms[topo] == {"alpha": 1.0, "beta": 1.0}


class TestFilePersistence:
    def test_save_and_load(self, tmp_path):
        from core.rl_policy import RLPolicy
        mock_redis = AsyncMock()
        persist = str(tmp_path / "rl.json")

        policy = RLPolicy(mock_redis, persist_path=persist)
        policy.arms = {
            "single": {"alpha": 3.0, "beta": 1.5},
            "pipeline": {"alpha": 2.0, "beta": 1.0},
            "supervisor": {"alpha": 1.0, "beta": 1.0},
            "fanout": {"alpha": 1.0, "beta": 1.0},
            "ensemble": {"alpha": 1.0, "beta": 1.0},
        }
        policy.total_tasks = 15
        policy._save_to_file()

        import json
        with open(persist, "r") as f:
            data = json.load(f)
        assert data["total_tasks"] == 15
        assert data["arms"]["single"]["alpha"] == 3.0

        policy2 = RLPolicy(mock_redis, persist_path=persist)
        policy2._load_from_file()
        assert policy2.arms["single"]["alpha"] == 3.0
        assert policy2.total_tasks == 15

    def test_load_from_file_when_missing(self, tmp_path):
        from core.rl_policy import RLPolicy
        mock_redis = AsyncMock()
        policy = RLPolicy(mock_redis, persist_path=str(tmp_path / "missing.json"))
        policy._load_from_file()
        assert policy.arms == {}

    def test_atomic_write(self, tmp_path):
        from core.rl_policy import RLPolicy
        mock_redis = AsyncMock()
        persist = str(tmp_path / "rl.json")

        policy = RLPolicy(mock_redis, persist_path=persist)
        policy.arms = {t: {"alpha": 1.0, "beta": 1.0} for t in TOPOLOGIES}
        policy.total_tasks = 0
        policy._save_to_file()

        assert not os.path.exists(persist + ".tmp")
        assert os.path.exists(persist)

    @pytest.mark.asyncio
    async def test_reward_persists_to_file(self, tmp_path):
        from core.rl_policy import RLPolicy
        mock_redis = AsyncMock()
        mock_redis.hincrbyfloat = AsyncMock()
        mock_redis.incr = AsyncMock()
        persist = str(tmp_path / "rl.json")

        policy = RLPolicy(mock_redis, persist_path=persist)
        policy.arms = {t: {"alpha": 1.0, "beta": 1.0} for t in TOPOLOGIES}
        policy.total_tasks = 0

        await policy.reward("pipeline", quality=1.0, cost_efficiency=1.0)

        assert os.path.exists(persist)
        import json
        with open(persist, "r") as f:
            data = json.load(f)
        assert data["arms"]["pipeline"]["alpha"] == 2.0
        assert data["total_tasks"] == 1
