from core.optimizer import VALID_TOPOLOGIES, OptimizerDecision, rule_based_select_topology


class TestRuleBasedSelectTopology:
    def test_code_task_returns_pipeline(self):
        assert rule_based_select_topology("Write a Python function") == "pipeline"

    def test_research_task_returns_supervisor(self):
        assert rule_based_select_topology("Explain how TCP works") == "supervisor"

    def test_data_task_returns_fanout(self):
        assert rule_based_select_topology("Analyze these datasets") == "fanout"

    def test_verify_task_returns_ensemble(self):
        assert rule_based_select_topology("Verify this proof") == "ensemble"

    def test_simple_task_returns_single(self):
        assert rule_based_select_topology("What is 2+2?") == "single"

    def test_create_keyword_returns_pipeline(self):
        assert rule_based_select_topology("Create a web scraper") == "pipeline"

    def test_compare_keyword_returns_supervisor(self):
        assert rule_based_select_topology("Compare React and Vue") == "supervisor"

    def test_why_keyword_returns_supervisor(self):
        assert rule_based_select_topology("Why is the sky blue") == "supervisor"

    def test_audit_keyword_returns_ensemble(self):
        assert rule_based_select_topology("Audit this code for security") == "ensemble"

    def test_bulk_keyword_returns_fanout(self):
        assert rule_based_select_topology("Bulk process these files") == "fanout"


class TestOptimizerDecision:
    def test_valid_decision(self):
        d = OptimizerDecision(
            topology="pipeline",
            model_tiers={"planner": "standard", "executor": "standard", "validator": "cheap", "judge": "standard"},
            rationale="code task",
            alternatives_considered=[],
        )
        assert d.topology in VALID_TOPOLOGIES

    def test_alternatives_normalization_from_dict(self):
        d = OptimizerDecision(
            topology="pipeline",
            model_tiers={"planner": "standard", "executor": "standard", "validator": "cheap", "judge": "standard"},
            rationale="test",
            alternatives_considered={"single": "too simple", "supervisor": "not research"},
        )
        assert isinstance(d.alternatives_considered, list)
        assert len(d.alternatives_considered) == 2

    def test_alternatives_normalization_from_list(self):
        d = OptimizerDecision(
            topology="pipeline",
            model_tiers={"planner": "standard", "executor": "standard", "validator": "cheap", "judge": "standard"},
            rationale="test",
            alternatives_considered=[{"topology": "single", "reason": "too simple"}],
        )
        assert isinstance(d.alternatives_considered, list)
        assert len(d.alternatives_considered) == 1

    def test_valid_topologies_set(self):
        assert VALID_TOPOLOGIES == {"single", "supervisor", "pipeline", "fanout", "ensemble", "feedback"}
