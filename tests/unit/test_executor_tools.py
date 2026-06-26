import pytest
from agent.nodes.executor import detect_task_type, _build_executor_prompt


class TestDetectTaskType:
    def test_code_task(self):
        assert detect_task_type("write a python function") == "code"

    def test_codeImplement(self):
        assert detect_task_type("implement a sorting algorithm") == "code"

    def test_math_task(self):
        assert detect_task_type("calculate the factorial of 20") == "math"

    def test_mathSolve(self):
        assert detect_task_type("solve the equation x^2 + 5x + 6 = 0") == "math"

    def test_research_task(self):
        assert detect_task_type("explain how neural networks work") == "research"

    def test_researchCompare(self):
        assert detect_task_type("compare React and Vue frameworks") == "research"

    def test_creative_task(self):
        assert detect_task_type("write a story about a robot") == "creative"

    def test_data_task(self):
        assert detect_task_type("analyze this dataset for trends") == "data"

    def test_general_task(self):
        assert detect_task_type("tell me a joke") == "general"

    def test_data_overrides_code(self):
        assert detect_task_type("analyze data and write code") == "data"


class TestBuildExecutorPrompt:
    def test_includes_tool_names(self):
        prompt = _build_executor_prompt("general")
        assert "code_executor" in prompt
        assert "web_search" in prompt

    def test_code_prompt(self):
        prompt = _build_executor_prompt("code")
        assert "code_executor" in prompt
        assert "software engineer" in prompt

    def test_math_prompt(self):
        prompt = _build_executor_prompt("math")
        assert "mathematical expert" in prompt

    def test_research_prompt(self):
        prompt = _build_executor_prompt("research")
        assert "web_search" in prompt
        assert "researcher" in prompt
