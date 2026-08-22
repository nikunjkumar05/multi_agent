from agent.nodes.planner import analyze_task_complexity


class TestTaskComplexity:
    def test_trivial_greeting(self):
        assert analyze_task_complexity("hi") == 1

    def test_trivial_hello(self):
        assert analyze_task_complexity("hello") == 1

    def test_trivial_thanks(self):
        assert analyze_task_complexity("thanks") == 1

    def test_trivial_ok(self):
        assert analyze_task_complexity("ok") == 1

    def test_simple_math(self):
        assert analyze_task_complexity("what is 2+2") == 1

    def test_simple_math_calculate(self):
        assert analyze_task_complexity("calculate 15 * 7") == 1

    def test_simple_define(self):
        assert analyze_task_complexity("define recursion") == 1

    def test_medium_write_code(self):
        assert analyze_task_complexity("write a fibonacci function") == 2

    def test_medium_explain(self):
        assert analyze_task_complexity("explain how HTTP works") == 2

    def test_medium_compare(self):
        assert analyze_task_complexity("compare Python and JavaScript") == 2

    def test_medium_create(self):
        assert analyze_task_complexity("create a REST API endpoint") == 2

    def test_complex_multi_phase(self):
        assert analyze_task_complexity("build a web scraper and then deploy it to AWS") == 3

    def test_complex_many_actions(self):
        assert analyze_task_complexity("research AI trends, compare approaches, and write a report") == 3

    def test_complex_long_task(self):
        task = "write a comprehensive analysis of the current state of quantum computing, " * 5
        assert analyze_task_complexity(task) == 3

    def test_medium_two_words_with_action(self):
        assert analyze_task_complexity("write code") == 2

    def test_trivial_two_words_no_action(self):
        assert analyze_task_complexity("yes please") == 1
