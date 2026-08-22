"""
Unit tests for BAMAS tools: code_executor, db_query, file_ops.

Tests security protections, error handling, and basic functionality.
"""
import os
import sqlite3
import tempfile
from pathlib import Path

from agent.tools.code_executor import CodeExecutor
from agent.tools.db_query import DBQueryTool
from agent.tools.file_ops import FileListTool, FileReadTool, FileWriteTool, _safe_path


class TestCodeExecutor:
    """Tests for the code_executor tool."""

    def setup_method(self):
        self.tool = CodeExecutor()

    def test_execute_simple_code(self):
        result = self.tool.execute(code="print(2 + 2)")
        assert result.success is True
        assert "4" in result.output

    def test_execute_empty_code(self):
        result = self.tool.execute(code="")
        assert result.success is False
        assert "No code" in result.error

    def test_execute_syntax_error(self):
        result = self.tool.execute(code="def foo(")
        assert result.success is False

    def test_execute_runtime_error(self):
        result = self.tool.execute(code="x = 1 / 0")
        assert result.success is False

    def test_execute_no_output(self):
        result = self.tool.execute(code="x = 1 + 2")
        assert result.success is True
        assert "no output" in result.output.lower()

    def test_execute_with_args(self):
        code = "import sys; print(sys.argv[1:])"
        result = self.tool.execute(code=code, args=["-l", "12"])
        assert result.success is True
        assert "-l" in result.output
        assert "12" in result.output

    def test_execute_timeout(self):
        code = "import time; time.sleep(20)"
        result = self.tool.execute(code=code)
        assert result.success is False
        assert "timed out" in result.error.lower()

    def test_temp_file_cleanup(self):
        """Verify temp files are cleaned up after execution."""
        before = set(Path(tempfile.gettempdir()).glob("*.py"))
        self.tool.execute(code="print('hello')")
        after = set(Path(tempfile.gettempdir()).glob("*.py"))
        # Should not leave new temp files
        assert len(after - before) == 0


class TestDBQueryTool:
    """Tests for the db_query tool."""

    def setup_method(self):
        self.tool = DBQueryTool()
        # Create a temp database
        self.db_path = tempfile.mktemp(suffix=".db")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
            conn.execute("INSERT INTO users VALUES (1, 'Alice')")
            conn.execute("INSERT INTO users VALUES (2, 'Bob')")

    def teardown_method(self):
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
        except PermissionError:
            pass  # Windows file locking

    def test_select_query(self):
        result = self.tool.execute(query="SELECT * FROM users", database=self.db_path)
        assert result.success is True
        assert len(result.output) == 2

    def test_empty_query(self):
        result = self.tool.execute(query="")
        assert result.success is False

    def test_blocked_delete(self):
        result = self.tool.execute(query="DELETE FROM users", database=self.db_path)
        assert result.success is False
        assert "Blocked" in result.error

    def test_blocked_drop(self):
        result = self.tool.execute(query="DROP TABLE users", database=self.db_path)
        assert result.success is False
        assert "Blocked" in result.error

    def test_blocked_insert(self):
        result = self.tool.execute(query="INSERT INTO users VALUES (3, 'Charlie')", database=self.db_path)
        assert result.success is False

    def test_blocked_update(self):
        result = self.tool.execute(query="UPDATE users SET name = 'X'", database=self.db_path)
        assert result.success is False

    def test_blocked_create(self):
        result = self.tool.execute(query="CREATE TABLE evil (id INTEGER)", database=self.db_path)
        assert result.success is False

    def test_invalid_table(self):
        result = self.tool.execute(query="SELECT * FROM nonexistent", database=self.db_path)
        assert result.success is False

    def test_case_insensitive_block(self):
        result = self.tool.execute(query="delete from users", database=self.db_path)
        assert result.success is False


class TestFileOps:
    """Tests for file_read, file_write, file_list tools."""

    def setup_method(self):
        self.read_tool = FileReadTool()
        self.write_tool = FileWriteTool()
        self.list_tool = FileListTool()

    def test_safe_path_valid(self):
        path = _safe_path("test.txt")
        assert path is not None
        assert "workspace" in str(path)

    def test_safe_path_traversal(self):
        path = _safe_path("../../etc/passwd")
        assert path is None

    def test_safe_path_absolute(self):
        path = _safe_path("/etc/passwd")
        assert path is None

    def test_safe_path_dotdot(self):
        path = _safe_path("subdir/../../etc/passwd")
        assert path is None

    def test_write_and_read(self):
        result = self.write_tool.execute(path="test_write.txt", content="hello world")
        assert result.success is True

        result = self.read_tool.execute(path="test_write.txt")
        assert result.success is True
        assert result.output == "hello world"

        # Cleanup
        os.remove(Path("./workspace/test_write.txt"))

    def test_read_nonexistent(self):
        result = self.read_tool.execute(path="nonexistent.txt")
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_read_traversal(self):
        result = self.read_tool.execute(path="../../etc/passwd")
        assert result.success is False

    def test_write_traversal(self):
        result = self.write_tool.execute(path="../../evil.txt", content="bad")
        assert result.success is False

    def test_list_directory(self):
        result = self.list_tool.execute()
        assert result.success is True
        assert isinstance(result.output, list)

    def test_list_nonexistent(self):
        result = self.list_tool.execute(path="nonexistent_dir")
        assert result.success is False
