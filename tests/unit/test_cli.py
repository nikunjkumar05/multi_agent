"""Unit tests for cli/bamas_cli.py."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from cli.bamas_cli import _call_estimate, _colorise, _print_report, _run


# ── _colorise ──────────────────────────────────────────────────────────


class TestColorise:
    def test_applies_color_when_tty(self):
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = True
            result = _colorise("hello", "\033[92m")
        assert result == "\033[92mhello\033[0m"

    def test_no_color_when_not_tty(self):
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = False
            result = _colorise("hello", "\033[92m")
        assert result == "hello"


# ── _print_report ──────────────────────────────────────────────────────


class TestPrintReport:
    SAMPLE_DATA = {
        "risk_level": "LOW",
        "budget_usd": 0.50,
        "topology": "pipeline",
        "model_tiers": ["cheap", "standard"],
        "estimated_cost_usd": 0.001234,
        "budget_headroom_pct": 99.7,
        "rationale": "Plenty of budget",
        "alternatives_considered": [
            {"topology": "single", "reason": "Cheaper option"},
            {"topology": "ensemble", "reason": "Better quality"},
        ],
    }

    def test_json_mode_outputs_json(self, capsys):
        exit_code = _print_report(self.SAMPLE_DATA, "test task", as_json=True)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["risk_level"] == "LOW"
        assert exit_code == 0

    def test_formatted_mode_prints_report(self, capsys):
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = False
            mock_stdout.write = lambda s: None
            exit_code = _print_report(self.SAMPLE_DATA, "test task", as_json=False)
        # Just verify it doesn't crash and returns correct exit code
        assert exit_code == 0

    def test_high_risk_returns_2(self, capsys):
        data = {**self.SAMPLE_DATA, "risk_level": "HIGH"}
        exit_code = _print_report(data, "task", as_json=True)
        assert exit_code == 2

    def test_medium_risk_returns_1(self, capsys):
        data = {**self.SAMPLE_DATA, "risk_level": "MEDIUM"}
        exit_code = _print_report(data, "task", as_json=True)
        assert exit_code == 1

    def test_unknown_risk_returns_2(self, capsys):
        data = {**self.SAMPLE_DATA, "risk_level": "UNKNOWN"}
        exit_code = _print_report(data, "task", as_json=True)
        assert exit_code == 2

    def test_long_task_truncated(self, capsys):
        data = {**self.SAMPLE_DATA}
        long_task = "x" * 120
        exit_code = _print_report(data, long_task, as_json=True)
        assert exit_code == 0

    def test_no_alternatives(self, capsys):
        data = {**self.SAMPLE_DATA, "alternatives_considered": []}
        exit_code = _print_report(data, "task", as_json=True)
        assert exit_code == 0


# ── _call_estimate ─────────────────────────────────────────────────────


def _make_mock_client(response):
    """Build a mock httpx.AsyncClient that supports `async with` and `await client.post(...)`."""
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=response)
    return mock_client


class TestCallEstimate:
    async def test_success_without_token(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = lambda: None
        mock_response.json.return_value = {"risk_level": "LOW"}

        mock_client = _make_mock_client(mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await _call_estimate("test task", 0.5, "http://localhost:8000")

        assert result == {"risk_level": "LOW"}
        url = mock_client.post.call_args[0][0]
        assert url == "http://localhost:8000/estimate"

    async def test_success_with_token(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = lambda: None
        mock_response.json.return_value = {"risk_level": "LOW"}

        mock_client = _make_mock_client(mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client) as mock_cls:
            await _call_estimate("task", 1.0, "http://localhost:8000", token="abc123")

        # headers are passed to httpx.AsyncClient(...), not to client.post()
        _, kwargs = mock_cls.call_args
        assert kwargs.get("headers") == {"Authorization": "Bearer abc123"}

    async def test_timeout_passed(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = lambda: None
        mock_response.json.return_value = {}

        mock_client = _make_mock_client(mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client) as mock_cls:
            await _call_estimate("task", 1.0, "http://localhost:8000", timeout=60.0)

        _, kwargs = mock_cls.call_args
        assert kwargs.get("timeout") == 60.0

    async def test_server_trailing_slash_stripped(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = lambda: None
        mock_response.json.return_value = {}

        mock_client = _make_mock_client(mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await _call_estimate("task", 1.0, "http://localhost:8000/")

        url = mock_client.post.call_args[0][0]
        assert url == "http://localhost:8000/estimate"


# ── _run ───────────────────────────────────────────────────────────────


class TestRun:
    async def test_connect_error_returns_3(self, capsys):
        with patch("cli.bamas_cli._call_estimate", side_effect=httpx.ConnectError("refused")):
            exit_code = await _run("task", 0.5, "http://localhost:8000", False)
        assert exit_code == 3
        captured = capsys.readouterr()
        assert "Cannot connect" in captured.err

    async def test_http_status_error_returns_3(self, capsys):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"detail": "Internal error"}
        exc = httpx.HTTPStatusError("500", request=MagicMock(), response=mock_resp)
        with patch("cli.bamas_cli._call_estimate", side_effect=exc):
            exit_code = await _run("task", 0.5, "http://localhost:8000", False)
        assert exit_code == 3

    async def test_generic_error_returns_3(self, capsys):
        with patch("cli.bamas_cli._call_estimate", side_effect=RuntimeError("boom")):
            exit_code = await _run("task", 0.5, "http://localhost:8000", False)
        assert exit_code == 3

    async def test_success_returns_risk_exit_code(self):
        data = {
            "risk_level": "LOW",
            "budget_usd": 0.5,
            "topology": "single",
            "model_tiers": ["cheap"],
            "estimated_cost_usd": 0.001,
            "budget_headroom_pct": 99.8,
            "rationale": "Low cost",
            "alternatives_considered": [],
        }
        with patch("cli.bamas_cli._call_estimate", return_value=data):
            exit_code = await _run("task", 0.5, "http://localhost:8000", True)
        assert exit_code == 0
