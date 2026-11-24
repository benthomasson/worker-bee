"""Process-boundary behavior for the command-line interface."""

import shutil
import subprocess
import sys
from unittest.mock import patch

import pytest

from worker_bee.cli import main


def test_no_command_prints_help_and_succeeds(capsys):
    assert main([]) == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out


def test_run_forwards_progress_options(tmp_path):
    with patch("worker_bee.pipeline.run_pipeline") as run:
        assert main(["run", "brain.db", "--progress-file", str(tmp_path / "p.json"), "--resume"]) == 0
    run.assert_called_once_with(
        "brain.db", model="ollama:qwen3.8:27b", dry_run=False,
        requests_per_second=None, progress_file=str(tmp_path / "p.json"), resume=True,
    )


def test_run_help_mentions_progress_options(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--progress-file" in output
    assert "--resume" in output


def test_argparse_errors_keep_status_two():
    with pytest.raises(SystemExit) as exc_info:
        main(["extract"])
    assert exc_info.value.code == 2


def test_operational_errors_return_status_one_without_traceback(capsys, tmp_path):
    result = main(["extract", str(tmp_path / "missing.db")])

    assert result == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("Error: ")
    assert "Database not found" in captured.err
    assert "Traceback" not in captured.err


def test_mcp_configuration_errors_return_status_one(capsys):
    result = main(["edit", "task", "--mcp", "https://example.test/mcp"])

    assert result == 1
    captured = capsys.readouterr()
    assert "--mcp-token or REASONS_MCP_TOKEN env var required" in captured.err
    assert "Traceback" not in captured.err


def test_module_entrypoint_returns_operational_status_without_traceback(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "worker_bee.cli", "extract", str(tmp_path / "missing.db")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr.startswith("Error: ")
    assert "Traceback" not in result.stderr


def test_module_entrypoint_preserves_version_exit():
    result = subprocess.run(
        [sys.executable, "-m", "worker_bee.cli", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.startswith("worker-bee ")
    assert result.stderr == ""


def test_installed_entrypoint_returns_operational_status(tmp_path):
    executable = shutil.which("worker-bee")
    if executable is None:
        pytest.skip("worker-bee console script is not installed")

    result = subprocess.run(
        [executable, "extract", str(tmp_path / "missing.db")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr.startswith("Error: ")
    assert "Traceback" not in result.stderr
