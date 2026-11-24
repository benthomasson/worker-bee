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


def test_verify_forwards_retract_option():
    with patch("worker_bee.verifier.verify_belief") as verify:
        verify.return_value.verified = None
        assert main(["verify", "brain.db", "belief", "--retract"]) == 0
    verify.assert_called_once_with(
        "brain.db", "belief", project_dir=None, model="ollama:qwen3.8:27b",
        dry_run=False, verbose=False, max_turns=20, num_ctx=None,
        retract_inaccurate=True,
    )


def test_verify_single_defaults_to_evidence_only():
    with patch("worker_bee.verifier.verify_belief") as verify:
        verify.return_value.verified = None
        assert main(["verify", "brain.db", "belief"]) == 0
    assert verify.call_args.kwargs["retract_inaccurate"] is False


@pytest.mark.parametrize("argv, expected", [
    (["verify", "brain.db"], False),
    (["verify", "brain.db", "--retract"], True),
])
def test_verify_batch_forwards_retract_option(argv, expected):
    with patch("worker_bee.verifier.verify_unverified", return_value=[]) as verify:
        assert main(argv) == 0
    assert verify.call_args.kwargs["retract_inaccurate"] is expected


def test_verify_help_describes_retract_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["verify", "--help"])
    assert exc_info.value.code == 0
    assert "--retract" in capsys.readouterr().out


def test_run_forwards_progress_options(tmp_path):
    with patch("worker_bee.pipeline.run_pipeline") as run:
        assert main(["run", "brain.db", "--progress-file", str(tmp_path / "p.json"), "--resume"]) == 0
    run.assert_called_once_with(
        "brain.db", model="ollama:qwen3.8:27b", dry_run=False,
        requests_per_second=None, progress_file=str(tmp_path / "p.json"), resume=True,
    )


def test_run_uses_default_checkpoint_and_resumes(tmp_path):
    db = str(tmp_path / "brain.db")
    with patch("worker_bee.pipeline.run_pipeline") as run:
        assert main(["run", db]) == 0
    run.assert_called_once_with(
        db, model="ollama:qwen3.8:27b", dry_run=False,
        requests_per_second=None, progress_file=f"{db}.progress.json", resume=True,
    )


def test_dry_run_does_not_create_default_checkpoint(tmp_path):
    db = str(tmp_path / "brain.db")
    with patch("worker_bee.pipeline.run_pipeline") as run:
        assert main(["run", db, "--dry-run"]) == 0
    run.assert_called_once_with(
        db, model="ollama:qwen3.8:27b", dry_run=True,
        requests_per_second=None, progress_file=None, resume=False,
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
