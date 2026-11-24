"""Tests for the prompter module."""

import tempfile
from pathlib import Path

import pytest

from worker_bee.prompter import build_prompt


def test_build_prompt_basic():
    issue = {
        "id": "gated-0",
        "type": "gated",
        "belief_id": "test-belief",
        "belief_text": "The function does X",
        "source_files": [],
        "description": "Belief has OUT antecedent",
    }
    prompt = build_prompt(issue)
    assert "test-belief" in prompt
    assert "The function does X" in prompt
    assert "ANALYSIS" in prompt
    assert "FIX" in prompt


def test_build_prompt_with_source(tmp_path):
    src = tmp_path / "example.py"
    src.write_text("def hello():\n    return 'world'\n")

    issue = {
        "id": "stale-0",
        "type": "stale",
        "belief_id": "hello-returns-world",
        "belief_text": "hello() returns 'world'",
        "source_files": [str(src)],
        "description": "Source changed",
    }
    prompt = build_prompt(issue, workspace_root=tmp_path)
    assert "def hello():" in prompt
    assert "return 'world'" in prompt


def test_build_prompt_truncates_large_source(tmp_path):
    src = tmp_path / "big.py"
    src.write_text("x" * 200_000)

    issue = {
        "id": "stale-0",
        "type": "stale",
        "belief_id": "big-file",
        "belief_text": "Something about big file",
        "source_files": [str(src)],
        "description": "Source changed",
    }
    prompt = build_prompt(issue, token_budget=1000, workspace_root=tmp_path)
    assert "[truncated]" in prompt


def test_build_prompt_rejects_source_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("secret = True\n")
    issue = {
        "type": "stale", "belief_id": "outside", "belief_text": "",
        "source_files": [str(outside)], "description": "source",
    }
    with pytest.raises(ValueError, match="outside workspace"):
        build_prompt(issue, workspace_root=workspace)


def test_build_prompt_resolves_relative_source_in_workspace(tmp_path):
    source = tmp_path / "src.py"
    source.write_text("answer = 42\n")
    issue = {
        "type": "stale", "belief_id": "relative", "belief_text": "",
        "source_files": ["src.py"], "description": "source",
    }
    prompt = build_prompt(issue, workspace_root=tmp_path)
    assert "answer = 42" in prompt


def test_build_prompt_missing_source_inside_workspace(tmp_path):
    issue = {
        "id": "stale-0",
        "type": "stale",
        "belief_id": "missing",
        "belief_text": "Gone",
        "source_files": ["missing/path.py"],
        "description": "Source changed",
    }
    prompt = build_prompt(issue, workspace_root=tmp_path)
    assert "file not found" in prompt


def test_build_prompt_rejects_missing_absolute_source_outside_workspace(tmp_path):
    issue = {
        "type": "stale", "belief_id": "missing-outside", "belief_text": "",
        "source_files": [str(tmp_path.parent / "not-present.py")],
        "description": "source",
    }
    with pytest.raises(ValueError, match="outside workspace"):
        build_prompt(issue, workspace_root=tmp_path)


def test_build_prompt_rejects_symlink_source(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("secret = True\n")
    link = tmp_path / "link.py"
    link.symlink_to(source)
    issue = {
        "type": "stale", "belief_id": "symlink", "belief_text": "",
        "source_files": [str(link)], "description": "source",
    }
    with pytest.raises(ValueError, match="symlink"):
        build_prompt(issue, workspace_root=tmp_path)
