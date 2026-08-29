"""Tests for the prompter module."""

import tempfile
from pathlib import Path

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
    prompt = build_prompt(issue)
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
    prompt = build_prompt(issue, token_budget=1000)
    assert "[truncated]" in prompt


def test_build_prompt_missing_source():
    issue = {
        "id": "stale-0",
        "type": "stale",
        "belief_id": "missing",
        "belief_text": "Gone",
        "source_files": ["/nonexistent/path.py"],
        "description": "Source changed",
    }
    prompt = build_prompt(issue)
    assert "file not found" in prompt
