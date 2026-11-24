"""Tests for process-wide workspace configuration."""

from pathlib import Path

import pytest

from worker_bee.workspace import get_workspace_root, set_workspace_root


def test_workspace_root_is_canonical_and_shared(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    try:
        configured = set_workspace_root(root / ".." / root.name)

        assert configured == root.resolve()
        assert get_workspace_root() == root.resolve()
    finally:
        set_workspace_root(None)


def test_workspace_root_rejects_missing_directory(tmp_path):
    try:
        with pytest.raises(ValueError, match="not a directory"):
            set_workspace_root(tmp_path / "missing")
    finally:
        set_workspace_root(None)


def test_workspace_root_none_resets_to_current_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert set_workspace_root(None) == Path.cwd().resolve()
    assert get_workspace_root() == Path.cwd().resolve()
