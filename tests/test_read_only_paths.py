"""Read-only paths: read tools may cross into them, writes may not."""

from pathlib import Path

import pytest

from worker_bee import tools


@pytest.fixture
def env(tmp_path):
    """Workspace + a sibling 'other' project containing readable source."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    other = tmp_path / "other-project"
    other.mkdir()
    (other / "lib.py").write_text("def hello():\n    return 42\n")
    (other / "src" / "deep.py").parent.mkdir()
    (other / "src" / "deep.py").write_text("x = 1\n")
    tools.set_workspace_root(workspace)
    return workspace, other


def test_read_only_path_grants_read_access(env):
    workspace, other = env
    tools.set_read_only_paths(other)

    result = tools.execute_tool("read_file", {"path": str(other / "lib.py")})
    assert "return 42" in result

    grep = tools.execute_tool("grep", {"pattern": "def hello", "path": str(other)})
    assert "lib.py:1" in grep

    glob = tools.execute_tool("glob", {"pattern": "**/*.py", "path": str(other)})
    assert any("lib.py" in line for line in glob.splitlines())


def test_read_file_rejects_paths_outside_any_root(env):
    workspace, other = env
    stray = workspace.parent / "stray.txt"
    stray.write_text("secret")
    tools.set_read_only_paths(other)

    assert "outside workspace" in tools.execute_tool("read_file", {"path": str(stray)})
    assert "outside workspace" in tools.execute_tool(
        "grep", {"pattern": "x", "path": str(workspace.parent)}
    )


def test_write_file_rejects_read_only_roots(env):
    workspace, other = env
    tools.set_read_only_paths(other)

    result = tools.execute_tool(
        "write_file", {"path": str(other / "hacked.txt"), "content": "nope"}
    )
    assert "outside workspace" in result
    assert not (other / "hacked.txt").exists()

    result = tools.execute_tool(
        "edit_file",
        {"path": str(other / "lib.py"), "old_string": "42", "new_string": "0"},
    )
    assert "outside workspace" in result
    assert "42" in (other / "lib.py").read_text()


def test_run_command_stays_confined_to_workspace(env):
    workspace, other = env
    tools.set_read_only_paths(other)

    assert "outside workspace" in tools.execute_tool(
        "run_command", {"command": f"cat {other / 'lib.py'}"}
    )


def test_set_read_only_paths_validates_input(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    tools.set_workspace_root(workspace)

    # Nonexistent path
    with pytest.raises(ValueError, match="cannot be resolved|not a directory"):
        tools.set_read_only_paths(tmp_path / "does-not-exist")

    # A file, not a directory
    stray = tmp_path / "file.txt"
    stray.write_text("x")
    with pytest.raises(ValueError, match="not a directory"):
        tools.set_read_only_paths(stray)

    # Inside the workspace — redundant
    inner = workspace / "sub"
    inner.mkdir()
    with pytest.raises(ValueError, match="inside the workspace"):
        tools.set_read_only_paths(inner)

    # Accepts single path and list; clears on None
    other = tmp_path / "other"
    other.mkdir()
    tools.set_read_only_paths(other)
    tools.set_read_only_paths([other])
    tools.set_read_only_paths(None)


def test_set_workspace_root_clears_read_only_paths(env):
    workspace, other = env
    tools.set_read_only_paths(other)
    result = tools.execute_tool("read_file", {"path": str(other / "lib.py")})
    assert "return 42" in result

    # A new workspace resets the grant.
    other_ws = workspace.parent / "other-ws"
    other_ws.mkdir()
    tools.set_workspace_root(other_ws)
    assert "outside workspace" in tools.execute_tool(
        "read_file", {"path": str(other / "lib.py")}
    )


def test_glob_lists_outside_matches_as_absolute(env):
    workspace, other = env
    tools.set_read_only_paths(other)

    result = tools.execute_tool("glob", {"pattern": "**/*.py", "path": str(other)})
    lines = result.splitlines()
    assert any(line == str(other / "lib.py") for line in lines)
    assert any(line == str(other / "src" / "deep.py") for line in lines)


def test_multiple_read_only_roots(env, tmp_path):
    workspace, other = env
    second = tmp_path / "second"
    second.mkdir()
    (second / "note.md").write_text("hello")
    tools.set_read_only_paths([other, second])

    assert "return 42" in tools.execute_tool("read_file", {"path": str(other / "lib.py")})
    assert "hello" in tools.execute_tool("read_file", {"path": str(second / "note.md")})


def test_relative_paths_resolve_against_workspace_not_read_only_roots(env):
    workspace, other = env
    tools.set_read_only_paths(other)

    # A relative path is anchored at the workspace, never at a read-only
    # root (even if one happens to contain a file with the same name).
    (workspace / "lib.py").write_text("ws copy\n")
    assert "ws copy" in tools.execute_tool("read_file", {"path": "lib.py"})
    assert "outside workspace" in tools.execute_tool("read_file", {"path": "../missing.txt"})
