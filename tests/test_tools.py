from pathlib import Path

from worker_bee import tools


def test_filesystem_tools_are_confined(tmp_path):
    tools.set_workspace_root(tmp_path)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")

    assert "outside workspace" in tools.execute_tool("read_file", {"path": "../outside.txt"})
    result = tools.execute_tool(
        "write_file", {"path": "../created.txt", "content": "nope"}
    )
    assert "outside workspace" in result
    assert not (tmp_path.parent / "created.txt").exists()


def test_run_command_uses_allowlist_and_no_shell(tmp_path):
    tools.set_workspace_root(tmp_path)

    assert tools.execute_tool("run_command", {"command": "pwd"}).strip() == str(tmp_path)
    assert "not allowlisted" in tools.execute_tool(
        "run_command", {"command": "sh -c 'touch escaped'"}
    )
    assert "not allowlisted" in tools.execute_tool(
        "run_command", {"command": "pwd; echo escaped"}
    )


def test_run_command_rejects_paths_outside_workspace(tmp_path):
    tools.set_workspace_root(tmp_path)
    result = tools.execute_tool("run_command", {"command": "cat ../secret.txt"})
    assert "outside workspace" in result


def test_workspace_symlink_cannot_escape(tmp_path):
    tools.set_workspace_root(tmp_path)
    outside = tmp_path.parent / "real.txt"
    outside.write_text("secret")
    link = tmp_path / "link.txt"
    link.symlink_to(outside)

    result = tools.execute_tool("read_file", {"path": "link.txt"})
    assert "outside workspace" in result


def test_glob_rejects_absolute_and_parent_patterns(tmp_path):
    tools.set_workspace_root(tmp_path)
    (tmp_path / "inside.txt").write_text("inside")
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")

    absolute = tools.execute_tool("glob", {"pattern": str(outside)})
    parent = tools.execute_tool("glob", {"pattern": "../*.txt"})

    assert "must be relative" in absolute
    assert "cannot contain '..'" in parent
    assert "outside.txt" not in parent


def test_glob_filters_symlink_matches_outside_workspace(tmp_path):
    tools.set_workspace_root(tmp_path)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")
    (tmp_path / "link.txt").symlink_to(outside)

    result = tools.execute_tool("glob", {"pattern": "*.txt"})

    assert result == "No files found"
