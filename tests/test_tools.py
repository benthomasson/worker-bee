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


def test_powerful_allowlisted_commands_are_restricted(tmp_path):
    tools.set_workspace_root(tmp_path)

    clone = tools.execute_tool("run_command", {
        "command": "git clone https://example.invalid/repo.git ../checkout",
    })
    target = tools.execute_tool("run_command", {
        "command": "uv pip install --target ../packages example",
    })
    execute = tools.execute_tool("run_command", {
        "command": "find . -name '*.txt' -exec cat {} +",
    })

    assert "git subcommand is not allowed" in clone
    assert "uv write target is outside workspace" in target
    assert "find execution actions are not allowed" in execute


def test_allowlisted_read_only_commands_remain_available(tmp_path):
    tools.set_workspace_root(tmp_path)
    (tmp_path / "file.txt").write_text("hello")

    git_result = tools.execute_tool("run_command", {"command": "git status"})
    assert "git subcommand is not allowed" not in git_result
    assert "file.txt" in tools.execute_tool(
        "run_command", {"command": "find . -name '*.txt'"}
    )


def test_git_add_and_commit_are_allowed_safely(tmp_path):
    tools.set_workspace_root(tmp_path)
    (tmp_path / "file.txt").write_text("hello")

    add_result = tools.execute_tool("run_command", {"command": "git add file.txt"})
    commit_result = tools.execute_tool(
        "run_command", {"command": "git commit -m 'test commit'"}
    )

    assert "git subcommand is not allowed" not in add_result
    assert "git subcommand is not allowed" not in commit_result


def test_git_commit_escape_options_are_rejected(tmp_path):
    tools.set_workspace_root(tmp_path)

    assert "git commit option is not allowed" in tools.execute_tool(
        "run_command", {"command": "git commit --no-verify -m bad"}
    )
    assert "requires an explicit message" in tools.execute_tool(
        "run_command", {"command": "git commit"}
    )


def test_uv_run_pytest_is_allowed_but_arbitrary_uv_run_is_not(tmp_path):
    tools.set_workspace_root(tmp_path)

    pytest_result = tools.execute_tool(
        "run_command", {"command": "uv run pytest -q"}
    )
    arbitrary_result = tools.execute_tool(
        "run_command", {"command": "uv run python -c 'print(1)'"}
    )

    assert "arbitrary program execution is not allowed" not in pytest_result
    assert "arbitrary program execution is not allowed" in arbitrary_result


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
