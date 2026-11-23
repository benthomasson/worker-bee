from pathlib import Path

import pytest

from worker_bee import tools


def test_execute_tool_exposes_status_without_breaking_string_contract(tmp_path):
    tools.set_workspace_root(tmp_path)
    success = tools.execute_tool("run_command", {"command": "pwd"})
    failure = tools.execute_tool("run_command", {"command": "not-allowlisted"})

    assert isinstance(success, str)
    assert success.ok is True
    assert success.error is None
    assert failure.ok is False
    assert failure.error == str(failure)


def test_execute_tool_classifies_unknown_and_non_prefix_failures(tmp_path):
    tools.set_workspace_root(tmp_path)

    unknown = tools.execute_tool("missing_tool", {})
    invalid_regex = tools.execute_tool("grep", {"pattern": "["})
    nonzero = tools.execute_tool("run_command", {"command": "git status"})

    assert unknown == "Unknown tool: missing_tool"
    assert unknown.ok is False
    assert invalid_regex.ok is False
    assert nonzero.ok is False


def test_execute_tool_validates_required_fields_and_types(tmp_path):
    tools.set_workspace_root(tmp_path)

    missing = tools.execute_tool("read_file", {})
    wrong_type = tools.execute_tool("read_file", {"path": 42})
    valid = tools.execute_tool("read_file", {"path": "missing.txt"})

    assert missing.ok is False
    assert "missing required field(s): path" in missing
    assert wrong_type.ok is False
    assert "field 'path' must be string" in wrong_type
    assert valid.ok is False  # the tool ran and reported the missing file
    assert "Error reading file" in valid


def test_execute_tool_rejects_non_object_input():
    result = tools.execute_tool("read_file", None)

    assert result.ok is False
    assert "input must be an object" in result


def test_all_advertised_tool_schemas_validate_required_fields_and_types():
    for tool in tools.TOOLS + tools.CHAT_TOOLS + tools.BELIEF_TOOLS:
        schema = tool["input_schema"]
        name = tool["name"]
        missing = tools._validate_tool_input(name, {})
        if schema.get("required"):
            assert missing is not None, name
        else:
            assert missing is None, name
        for field, property_schema in schema.get("properties", {}).items():
            if "type" not in property_schema:
                continue
            valid_values = {
                "string": "value",
                "integer": 1,
                "number": 1.5,
                "boolean": True,
                "object": {},
                "array": [],
            }
            value = valid_values[property_schema["type"]]
            assert tools._validate_schema_value(value, property_schema, field) is None
            invalid = {
                "string": 1,
                "integer": True,
                "number": True,
                "boolean": "true",
                "object": [],
                "array": {},
            }[property_schema["type"]]
            assert tools._validate_schema_value(invalid, property_schema, field) is not None


def test_validator_supports_documented_nested_constraints():
    schema = {
        "type": "object",
        "required": ["config", "values"],
        "properties": {
            "config": {
                "type": "object",
                "required": ["mode"],
                "properties": {"mode": {"type": "string", "enum": ["fast", "safe"]}},
            },
            "values": {
                "type": "array",
                "items": {"type": "number", "minimum": 0, "maximum": 10},
            },
        },
    }
    valid = {"config": {"mode": "safe"}, "values": [0, 2.5, 10], "extra": True}

    assert tools._validate_schema_value(valid, schema, "input") is None
    assert tools._validate_schema_value(
        {"config": {"mode": "unsafe"}, "values": [1]}, schema, "input"
    ) is not None
    assert tools._validate_schema_value(
        {"config": {"mode": "safe"}, "values": [-1]}, schema, "input"
    ) is not None


def test_tool_schema_registry_rejects_duplicate_names(monkeypatch):
    monkeypatch.setattr(tools, "TOOLS", tools.TOOLS + [tools.TOOLS[0]])

    with pytest.raises(ValueError, match="duplicate tool name.*read_file"):
        tools._tool_schemas()


@pytest.mark.parametrize("exception", [
    ValueError("malformed argument"),
    RuntimeError("backend unavailable"),
    OSError("filesystem unavailable"),
])
def test_execute_tool_converts_helper_exceptions_to_failures(monkeypatch, exception):
    def fail(_name, _input):
        raise exception

    monkeypatch.setattr(tools, "_execute_tool", fail)
    result = tools.execute_tool("list_memory", {})

    assert result.ok is False
    assert str(exception) in result


def test_execute_tool_converts_registry_validation_errors_to_failures(monkeypatch):
    monkeypatch.setattr(tools, "TOOLS", tools.TOOLS + [tools.TOOLS[0]])

    result = tools.execute_tool("list_memory", {})

    assert result.ok is False
    assert "duplicate tool name" in result


@pytest.mark.parametrize(
    ("tool_name", "rendering"),
    [
        ("show_belief", "Belief 'missing' not found"),
        ("retract_belief", "Error: belief 'missing' not found"),
        ("supersede_belief", "Error: belief 'missing' not found"),
        ("propose_retraction", "Belief 'missing' not found"),
        ("propose_supersede", "Belief 'missing' not found"),
    ],
)
def test_belief_missing_id_renderings_are_failures(tool_name, rendering):
    result = tools._as_tool_result(rendering, tool_name=tool_name)

    assert result.ok is False
    assert result.error == rendering


def test_execute_tool_classifies_mcp_failure(monkeypatch):
    class FakeMCP:
        def show(self, _belief_id):
            raise RuntimeError("backend unavailable")

        mcp_url = "http://example.invalid"

    monkeypatch.setattr("worker_bee.mcp_beliefs.get_mcp_store", lambda: FakeMCP())
    result = tools.execute_tool("show_belief", {"id": "example"})

    assert result.ok is False
    assert "Error (MCP): backend unavailable" == result


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


def test_python_m_pytest_is_allowed_but_arbitrary_modules_are_not(tmp_path):
    tools.set_workspace_root(tmp_path)

    pytest_result = tools.execute_tool(
        "run_command", {"command": "python -m pytest -q"}
    )
    module_result = tools.execute_tool(
        "run_command", {"command": "python -m http.server 8000"}
    )
    inline_result = tools.execute_tool(
        "run_command", {"command": "python -c 'print(1)'"}
    )

    assert "arbitrary module execution is not allowed" not in pytest_result
    assert "arbitrary module execution is not allowed" in module_result
    assert "inline Python execution is not allowed" in inline_result


def test_python_m_pytest_paths_are_confined(tmp_path):
    tools.set_workspace_root(tmp_path)

    result = tools.execute_tool(
        "run_command", {"command": "python -m pytest ../tests"}
    )
    assert "outside workspace" in result


def test_cargo_build_and_test_are_allowed_but_install_is_not(tmp_path):
    tools.set_workspace_root(tmp_path)

    build = tools.execute_tool("run_command", {"command": "cargo build"})
    check = tools.execute_tool("run_command", {"command": "cargo clippy --all-targets"})

    assert "cargo subcommand is not allowed" not in build
    assert "cargo subcommand is not allowed" not in check
    assert "cargo subcommand is not allowed" in tools.execute_tool(
        "run_command", {"command": "cargo install ripgrep"}
    )
    assert "cargo subcommand is not allowed" in tools.execute_tool(
        "run_command", {"command": "cargo add serde"}
    )
    assert "cargo toolchain overrides are not allowed" in tools.execute_tool(
        "run_command", {"command": "cargo +nightly build"}
    )


def test_cargo_manifest_path_is_confined(tmp_path):
    tools.set_workspace_root(tmp_path)

    result = tools.execute_tool(
        "run_command", {"command": "cargo build --manifest-path ../Cargo.toml"}
    )
    assert "outside workspace" in result


def test_workspace_symlink_cannot_escape(tmp_path):
    tools.set_workspace_root(tmp_path)
    outside = tmp_path.parent / "real.txt"
    outside.write_text("secret")
    link = tmp_path / "link.txt"
    link.symlink_to(outside)

    result = tools.execute_tool("read_file", {"path": "link.txt"})
    assert "outside workspace" in result


def test_read_file_does_not_follow_symlinked_file_inside_workspace(tmp_path):
    tools.set_workspace_root(tmp_path)
    target = tmp_path / "target.txt"
    target.write_text("secret")
    link = tmp_path / "link.txt"
    link.symlink_to(target)

    result = tools.execute_tool("read_file", {"path": "link.txt"})
    assert "Error reading file" in result


def test_edit_file_does_not_follow_symlinked_file_inside_workspace(tmp_path):
    tools.set_workspace_root(tmp_path)
    target = tmp_path / "target.txt"
    target.write_text("secret")
    link = tmp_path / "link.txt"
    link.symlink_to(target)

    result = tools.execute_tool(
        "edit_file", {"path": "link.txt", "old_string": "secret", "new_string": "changed"}
    )
    assert "Error editing file" in result
    assert target.read_text() == "secret"


def test_atomic_write_does_not_follow_symlinked_parent(tmp_path):
    """The final write must use the checked directory, not a replacement link."""
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError):
        tools._atomic_write_text(linked / "created.txt", "must stay confined")

    assert not (outside / "created.txt").exists()


def test_atomic_write_rejects_unsupported_directory_flags(tmp_path, monkeypatch):
    monkeypatch.delattr(tools.os, "O_NOFOLLOW", raising=False)

    with pytest.raises(OSError, match="O_DIRECTORY and O_NOFOLLOW"):
        tools._atomic_write_text(tmp_path / "created.txt", "unsupported")

    assert not (tmp_path / "created.txt").exists()


def test_atomic_write_rejects_unsupported_descriptor_operations(tmp_path, monkeypatch):
    supported = set(tools.os.supports_dir_fd)
    supported.discard(tools.os.replace)
    supported.discard(tools.os.rename)
    monkeypatch.setattr(tools.os, "supports_dir_fd", supported)

    with pytest.raises(OSError, match="descriptor-relative"):
        tools._atomic_write_text(tmp_path / "created.txt", "unsupported")

    assert not (tmp_path / "created.txt").exists()


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


def test_grep_filters_file_symlinks_outside_workspace(tmp_path):
    tools.set_workspace_root(tmp_path)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret needle")
    (tmp_path / "link.txt").symlink_to(outside)

    result = tools.execute_tool("grep", {"pattern": "needle"})

    assert result == "No matches found"


def test_grep_does_not_traverse_directory_symlinks(tmp_path):
    tools.set_workspace_root(tmp_path)
    outside = tmp_path.parent / "outside-dir"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret needle")
    (tmp_path / "linked-dir").symlink_to(outside, target_is_directory=True)

    result = tools.execute_tool("grep", {"pattern": "needle"})

    assert result == "No matches found"


def test_grep_reads_file_symlink_into_read_only_root(tmp_path):
    tools.set_workspace_root(tmp_path)
    readonly = tmp_path.parent / "readonly"
    readonly.mkdir()
    (readonly / "allowed.txt").write_text("allowed needle")
    tools.set_read_only_paths([readonly])
    (tmp_path / "link.txt").symlink_to(readonly / "allowed.txt")

    result = tools.execute_tool("grep", {"pattern": "needle"})

    assert "link.txt:1: allowed needle" in result


def test_grep_closes_root_descriptor_when_traversal_fails(tmp_path, monkeypatch):
    tools.set_workspace_root(tmp_path)
    opened = []
    closed = []
    real_open_root = tools._open_root_for_read
    real_close = tools.os.close

    def recording_open_root(root):
        fd = real_open_root(root)
        opened.append(fd)
        return fd

    def recording_close(fd):
        closed.append(fd)
        return real_close(fd)

    monkeypatch.setattr(tools, "_open_root_for_read", recording_open_root)
    monkeypatch.setattr(tools.os, "close", recording_close)
    monkeypatch.setattr(tools.os, "walk", lambda path: (_ for _ in ()).throw(OSError("walk failed")))

    with pytest.raises(OSError, match="walk failed"):
        tools._grep("needle", ".")

    assert opened
    assert opened[0] in closed


def test_grep_uses_descriptor_safe_open(tmp_path, monkeypatch):
    tools.set_workspace_root(tmp_path)
    (tmp_path / "inside.txt").write_text("safe needle")

    real_open = tools.os.open
    real_fdopen = tools.os.fdopen
    open_calls = []
    fdopen_calls = []

    def recording_open(*args, **kwargs):
        open_calls.append((args, kwargs))
        return real_open(*args, **kwargs)

    def recording_fdopen(*args, **kwargs):
        fdopen_calls.append((args, kwargs))
        return real_fdopen(*args, **kwargs)

    # _open_parent_directory checks descriptor-relative support by identity;
    # include the spy in the advertised support set while it is installed.
    monkeypatch.setattr(tools.os, "open", recording_open)
    monkeypatch.setattr(tools.os, "fdopen", recording_fdopen)
    monkeypatch.setattr(
        tools.os, "supports_dir_fd",
        set(tools.os.supports_dir_fd) | {recording_open},
    )
    result = tools.execute_tool("grep", {"pattern": "needle"})

    assert "inside.txt:1: safe needle" in result
    assert any(
        kwargs.get("dir_fd") is not None
        and args[1] & tools.os.O_NOFOLLOW
        for args, kwargs in open_calls
        if len(args) > 1
    )
    assert fdopen_calls


def test_grep_ignores_undecodable_bytes(tmp_path):
    tools.set_workspace_root(tmp_path)
    (tmp_path / "binary.bin").write_bytes(b"before \xff needle\n")

    result = tools.execute_tool("grep", {"pattern": "needle"})

    assert "binary.bin:1: before  needle" in result
