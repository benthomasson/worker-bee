"""Tool definitions and execution for code-editing loops.

Adapted from claude_code_python/tools.py. Each tool has a JSON schema
(sent to the LLM) and a Python implementation that runs when the model
requests it.
"""

from __future__ import annotations

import glob as glob_module
import os
import re
import subprocess


TOOLS = [
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file. For large files, use offset and limit "
            "to read specific line ranges. Files over 500 lines are automatically "
            "truncated — use offset/limit to read further."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to read",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-based, default: 1)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (default: 500)",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Make a surgical edit to a file by replacing an exact string match. "
            "The old_string must appear exactly once in the file. "
            "Use read_file first to see the current contents. "
            "Prefer this over write_file when modifying existing files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to edit",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact string to find and replace (must be unique in the file)",
                },
                "new_string": {
                    "type": "string",
                    "description": "The string to replace it with",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "grep",
        "description": (
            "Search for a regex pattern across files in a directory. "
            "Returns matching lines with file paths and line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "The directory to search in (defaults to current directory)",
                    "default": ".",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "glob",
        "description": (
            "Find files matching a glob pattern (e.g. '**/*.py' for all Python files). "
            "Returns a list of matching file paths."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The glob pattern to match",
                },
                "path": {
                    "type": "string",
                    "description": "The directory to search in (defaults to current directory)",
                    "default": ".",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "run_command",
        "description": (
            "Run a shell command and return its output. "
            "Use for running tests, git operations, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run",
                }
            },
            "required": ["command"],
        },
    },
    {
        "name": "list_memory",
        "description": (
            "List your previous tool calls from this session. Returns an index "
            "of tool call IDs, names, and short summaries. Use this when you need "
            "to recall what you did earlier — old tool calls may have been evicted "
            "from context but their results are still stored."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "retrieve_memory",
        "description": (
            "Retrieve the full result of a previous tool call by its ID. "
            "Use after list_memory to pull back a specific result that may have "
            "been evicted from context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "The tool call ID from list_memory",
                },
            },
            "required": ["id"],
        },
    },
    {
        "name": "write_note",
        "description": (
            "Write a note to yourself for later reference. Notes are stored in "
            "session memory and survive context eviction. Use this to record "
            "intermediate findings, decisions, or plans so you don't have to "
            "re-derive them if earlier turns are evicted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "The note to store",
                },
            },
            "required": ["note"],
        },
    },
]


MAX_READ_LINES = 500


def execute_tool(name: str, tool_input: dict) -> str:
    """Execute a tool by name. Returns the result as a string."""
    if name == "read_file":
        return _read_file(
            tool_input["path"],
            offset=tool_input.get("offset"),
            limit=tool_input.get("limit"),
        )
    elif name == "write_file":
        return _write_file(tool_input["path"], tool_input["content"])
    elif name == "edit_file":
        return _edit_file(tool_input["path"], tool_input["old_string"], tool_input["new_string"])
    elif name == "grep":
        return _grep(tool_input["pattern"], tool_input.get("path", "."))
    elif name == "glob":
        return _glob(tool_input["pattern"], tool_input.get("path", "."))
    elif name == "run_command":
        return _run_command(tool_input["command"])
    else:
        return f"Unknown tool: {name}"


def _read_file(path, offset=None, limit=None):
    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading file: {e}"

    total = len(lines)
    start = max(0, (offset or 1) - 1)
    max_lines = limit or MAX_READ_LINES

    selected = lines[start:start + max_lines]
    numbered = []
    for i, line in enumerate(selected, start + 1):
        numbered.append(f"{i}\t{line.rstrip()}")
    result = "\n".join(numbered)

    remaining = total - (start + len(selected))
    if remaining > 0:
        result += (
            f"\n\n... ({remaining} more lines. "
            f"Use offset={start + len(selected) + 1} to continue reading. "
            f"Total: {total} lines.)"
        )

    return result


def _write_file(path, content):
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def _edit_file(path, old_string, new_string):
    try:
        with open(path, "r") as f:
            content = f.read()

        count = content.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {path}"
        if count > 1:
            return f"Error: old_string appears {count} times in {path} — must be unique"

        new_content = content.replace(old_string, new_string, 1)
        with open(path, "w") as f:
            f.write(new_content)
        return f"Successfully edited {path}"
    except Exception as e:
        return f"Error editing file: {e}"


def _grep(pattern, path):
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex: {e}"

    matches = []
    skip = {".git", ".venv", "node_modules", "__pycache__", "target"}
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, "r", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append(f"{filepath}:{i}: {line.rstrip()}")
            except (OSError, IsADirectoryError):
                continue

    if not matches:
        return "No matches found"
    return "\n".join(matches[:100])


def _glob(pattern, path):
    full_pattern = os.path.join(path, pattern)
    matches = glob_module.glob(full_pattern, recursive=True)
    skip = {".git", ".venv", "node_modules", "__pycache__", "target"}
    filtered = []
    for m in matches:
        parts = m.split(os.sep)
        if any(p in skip or (p.startswith(".") and p != ".") for p in parts):
            continue
        if os.path.isfile(m):
            filtered.append(os.path.relpath(m, path))

    if not filtered:
        return "No files found"
    return "\n".join(sorted(filtered))


def _run_command(command):
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        if result.returncode != 0:
            output += f"\n(exit code: {result.returncode})"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds"
    except Exception as e:
        return f"Error running command: {e}"
