"""Tool definitions and execution for code-editing loops.

Adapted from claude_code_python/tools.py. Each tool has a JSON schema
(sent to the LLM) and a Python implementation that runs when the model
requests it.
"""

from __future__ import annotations

import glob as glob_module
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


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
            "Run an approved command in the workspace and return its output. "
            "Commands run without a shell; shell operators, pipelines, and "
            "redirections are not supported. Quote arguments with standard shell quoting."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command and arguments to run (no shell syntax)",
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
            "a persistent project-level notes file (survives context eviction "
            "AND is reloaded at the start of future sessions), and also in the "
            "in-memory session store. Use this to record intermediate findings, "
            "decisions, or plans so you don't have to re-derive them if earlier "
            "turns are evicted — or if you come back next session. "
            "For durable conclusions you want to keep long-term, prefer "
            "add_belief instead: notes are scratch, beliefs are knowledge."
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
    {
        "name": "list_notes",
        "description": (
            "List all persistent notes from the project-level notes store. "
            "Returns the notes with their index (0-based), timestamp, and "
            "text. Use this at the start of a session or when you need to "
            "recall what you or a previous session wrote down. "
            "Indexes are stable: use them with update_note and delete_note."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max notes to return (default: 50, most recent last)",
                },
            },
        },
    },
    {
        "name": "update_note",
        "description": (
            "Replace an existing persistent note at the given index. "
            "Use list_notes first to find the index. Returns confirmation "
            "with the new note text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "0-based index of the note to replace (from list_notes)",
                },
                "note": {
                    "type": "string",
                    "description": "The replacement note text",
                },
            },
            "required": ["index", "note"],
        },
    },
    {
        "name": "delete_note",
        "description": (
            "Remove a persistent note at the given index from the notes store. "
            "Use list_notes first to find the index. Other notes' indexes shift "
            "down after deletion."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "0-based index of the note to delete (from list_notes)",
                },
            },
            "required": ["index"],
        },
    },
]

CHAT_TOOLS = [
    {
        "name": "ask_user_question",
        "description": (
            "Ask the user a clarifying question and wait for their response. "
            "Use this when you need more information to proceed — e.g. which "
            "approach to take, what file to focus on, or confirmation before "
            "a destructive action. The user's answer is returned as the tool result."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user",
                },
            },
            "required": ["question"],
        },
    },
]

BELIEF_TOOLS = [
    {
        "name": "show_belief",
        "description": (
            "Look up a belief by ID from the reasons database. Returns the belief's "
            "text, truth value, source, justifications, dependents, and metadata."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The belief ID (e.g. 'wgpu-backend-is-mandatory')",
                },
            },
            "required": ["id"],
        },
    },
    {
        "name": "search_beliefs",
        "description": (
            "Search for beliefs by keyword. Searches both belief IDs and text. "
            "Returns matching beliefs with their truth values."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword to search for in belief IDs and text",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default: 20)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_blockers",
        "description": (
            "List beliefs that are blocking other beliefs (gated issues). "
            "These are IN beliefs whose presence keeps other beliefs OUT. "
            "This shows the most actionable issues in the belief network."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "add_belief",
        "description": (
            "Add a new belief to the local belief database. The belief ID must "
            "not already exist in either the local or hive database. Use this to "
            "record findings, conclusions, or new claims discovered during the task."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Unique belief ID in kebab-case (e.g. 'dispatcher-retries-on-429')",
                },
                "text": {
                    "type": "string",
                    "description": "The belief text — a single atomic claim",
                },
                "source": {
                    "type": "string",
                    "description": "Provenance — where this belief came from (e.g. a file path)",
                },
            },
            "required": ["id", "text"],
        },
    },
]


MAX_READ_LINES = 500

DEFAULT_NOTES_PATH = ".worker-bee/notes.jsonl"
MAX_PERSISTENT_NOTES = 100  # cap the persisted note history; oldest are rotated out


class NoteStore:
    """Persistent project-level notes store.

    Notes are stored as JSONL (one JSON object per line) in a shared file
    (default: .worker-bee/notes.jsonl). The same store can be reused across
    chat rounds and sessions so notes survive beyond a single edit loop.

    Each entry has the shape: {"ts": <iso-timestamp>, "note": <text>}.
    """

    def __init__(self, path: str | None = DEFAULT_NOTES_PATH,
                 max_notes: int = MAX_PERSISTENT_NOTES):
        self.path = Path(path) if path else None
        self.max_notes = max_notes
        self._notes: list[dict] = self._load()

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def _load(self) -> list[dict]:
        if not self.path:
            return []
        if not self.path.exists():
            return []
        notes: list[dict] = []
        try:
            with open(self.path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if isinstance(entry, dict) and "note" in entry:
                            notes.append(entry)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []
        return notes

    def _flush(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w") as f:
                for entry in self._notes:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            # Persistence is best-effort; in-memory copy still works.
            print(f"warning: could not persist notes: {e}", file=sys.stderr)

    def list(self) -> list[dict]:
        return list(self._notes)

    def add(self, note: str) -> int:
        """Append a note. Returns its index in the store."""
        if not note or not note.strip():
            raise ValueError("note text is empty")
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": note,
        }
        self._notes.append(entry)
        # Cap the history: rotate out the oldest entries beyond the cap.
        if self.max_notes and len(self._notes) > self.max_notes:
            self._notes = self._notes[-self.max_notes:]
        self._flush()
        return len(self._notes) - 1

    def get(self, index: int) -> dict:
        if index < 0 or index >= len(self._notes):
            raise IndexError(
                f"note index {index} out of range (0..{len(self._notes) - 1})"
            )
        return self._notes[index]

    def update(self, index: int, note: str) -> dict:
        if not note or not note.strip():
            raise ValueError("note text is empty")
        entry = self.get(index)
        entry["note"] = note
        entry["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._flush()
        return entry

    def delete(self, index: int) -> int:
        self.get(index)  # validates range
        self._notes.pop(index)
        self._flush()
        return len(self._notes)


class BeliefStore:
    """Layered belief store: local brain (read/write) over hive (read-only).

    Reads union both layers. Writes go to local only. ID conflicts
    between layers are errors — no silent shadowing.
    """

    def __init__(self, local_path: str | None = None, hive_path: str | None = None):
        self.local_path = local_path
        self.hive_path = hive_path
        if local_path:
            self._ensure_db(local_path)
        if local_path and hive_path:
            self._check_id_conflicts()

    @staticmethod
    def _ensure_db(db_path: str) -> None:
        """Initialize the database schema if it doesn't exist yet."""
        import os
        if not os.path.exists(db_path):
            from reasons.api import init_db
            init_db(db_path=db_path)
        else:
            from reasons.storage import Storage
            store = Storage(db_path)
            store.close()

    @staticmethod
    def _get_ids(db_path: str) -> set[str]:
        import sqlite3
        conn = sqlite3.connect(db_path)
        try:
            return {r[0] for r in conn.execute("SELECT id FROM nodes").fetchall()}
        except sqlite3.OperationalError:
            return set()
        finally:
            conn.close()

    def _check_id_conflicts(self):
        local_ids = self._get_ids(self.local_path) if self.local_path else set()
        hive_ids = self._get_ids(self.hive_path) if self.hive_path else set()
        overlap = local_ids & hive_ids
        if overlap:
            raise ValueError(f"Belief ID conflict between local and hive: {overlap}")

    @property
    def db_paths(self) -> list[str]:
        """All database paths, local first."""
        paths = []
        if self.local_path:
            paths.append(self.local_path)
        if self.hive_path:
            paths.append(self.hive_path)
        return paths

    @property
    def has_any(self) -> bool:
        return bool(self.local_path or self.hive_path)

    def id_exists_in_hive(self, belief_id: str) -> bool:
        if not self.hive_path:
            return False
        import sqlite3
        conn = sqlite3.connect(self.hive_path)
        row = conn.execute("SELECT 1 FROM nodes WHERE id = ?", (belief_id,)).fetchone()
        conn.close()
        return row is not None

    def id_exists_in_local(self, belief_id: str) -> bool:
        if not self.local_path:
            return False
        import sqlite3
        conn = sqlite3.connect(self.local_path)
        row = conn.execute("SELECT 1 FROM nodes WHERE id = ?", (belief_id,)).fetchone()
        conn.close()
        return row is not None


_belief_store = BeliefStore()

# Tool execution is deliberately confined to the project workspace.  Commands
# are also run without a shell and must use an approved executable.
_workspace_root = Path.cwd().resolve()
_ALLOWED_COMMANDS = frozenset({
    "cat", "find", "git", "grep", "ls", "python", "pytest", "pwd", "ruff", "uv",
})


def set_workspace_root(path: str | os.PathLike[str] | None) -> None:
    """Set the root directory visible to filesystem and command tools.

    All filesystem paths are resolved beneath this directory.  ``None`` uses
    the current working directory.  Symlinks are resolved before the boundary
    check so a link cannot escape the workspace.
    """
    global _workspace_root
    root = Path.cwd() if path is None else Path(path)
    _workspace_root = root.expanduser().resolve()
    if not _workspace_root.is_dir():
        raise ValueError(f"workspace root is not a directory: {path}")


def _workspace_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.expanduser().resolve()
    else:
        resolved = (_workspace_root / candidate).resolve()
    try:
        resolved.relative_to(_workspace_root)
    except ValueError as exc:
        raise ValueError(f"path is outside workspace: {path}") from exc
    return resolved


def set_belief_db(db_path: str | None, brain_path: str | None = None) -> None:
    """Configure the belief store.

    db_path: the hive (read-only project beliefs)
    brain_path: local brain (read/write session beliefs)

    If only db_path is given and no brain_path, db_path is used as
    both the hive and the writable layer (backwards compatible).
    """
    global _belief_store
    if brain_path:
        _belief_store = BeliefStore(local_path=brain_path, hive_path=db_path)
    elif db_path:
        _belief_store = BeliefStore(local_path=db_path)
    else:
        _belief_store = BeliefStore()


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
    elif name == "show_belief":
        return _show_belief(tool_input["id"])
    elif name == "search_beliefs":
        return _search_beliefs(tool_input["query"], tool_input.get("limit", 20))
    elif name == "list_blockers":
        return _list_blockers()
    elif name == "add_belief":
        return _add_belief(tool_input["id"], tool_input["text"], tool_input.get("source", ""))
    elif name == "list_notes":
        return _list_notes(limit=tool_input.get("limit", 50))
    elif name == "update_note":
        return _update_note(tool_input.get("index", -1), tool_input.get("note", ""))
    elif name == "delete_note":
        return _delete_note(tool_input.get("index", -1))
    else:
        return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Persistent notes
#
# A single NoteStore can be active at a time. It's set by the editor/chat
# layer via set_notes_file() before the loop begins. When None (persistence
# disabled), the tools fall back to returning a helpful message so the LLM
# knows the feature is off for this run.
# ---------------------------------------------------------------------------

_active_notes: NoteStore | None = None


def set_notes_file(path: str | None) -> None:
    """Configure the active notes store.

    path:
        - a string path (e.g. '.worker-bee/notes.jsonl') enables persistence
        - None disables persistence (in-memory notes only — tools still work,
          just don't survive the process)
    """
    global _active_notes
    _active_notes = NoteStore(path=path) if path else None


def get_notes_store() -> NoteStore | None:
    return _active_notes


def _write_note(note: str) -> str:
    if not note or not note.strip():
        return "Error: note text is empty"
    if _active_notes is None:
        # No persistent store configured. Still accept the call — return a
        # message so the LLM knows persistence is off for this run.
        return "Note stored in-memory only (no --notes path configured for this session). " \
               "It will not be visible to future sessions."
    try:
        idx = _active_notes.add(note)
        return f"Note saved. ({idx + 1} total) [persisted to {_active_notes.path}]"
    except Exception as e:
        return f"Error writing note: {e}"


def _list_notes(limit: int = 50) -> str:
    if _active_notes is None or not _active_notes.enabled:
        return "No persistent notes configured for this session."
    notes = _active_notes.list()
    if not notes:
        return "No notes yet."
    if limit > 0:
        notes = notes[-limit:]
        offset = 0  # we keep the original indexes by computing them below
    # Compute real indexes into the full store so update/delete keep working.
    full = _active_notes.list()
    lines = []
    for i, entry in enumerate(full):
        if limit > 0 and i < len(full) - limit:
            continue
        ts = entry.get("ts", "")
        lines.append(f"[{i}] {ts}  {entry['note']}")
    header = f"{len(full)} persistent note(s) in {_active_notes.path}:"
    return header + "\n" + "\n".join(lines)


def _update_note(index: int, note: str) -> str:
    if _active_notes is None or not _active_notes.enabled:
        return "Error: no persistent notes store configured for this session"
    if not note or not note.strip():
        return "Error: note text is empty"
    try:
        entry = _active_notes.update(int(index), note)
        return f"Updated note [{index}]: {entry['note'][:120]}"
    except IndexError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error updating note: {e}"


def _delete_note(index: int) -> str:
    if _active_notes is None or not _active_notes.enabled:
        return "Error: no persistent notes store configured for this session"
    try:
        remaining = _active_notes.delete(int(index))
        return f"Deleted note. ({remaining} remaining)"
    except IndexError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error deleting note: {e}"


def _read_file(path, offset=None, limit=None):
    try:
        path = _workspace_path(path)
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
        path = _workspace_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def _edit_file(path, old_string, new_string):
    try:
        path = _workspace_path(path)
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
        path = _workspace_path(path)
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
    # Do not let glob's own path semantics bypass the workspace boundary:
    # os.path.join() discards its first argument for an absolute second
    # argument, and a pattern containing ``..`` can walk back above it.
    if not isinstance(pattern, str) or not pattern:
        return "Error: glob pattern must be a non-empty string"
    if os.path.isabs(pattern):
        return "Error: glob pattern must be relative to the workspace"
    if any(component == ".." for component in Path(pattern).parts):
        return "Error: glob pattern cannot contain '..'"

    try:
        path = _workspace_path(path)
    except ValueError as e:
        return f"Error: {e}"
    full_pattern = os.path.join(str(path), pattern)
    matches = glob_module.glob(full_pattern, recursive=True)
    skip = {".git", ".venv", "node_modules", "__pycache__", "target"}
    filtered = []
    for m in matches:
        parts = Path(m).parts
        if any(p in skip or (p.startswith(".") and p != ".") for p in parts):
            continue
        # glob follows directory symlinks and can return a symlink to a file
        # outside the workspace. Resolve every match before exposing it.
        try:
            resolved = Path(m).resolve()
            resolved.relative_to(_workspace_root)
        except (OSError, ValueError):
            continue
        if resolved.is_file():
            filtered.append(os.path.relpath(m, path))

    if not filtered:
        return "No files found"
    return "\n".join(sorted(filtered))


def _run_command(command):
    try:
        argv = shlex.split(command)
    except ValueError as e:
        return f"Error: invalid command quoting: {e}"
    if not argv:
        return "Error: command is empty"
    if any(token in {";", "&&", "||", "|", ">", ">>", "<", "&"} for token in argv):
        return "Error: shell syntax is not supported"

    executable = Path(argv[0]).name
    if executable not in _ALLOWED_COMMANDS or argv[0] != executable:
        return f"Error: command not allowlisted: {argv[0]}"

    # Do not let command arguments turn a confined process into an escape hatch.
    # Validate directory options and the positional path arguments of commands
    # that can read or execute files.  Resolution also rejects symlinks leaving
    # the workspace.
    path_options = {"-C", "--directory", "--cwd", "--work-tree", "--git-dir"}
    path_args = set()
    if executable in {"cat", "find", "ls", "pytest", "ruff"}:
        path_args = {i for i, arg in enumerate(argv[1:], 1) if not arg.startswith("-")}
    elif executable == "grep":
        non_options = [i for i, arg in enumerate(argv[1:], 1) if not arg.startswith("-")]
        path_args = set(non_options[1:])  # first non-option is the pattern
    elif executable == "python":
        if "-c" in argv or "-m" in argv or "-" in argv:
            return "Error: inline/module Python execution is not allowed"
        path_args = {i for i, arg in enumerate(argv[1:], 1) if not arg.startswith("-")}

    for index, arg in enumerate(argv[1:], 1):
        value = None
        if arg.startswith(("/", "../", "./", "~")):
            value = arg
        elif arg in path_options and index + 1 < len(argv):
            value = argv[index + 1]
        elif any(arg.startswith(option + "=") for option in path_options):
            value = arg.split("=", 1)[1]
        elif index in path_args:
            value = arg
        if value is not None:
            try:
                _workspace_path(value)
            except ValueError as e:
                return f"Error: command path is outside workspace: {e}"

    try:
        result = subprocess.run(
            argv, cwd=_workspace_root, shell=False,
            capture_output=True, text=True, timeout=30,
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


def _show_belief(belief_id):
    if not _belief_store.has_any:
        return "Error: no belief database configured for this session"
    try:
        from reasons.api import show_node
        for db_path in _belief_store.db_paths:
            try:
                node = show_node(belief_id, db_path=db_path)
                break
            except (KeyError, Exception):
                continue
        else:
            return f"Belief '{belief_id}' not found"
        lines = [
            f"ID: {node['id']}",
            f"Text: {node['text']}",
            f"Truth value: {node['truth_value']}",
            f"Source: {node.get('source', '')}",
        ]
        if node.get("verified_at"):
            lines.append(f"Verified at: {node['verified_at']}")
        if node.get("reviewed_at"):
            lines.append(f"Reviewed at: {node['reviewed_at']}")
        meta = node.get("metadata") or {}
        if meta.get("verify_result"):
            vr = meta["verify_result"]
            lines.append(f"Verify result: verified={vr.get('verified')}, confidence={vr.get('confidence')}")
            if vr.get("evidence"):
                lines.append(f"Evidence: {vr['evidence']}")
        if node.get("justifications"):
            lines.append("Justifications:")
            for j in node["justifications"]:
                ant = ", ".join(j.get("antecedents", []))
                out = ", ".join(j.get("outlist", []))
                parts = []
                if ant:
                    parts.append(f"antecedents=[{ant}]")
                if out:
                    parts.append(f"outlist=[{out}]")
                lines.append(f"  {j.get('type', '?')}: {', '.join(parts)}")
        if node.get("dependents"):
            lines.append(f"Dependents: {', '.join(node['dependents'])}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def _search_beliefs(query, limit=20):
    if not _belief_store.has_any:
        return "Error: no belief database configured for this session"
    try:
        import sqlite3
        all_rows = []
        seen_ids = set()
        pattern = f"%{query}%"
        for db_path in _belief_store.db_paths:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, text, truth_value FROM nodes "
                "WHERE id LIKE ? OR text LIKE ? "
                "ORDER BY truth_value DESC, id",
                (pattern, pattern),
            ).fetchall()
            conn.close()
            for r in rows:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    all_rows.append(r)
        all_rows.sort(key=lambda r: (r["truth_value"] != "IN", r["id"]))
        all_rows = all_rows[:limit]
        if not all_rows:
            return f"No beliefs matching '{query}'"
        lines = []
        for r in all_rows:
            lines.append(f"[{r['truth_value']}] {r['id']}: {r['text'][:120]}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def _list_blockers():
    if not _belief_store.has_any:
        return "Error: no belief database configured for this session"
    try:
        from reasons.api import list_gated
        all_blockers = {}
        total_blocker_count = 0
        total_gated_count = 0
        for db_path in _belief_store.db_paths:
            result = list_gated(db_path=db_path)
            blockers = result.get("blockers", {})
            for bid, info in blockers.items():
                if bid not in all_blockers:
                    all_blockers[bid] = info
            total_blocker_count += result.get("blocker_count", 0)
            total_gated_count += result.get("gated_count", 0)
        if not all_blockers:
            return "No gated beliefs found."
        lines = [f"{total_blocker_count} blocker(s) gating {total_gated_count} belief(s):", ""]
        for bid, info in all_blockers.items():
            lines.append(f"[{bid}] {info['text'][:120]}")
            for g in info["gated"]:
                lines.append(f"  blocks: {g['id']}: {g['text'][:100]}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def _add_belief(belief_id, text, source=""):
    if not _belief_store.has_any:
        return "Error: no belief database configured for this session"
    if not _belief_store.local_path:
        return "Error: no writable belief database configured"
    if _belief_store.id_exists_in_hive(belief_id):
        return f"Error: belief '{belief_id}' already exists in hive. Use a different ID."
    if _belief_store.id_exists_in_local(belief_id):
        return f"Error: belief '{belief_id}' already exists in local database."
    try:
        from reasons.api import add_node
        add_node(belief_id, text, source=source, db_path=_belief_store.local_path)
        return f"Added belief '{belief_id}' [IN]"
    except Exception as e:
        return f"Error adding belief: {e}"
