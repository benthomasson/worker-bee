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
import tempfile
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
    {
        "name": "retract_belief",
        "description": (
            "Retract an existing belief (mark it OUT) in the local belief database. "
            "This cascades: any beliefs that depend on it are re-evaluated and may "
            "go OUT too. Use this when a belief has turned out to be wrong or "
            "outdated. Provide a short reason for the retraction (recommended)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The belief ID to retract (e.g. 'notes-not-persisted-to-disk')",
                },
                "reason": {
                    "type": "string",
                    "description": "Why the belief is being retracted (stored in metadata for the audit trail)",
                },
            },
            "required": ["id"],
        },
    },
    {
        "name": "supersede_belief",
        "description": (
            "Replace an existing belief with a new, corrected version. Creates a new "
            "belief (auto-ID'd '<old_id>-v2' unless you supply one) that supersedes the "
            "old one. The old belief goes OUT while the new one is IN; supersession is "
            "reversible. Use this when a belief needs to be updated rather than deleted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The belief ID to supersede (the old one)",
                },
                "text": {
                    "type": "string",
                    "description": "The corrected belief text — a single atomic claim that replaces the old one",
                },
                "new_id": {
                    "type": "string",
                    "description": "Optional ID for the new belief (defaults to '<id>-v2' if it would not collide)",
                },
            },
            "required": ["id", "text"],
        },
    },
    {
        "name": "propose_retraction",
        "description": (
            "Propose retracting a hive belief (mark it OUT) WITHOUT changing any "
            "truth value. Creates a pending proposal in the hive that a human (or "
            "policy) must accept before the retraction applies. Use this for beliefs "
            "that live in the read-only hive — retract_belief only works on the local "
            "brain. Returns the proposal id and the computed cascade impact."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_id": {
                    "type": "string",
                    "description": "The belief ID to retract (e.g. 'notes-not-persisted-to-disk')",
                },
                "reason": {
                    "type": "string",
                    "description": "Why the belief should be retracted (free text)",
                },
                "basis": {
                    "type": "string",
                    "description": "Evidential basis: 'source-divergence' | 'detected-contradiction' | 'prior-knowledge' (default)",
                },
                "evidence": {
                    "type": "string",
                    "description": "Supporting evidence for the retraction",
                },
                "failure_mode": {
                    "type": "string",
                    "description": "How the original belief fails: 'smuggled-premise' | 'unsupported-superlative' | 'false-causal-link' | 'domain-conflation' | 'contradicted-by-source' | 'stale'",
                },
            },
            "required": ["target_id"],
        },
    },
    {
        "name": "propose_supersede",
        "description": (
            "Propose superseding a hive belief with corrected text WITHOUT changing "
            "any truth value. Creates a pending proposal in the hive that a human "
            "(or policy) must accept before the supersession applies. Use this for "
            "beliefs that live in the read-only hive — supersede_belief only works on "
            "the local brain. Returns the proposal id, the generated new belief id, "
            "and the computed cascade impact."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "old_id": {
                    "type": "string",
                    "description": "The belief ID to supersede (the old one)",
                },
                "new_text": {
                    "type": "string",
                    "description": "The corrected belief text — a single atomic claim that replaces the old one",
                },
                "new_id": {
                    "type": "string",
                    "description": "Optional ID for the replacement belief (defaults to '<old_id>-vN' if it would not collide)",
                },
                "reason": {
                    "type": "string",
                    "description": "Why the belief should be superseded (free text)",
                },
                "basis": {
                    "type": "string",
                    "description": "Evidential basis: 'source-divergence' | 'detected-contradiction' | 'prior-knowledge' (default)",
                },
                "evidence": {
                    "type": "string",
                    "description": "Supporting evidence for the replacement",
                },
                "failure_mode": {
                    "type": "string",
                    "description": "How the original belief fails: 'smuggled-premise' | 'unsupported-superlative' | 'false-causal-link' | 'domain-conflation' | 'contradicted-by-source' | 'stale'",
                },
            },
            "required": ["old_id", "new_text"],
        },
    },
    {
        "name": "list_hive_proposals",
        "description": (
            "List hive proposals (pending retraction/supersession/addition requests) "
            "and their status. Read-only. Use this to see what has been proposed for "
            "the hive and what is still awaiting a human accept/reject."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Filter by status: 'pending' | 'accepted' | 'rejected' | 'withdrawn' | 'stale' (default: all)",
                },
                "target": {
                    "type": "string",
                    "description": "Filter by target belief ID",
                },
                "proposer": {
                    "type": "string",
                    "description": "Filter by proposer (e.g. 'worker-bee')",
                },
            },
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(entry, ensure_ascii=False) + "\n"
                       for entry in self._notes)
        # Atomic replace: on success the file is complete; on failure the
        # previous file (if any) is left intact and the error propagates so
        # callers can roll back their in-memory state and report the failure.
        _atomic_write_text(self.path, text)

    def list(self) -> list[dict]:
        return list(self._notes)

    def add(self, note: str) -> int:
        """Append a note. Returns its index in the store.

        Raises on persistence failure: the in-memory state is rolled back so
        it never diverges from what is actually on disk.
        """
        if not note or not note.strip():
            raise ValueError("note text is empty")
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": note,
        }
        previous = [dict(e) for e in self._notes]
        self._notes.append(entry)
        # Cap the history: rotate out the oldest entries beyond the cap.
        if self.max_notes and len(self._notes) > self.max_notes:
            self._notes = self._notes[-self.max_notes:]
        try:
            self._flush()
        except BaseException:
            self._notes = previous
            raise
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
        previous = [dict(e) for e in self._notes]
        entry["note"] = note
        entry["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            self._flush()
        except BaseException:
            self._notes = previous
            raise
        return entry

    def delete(self, index: int) -> int:
        self.get(index)  # validates range
        previous = [dict(e) for e in self._notes]
        self._notes.pop(index)
        try:
            self._flush()
        except BaseException:
            self._notes = previous
            raise
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
    def hive_proposal_path(self) -> str | None:
        """The database path hive proposals target.

        Proposals live in the shared hive layer. When no separate hive is
        configured (single-DB mode), proposals fall back to the local layer
        so the propose → accept workflow still works.
        """
        return self.hive_path or self.local_path

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
#
# In addition to the writable workspace, callers may grant READ-ONLY access to
# other directories (e.g. a dependency's source tree).  Read tools
# (read_file, grep, glob) resolve paths against the workspace OR any
# read-only root; write tools (write_file, edit_file) and run_command stay
# confined to the workspace.
_workspace_root = Path.cwd().resolve()
_read_only_roots: list[Path] = []
_ALLOWED_COMMANDS = frozenset({
    "cat", "cargo", "find", "git", "grep", "ls", "python", "pytest", "pwd",
    "ruff", "uv",
})


def set_workspace_root(path: str | os.PathLike[str] | None) -> None:
    """Set the root directory visible to filesystem and command tools.

    All filesystem paths are resolved beneath this directory.  ``None`` uses
    the current working directory.  Symlinks are resolved before the boundary
    check so a link cannot escape the workspace.  Resetting the workspace also
    clears any previously configured read-only paths — they must be re-added
    with :func:`set_read_only_paths` for each new workspace.
    """
    global _workspace_root
    root = Path.cwd() if path is None else Path(path)
    _workspace_root = root.expanduser().resolve()
    if not _workspace_root.is_dir():
        raise ValueError(f"workspace root is not a directory: {path}")
    _read_only_roots.clear()


def set_read_only_paths(paths: str | os.PathLike[str] | list[str | os.PathLike[str]] | None) -> None:
    """Grant READ-ONLY access to directories outside the workspace.

    Each path must be an existing directory and must not lie inside the
    workspace (a directory inside the workspace is already fully accessible).
    Read tools (read_file, grep, glob) accept paths beneath these roots; write
    tools and run_command remain confined to the workspace.  Pass ``None`` or
    an empty list to clear all read-only paths.
    """
    if paths is None:
        paths = []
    elif isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    roots: list[Path] = []
    for entry in paths:
        root = Path(entry).expanduser()
        try:
            resolved = root.resolve()
        except OSError as exc:
            raise ValueError(f"read-only path cannot be resolved: {entry}") from exc
        if not resolved.is_dir():
            raise ValueError(f"read-only path is not a directory: {entry}")
        try:
            resolved.relative_to(_workspace_root)
        except ValueError:
            pass  # fine — it is outside the workspace
        else:
            raise ValueError(
                f"read-only path is inside the workspace and does not need "
                f"to be listed: {entry}"
            )
        roots.append(resolved)
    _read_only_roots.clear()
    _read_only_roots.extend(roots)


def _any_allowed_root(path: Path) -> Path | None:
    """Return the root (workspace or read-only) containing ``path``, if any."""
    for root in (_workspace_root, *_read_only_roots):
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return root
    return None


def _workspace_path(path: str | os.PathLike[str], *, read_only_allowed: bool = True) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.expanduser().resolve()
    else:
        resolved = (_workspace_root / candidate).resolve()
    if read_only_allowed:
        if _any_allowed_root(resolved) is None:
            raise ValueError(f"path is outside workspace: {path}")
        return resolved
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


def _execute_mcp_belief_tool(mcp, name: str, tool_input: dict) -> str:
    """Route a belief tool call through the MCP store."""
    try:
        if name == "show_belief":
            return mcp.show(tool_input["id"])
        elif name == "search_beliefs":
            return mcp.search(tool_input["query"], tool_input.get("limit", 20))
        elif name == "list_blockers":
            return mcp.list_beliefs(truth_value="IN")
        elif name == "add_belief":
            return mcp.propose_belief(
                tool_input["id"], tool_input["text"], tool_input.get("source", "")
            )
        elif name == "propose_retraction":
            return mcp.propose_retraction(
                tool_input["target_id"], tool_input.get("reason", "")
            )
        elif name == "propose_supersede":
            result = mcp.propose_retraction(
                tool_input["old_id"], tool_input.get("reason", "")
            )
            new_text = tool_input.get("new_text", "")
            if new_text:
                new_id = tool_input.get("new_id", f"{tool_input['old_id']}-v2")
                result += "\n" + mcp.propose_belief(new_id, new_text)
            return result
        return f"Unknown MCP belief tool: {name}"
    except Exception as e:
        return f"Error (MCP): {e}"


def execute_tool(name: str, tool_input: dict) -> str:
    """Execute a tool by name. Returns the result as a string."""
    # Route belief tools through MCP when an MCP store is active
    from .mcp_beliefs import get_mcp_store
    mcp = get_mcp_store()
    if mcp and name in (
        "show_belief", "search_beliefs", "list_blockers",
        "add_belief", "propose_retraction", "propose_supersede",
    ):
        return _execute_mcp_belief_tool(mcp, name, tool_input)
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
    elif name == "retract_belief":
        return _retract_belief(tool_input["id"], tool_input.get("reason", ""))
    elif name == "supersede_belief":
        return _supersede_belief(tool_input["id"], tool_input.get("text", ""), tool_input.get("new_id"))
    elif name == "propose_retraction":
        return _propose_retraction(
            tool_input["target_id"],
            reason=tool_input.get("reason", ""),
            basis=tool_input.get("basis", "prior-knowledge"),
            evidence=tool_input.get("evidence", ""),
            failure_mode=tool_input.get("failure_mode", ""),
        )
    elif name == "propose_supersede":
        return _propose_supersede(
            tool_input["old_id"],
            tool_input.get("new_text", ""),
            new_id=tool_input.get("new_id"),
            reason=tool_input.get("reason", ""),
            basis=tool_input.get("basis", "prior-knowledge"),
            evidence=tool_input.get("evidence", ""),
            failure_mode=tool_input.get("failure_mode", ""),
        )
    elif name == "list_hive_proposals":
        return _list_hive_proposals(
            status=tool_input.get("status", ""),
            target=tool_input.get("target"),
            proposer=tool_input.get("proposer"),
        )
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


def _atomic_write_text(path, text: str) -> None:
    """Write ``text`` to ``path`` atomically.

    The content is written to a temporary file in the same directory
    (guaranteeing the same filesystem), flushed, fsynced, then moved over
    ``path`` with ``os.replace`` — an atomic rename on POSIX. If anything
    fails before the rename, the existing ``path`` is left completely
    untouched and no stale temporary file remains.
    """
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())
            except (OSError, ValueError):
                pass  # not fatal on all filesystems
        try:
            if path.exists():
                # Preserve the target's permissions on the replacement.
                os.chmod(tmp_name, path.stat().st_mode & 0o777)
        except OSError:
            pass
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _write_file(path, content):
    try:
        # Writes stay confined to the workspace; read-only roots are
        # read-only by design.
        path = _workspace_path(path, read_only_allowed=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(path, content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def _edit_file(path, old_string, new_string):
    try:
        # Edits stay confined to the workspace; read-only roots are
        # read-only by design.
        path = _workspace_path(path, read_only_allowed=False)
        with open(path, "r") as f:
            content = f.read()

        count = content.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {path}"
        if count > 1:
            return f"Error: old_string appears {count} times in {path} — must be unique"

        new_content = content.replace(old_string, new_string, 1)
        _atomic_write_text(path, new_content)
        return f"Successfully edited {path}"
    except Exception as e:
        return f"Error editing file: {e}"


def _grep(pattern, path):
    try:
        path = _workspace_path(path)
    except ValueError as e:
        return f"Error: {e}"
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex: {e}"

    matches = []
    skip = {".git", ".venv", "node_modules", "__pycache__", "target"}
    for root, dirs, files in os.walk(path):
        # os.walk does not follow directory symlinks by default, but prune them
        # explicitly so this remains true if traversal options change later.
        dirs[:] = [
            d for d in dirs
            if d not in skip
            and not d.startswith(".")
            and not Path(root, d).is_symlink()
        ]
        for filename in files:
            filepath = Path(root, filename)
            try:
                # A file symlink can point outside the workspace (or an
                # explicitly granted read-only root). Resolve and validate the
                # target before opening it, just as _glob does for its matches.
                resolved = filepath.resolve()
                if _any_allowed_root(resolved) is None or not resolved.is_file():
                    continue
                with open(resolved, "r", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append(f"{filepath}:{i}: {line.rstrip()}")
            except (OSError, IsADirectoryError, ValueError):
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
        # outside the workspace. Resolve every match before exposing it and
        # keep only files inside the workspace or a read-only root.
        try:
            resolved = Path(m).resolve()
            root = _any_allowed_root(resolved)
        except (OSError, ValueError):
            continue
        if root is None or not resolved.is_file():
            continue
        if root == _workspace_root:
            filtered.append(os.path.relpath(m, path))
        else:
            # Outside the workspace: expose the absolute path so the bee can
            # pass it straight back to read_file.
            filtered.append(str(resolved))

    if not filtered:
        return "No files found"
    return "\n".join(sorted(filtered))


def _check_pytest_paths(pytest_args):
    """Keep pytest's file/config roots confined to the workspace.

    These options and positional paths are interpreted by the pytest process
    itself rather than the host shell.
    """
    path_options = {"--rootdir", "--confcutdir", "--basetemp"}
    for index, arg in enumerate(pytest_args):
        value = None
        if arg in path_options and index + 1 < len(pytest_args):
            value = pytest_args[index + 1]
        elif any(arg.startswith(option + "=") for option in path_options):
            value = arg.split("=", 1)[1]
        elif not arg.startswith("-"):
            value = arg
        if value is not None:
            try:
                _workspace_path(value, read_only_allowed=False)
            except ValueError as e:
                return f"Error: pytest path is outside workspace: {e}"
    return None


def _validate_special_command(argv):
    """Reject command-specific escape hatches not covered by path checks."""
    executable = Path(argv[0]).name
    args = argv[1:]

    if executable == "find":
        dangerous = {"-exec", "-execdir", "-ok", "-okdir"}
        if any(arg in dangerous or any(arg.startswith(option + ",") for option in dangerous)
               for arg in args):
            return "Error: find execution actions are not allowed"

    if executable == "git":
        # Permit the small set of repository operations needed by the worker
        # (including staging and committing its own changes).  Network,
        # checkout, config, and other operations that can alter or escape the
        # workspace remain rejected.
        subcommand = next((arg for arg in args if not arg.startswith("-")), None)
        read_only = {"branch", "diff", "grep", "log", "ls-files", "rev-parse", "show", "status"}
        if subcommand == "add":
            add_args = [arg for arg in args[1:] if not arg.startswith("-")]
            if not add_args:
                return "Error: git add requires workspace paths"
            for value in add_args:
                try:
                    _workspace_path(value, read_only_allowed=False)
                except ValueError as e:
                    return f"Error: git path is outside workspace: {e}"
        elif subcommand == "commit":
            # Do not permit hooks/config overrides or commits without an
            # explicit message.  The commit itself is confined to this repo.
            forbidden = {"--amend", "--no-verify", "--gpg-sign", "--signoff", "--author"}
            if any(arg in forbidden or arg.startswith("--author=") for arg in args):
                return "Error: git commit option is not allowed"
            if "-m" not in args and not any(arg.startswith("--message=") for arg in args):
                return "Error: git commit requires an explicit message"
        elif subcommand not in read_only:
            return f"Error: git subcommand is not allowed: {subcommand or '(missing)'}"
        if any(arg in {"--upload-pack", "--receive-pack", "--exec-path", "--config-env"}
               or arg.startswith(("--upload-pack=", "--receive-pack=", "--exec-path=", "--config-env="))
               for arg in args):
            return "Error: git helper options are not allowed"

    if executable == "cargo":
        # Permit building, testing, and inspecting the workspace's Rust
        # packages.  Installing, linking, and toolchain operations that fetch
        # or execute arbitrary code remain rejected.
        subcommand = next((arg for arg in args if not arg.startswith("-") and not arg.startswith("+")), None)
        if subcommand in {"add", "bench", "install", "login", "publish", "remove", "vendor"}:
            return f"Error: cargo subcommand is not allowed: {subcommand}"
        if any(arg.startswith("+") for arg in args):
            return "Error: cargo toolchain overrides are not allowed"
        allowed = {"build", "check", "clippy", "doc", "fmt", "test"}
        if subcommand not in allowed:
            return f"Error: cargo subcommand is not allowed: {subcommand or '(missing)'}"

    if executable == "uv":
        # uv run/tool can execute an arbitrary program.  uv's environment
        # selectors and cache/output locations must not point outside the root.
        if args and args[0] == "run":
            # Permit the project's test runner, but do not turn uv into a
            # general arbitrary-program launcher.  pytest itself may execute
            # test code, so this is an intentional, narrowly scoped exception.
            if len(args) < 2 or args[1] != "pytest":
                return "Error: uv arbitrary program execution is not allowed"
            # Keep pytest's file/config roots confined as well.
            return _check_pytest_paths(args[2:])
        elif args and args[0] == "tool":
            return "Error: uv arbitrary program execution is not allowed"
        if any(arg == "--system" for arg in args):
            return "Error: uv system-wide writes are not allowed"
        forbidden = {"--python", "--python-preference", "--cache-dir"}
        if any(arg in forbidden or any(arg.startswith(option + "=") for option in forbidden)
               for arg in args):
            return "Error: uv interpreter/cache options are not allowed"
        for index, arg in enumerate(args):
            if arg in {"--target", "--prefix"}:
                if index + 1 >= len(args):
                    return f"Error: {arg} requires a workspace path"
                try:
                    _workspace_path(args[index + 1], read_only_allowed=False)
                except ValueError as e:
                    return f"Error: uv write target is outside workspace: {e}"
            elif arg.startswith(("--target=", "--prefix=")):
                try:
                    _workspace_path(arg.split("=", 1)[1], read_only_allowed=False)
                except ValueError as e:
                    return f"Error: uv write target is outside workspace: {e}"

    return None


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

    special_error = _validate_special_command(argv)
    if special_error:
        return special_error

    # Do not let command arguments turn a confined process into an escape hatch.
    # Validate directory options and the positional path arguments of commands
    # that can read or execute files.  Resolution also rejects symlinks leaving
    # the workspace.
    path_options = {"-C", "--directory", "--cwd", "--work-tree", "--git-dir", "--manifest-path"}
    path_args = set()
    if executable in {"cat", "find", "ls", "pytest", "ruff"}:
        path_args = {i for i, arg in enumerate(argv[1:], 1) if not arg.startswith("-")}
    elif executable == "grep":
        non_options = [i for i, arg in enumerate(argv[1:], 1) if not arg.startswith("-")]
        path_args = set(non_options[1:])  # first non-option is the pattern
    elif executable == "python":
        if "-c" in argv or "-" in argv:
            return "Error: inline Python execution is not allowed"
        if "-m" in argv:
            # Permit the project's test runner as a module, but do not let
            # -m become a general arbitrary-module launcher.
            index = argv.index("-m")
            if index + 1 >= len(argv) or argv[index + 1] != "pytest":
                return "Error: python -m arbitrary module execution is not allowed"
            error = _check_pytest_paths(argv[index + 2:])
            if error:
                return error
        else:
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
                _workspace_path(value, read_only_allowed=False)
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
        from reasons.api import search
        seen_ids = set()
        all_results = []
        for db_path in _belief_store.db_paths:
            raw = search(query, db_path=db_path, format="json", depth=0)
            if isinstance(raw, str):
                import json as _json
                try:
                    items = _json.loads(raw)
                except (ValueError, TypeError):
                    continue
            else:
                items = raw
            if not isinstance(items, list):
                continue
            for item in items:
                nid = item.get("id", "")
                if nid not in seen_ids:
                    seen_ids.add(nid)
                    all_results.append(item)
        all_results.sort(key=lambda r: (r.get("truth_value") != "IN", r.get("id", "")))
        all_results = all_results[:limit]
        if not all_results:
            return f"No beliefs matching '{query}'"
        lines = []
        for r in all_results:
            lines.append(f"[{r.get('truth_value', '?')}] {r['id']}: {r.get('text', '')[:120]}")
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


def _local_or_hive_status(belief_id: str) -> str:
    """Return 'local', 'hive', or 'none' for a belief id across the layers."""
    if _belief_store.id_exists_in_local(belief_id):
        return "local"
    if _belief_store.id_exists_in_hive(belief_id):
        return "hive"
    return "none"


def _retract_belief(belief_id: str, reason: str = "") -> str:
    if not _belief_store.has_any:
        return "Error: no belief database configured for this session"
    if not _belief_store.local_path:
        return "Error: no writable belief database configured"
    status = _local_or_hive_status(belief_id)
    if status == "none":
        return f"Error: belief '{belief_id}' not found"
    if status == "hive":
        return (
            f"Error: belief '{belief_id}' exists only in the read-only hive. "
            "Retracting it directly is not permitted; use the propose_retraction "
            "tool to propose it to the hive for a human to accept."
        )
    try:
        from reasons.api import retract_node
        result = retract_node(belief_id, reason=reason, db_path=_belief_store.local_path)
    except Exception as e:
        return f"Error retracting belief: {e}"
    lines = [f"Retracted belief '{belief_id}' [OUT]"]
    if reason:
        lines.append(f"Reason: {reason}")
    went_out = [n for n in result.get("went_out", []) if n != belief_id]
    if went_out:
        lines.append(f"Cascaded OUT: {', '.join(went_out)}")
    went_in = result.get("went_in", [])
    if went_in:
        lines.append(f"Restored IN: {', '.join(went_in)}")
    return "\n".join(lines)


def _supersede_belief(belief_id: str, text: str, new_id: str | None = None) -> str:
    if not _belief_store.has_any:
        return "Error: no belief database configured for this session"
    if not _belief_store.local_path:
        return "Error: no writable belief database configured"
    status = _local_or_hive_status(belief_id)
    if status == "none":
        return f"Error: belief '{belief_id}' not found"
    if status == "hive":
        return (
            f"Error: belief '{belief_id}' exists only in the read-only hive. "
            "Superseding it directly is not permitted; use the propose_supersede "
            "tool to propose a replacement to the hive for a human to accept."
        )
    if not text or not text.strip():
        return "Error: replacement text is empty"

    # Resolve a collision-free id for the successor, checking both layers so a
    # new belief never shadows or duplicates a hive id (same rule as add_belief).
    if new_id:
        target = new_id
    else:
        target = f"{belief_id}-v2"
        suffix = 3
        while _local_or_hive_status(target) != "none":
            target = f"{belief_id}-v{suffix}"
            suffix += 1
    if _local_or_hive_status(target) != "none":
        return f"Error: target belief '{target}' already exists. Choose a different new_id."

    try:
        from reasons.api import supersede_with_text
        result = supersede_with_text(
            belief_id, text, new_id=target, db_path=_belief_store.local_path
        )
    except Exception as e:
        return f"Error superseding belief: {e}"
    resolved_new_id = result.get("new_id", target)
    lines = [f"Superseded belief '{belief_id}' → new belief '{resolved_new_id}' [IN]"]
    changed = [n for n in result.get("changed", []) if n != belief_id]
    if changed:
        lines.append(f"Also affected: {', '.join(changed)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hive proposals (read-only hive → propose, human accepts)
#
# These tools implement the worker-bee surface of docs/hive-proposals.md §8:
# instead of writing the shared hive directly (refused by retract_belief /
# supersede_belief), the bee *proposes* a change and a human (or policy)
# accepts it. Proposals never change any truth value.
# ---------------------------------------------------------------------------

_PROPOSER = "worker-bee"


def _hive_proposal_path() -> str | None:
    """Resolve the database path hive proposals target, or None if there is
    no belief database configured at all."""
    if not _belief_store.has_any:
        return None
    return _belief_store.hive_proposal_path


def _format_proposal_impact(impact: dict) -> list[str]:
    """Render a what_if_* impact diff as human-readable lines."""
    lines: list[str] = []
    total = impact.get("total_affected", 0)
    lines.append(f"Impact: {total} belief(s) would be affected")
    retracted = impact.get("retracted") or []
    if retracted:
        ids = ", ".join(n.get("id", "?") for n in retracted)
        lines.append(f"Would go OUT: {ids}")
    restored = impact.get("restored") or []
    if restored:
        ids = ", ".join(n.get("id", "?") for n in restored)
        lines.append(f"Would be restored IN: {ids}")
    return lines


def _propose_retraction(
    target_id: str,
    reason: str = "",
    basis: str = "prior-knowledge",
    evidence: str = "",
    failure_mode: str = "",
) -> str:
    path = _hive_proposal_path()
    if not path:
        return "Error: no belief database configured for this session"
    try:
        from reasons.api import propose_retraction
        result = propose_retraction(
            target_id,
            reason=reason,
            failure_mode=failure_mode,
            basis=basis,
            evidence=evidence,
            proposer=_PROPOSER,
            db_path=path,
        )
    except Exception as e:
        return f"Error proposing retraction: {e}"
    lines = [
        f"Proposed retraction of '{target_id}' — pending (no truth values changed). "
        f"A human must accept it (e.g. `reasons accept {result['proposal_id']}`)."
    ]
    if reason:
        lines.append(f"Reason: {reason}")
    if failure_mode:
        lines.append(f"Failure mode: {failure_mode}")
    lines.append(f"Basis: {basis}")
    lines.extend(_format_proposal_impact(result.get("impact", {})))
    staled = result.get("staled") or []
    if staled:
        lines.append(f"Superseded older pending proposal(s): {', '.join(staled)}")
    lines.append(f"Proposal id: {result['proposal_id']}")
    return "\n".join(lines)


def _propose_supersede(
    old_id: str,
    new_text: str,
    new_id: str | None = None,
    reason: str = "",
    basis: str = "prior-knowledge",
    evidence: str = "",
    failure_mode: str = "",
) -> str:
    if not new_text or not new_text.strip():
        return "Error: replacement text is empty"
    path = _hive_proposal_path()
    if not path:
        return "Error: no belief database configured for this session"
    try:
        from reasons.api import propose_supersession
        result = propose_supersession(
            old_id,
            new_text,
            new_id=new_id or None,
            reason=reason,
            failure_mode=failure_mode,
            basis=basis,
            evidence=evidence,
            proposer=_PROPOSER,
            db_path=path,
        )
    except Exception as e:
        return f"Error proposing supersession: {e}"
    lines = [
        f"Proposed supersession of '{old_id}' → '{result.get('new_id')}' — pending "
        "(no truth values changed). A human must accept it "
        f"(e.g. `reasons accept {result['proposal_id']}`)."
    ]
    if reason:
        lines.append(f"Reason: {reason}")
    if failure_mode:
        lines.append(f"Failure mode: {failure_mode}")
    lines.append(f"Basis: {basis}")
    lines.extend(_format_proposal_impact(result.get("impact", {})))
    staled = result.get("staled") or []
    if staled:
        lines.append(f"Superseded older pending proposal(s): {', '.join(staled)}")
    lines.append(f"Proposal id: {result['proposal_id']}")
    return "\n".join(lines)


def _list_hive_proposals(
    status: str = "",
    target: str | None = None,
    proposer: str | None = None,
) -> str:
    path = _hive_proposal_path()
    if not path:
        return "Error: no belief database configured for this session"
    try:
        from reasons.api import list_proposals
        result = list_proposals(
            status=status or None,
            target_id=target or None,
            proposer=proposer or None,
            db_path=path,
        )
    except Exception as e:
        return f"Error listing proposals: {e}"
    proposals = result.get("proposals", [])
    if not proposals:
        scope = f" (status={status})" if status else ""
        return f"No hive proposals found{scope}."
    filters = []
    if status:
        filters.append(f"status={status}")
    if target:
        filters.append(f"target={target}")
    if proposer:
        filters.append(f"proposer={proposer}")
    header = f"{result.get('count', len(proposals))} hive proposal(s)"
    if filters:
        header += f" [{', '.join(filters)}]"
    header += ":"
    lines = [header]
    for p in proposals:
        entry = (
            f"{p['id']}  {p['action']}: {p['target_id']}"
        )
        if p.get("new_id"):
            entry += f" → {p['new_id']}"
        entry += f"  [{p['status']}]"
        if p.get("proposer"):
            entry += f"  proposer={p['proposer']}"
        if p.get("basis"):
            entry += f"  basis={p['basis']}"
        entry += f"  impact={p.get('impact', {}).get('total_affected', 0)}"
        if p.get("created_at"):
            entry += f"  created={p['created_at']}"
        lines.append(entry)
    lines.append("Accept/reject via `reasons accept <id>` / `reasons reject <id>` (CLI), "
                 "or ask a human to decide.")
    return "\n".join(lines)
