"""Tests for atomic file writes (tmpfile + fsync + os.replace).

Covers:
- _atomic_write_text: creates new file, replaces existing, preserves
  permissions, and on failure leaves the existing file untouched with no
  stray temp file behind.
- write_file / edit_file: write content, create parent dirs, reject
  non-unique old_string, and on write failure preserve the original.
- NoteStore: _flush is an atomic replace; add/update/delete roll back
  in-memory state on persistence failure so memory stays consistent with
  disk, and the original notes file is left intact.
- Tool wrappers (write_note / update_note / delete_note) surface a
  persistence failure as an error string rather than a misleading success.
"""

import json
from pathlib import Path

import pytest

from worker_bee import tools


@pytest.fixture(autouse=True)
def _isolated_notes(tmp_path):
    """Point the active NoteStore at a fresh tmp_path so tests don't touch
    the real .worker-bee/notes.jsonl. Restore afterwards."""
    original = tools.get_notes_store()
    tools.set_notes_file(str(tmp_path / "notes.jsonl"))
    yield
    tools.set_notes_file(None)
    if original is not None and original.path:
        tools.set_notes_file(str(original.path))


def _force_rename_failure(monkeypatch):
    """Make the final os.replace inside _atomic_write_text fail, exercising
    the failure/cleanup path without touching tempfile.mkstemp."""
    import os

    def failing_replace(*a, **k):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(os, "replace", failing_replace)


# ---------------------------------------------------------------------------
# _atomic_write_text helper
# ---------------------------------------------------------------------------


def test_atomic_write_creates_new_file(tmp_path):
    target = tmp_path / "a.txt"
    tools._atomic_write_text(target, "hello")
    assert target.read_text() == "hello"
    leftovers = [p for p in tmp_path.iterdir() if p.name != "a.txt"]
    assert leftovers == []


def test_atomic_write_replaces_existing_file(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("old")
    tools._atomic_write_text(target, "new content, longer than the old")
    assert target.read_text() == "new content, longer than the old"


def test_atomic_write_preserves_existing_permissions(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("old")
    target.chmod(0o644)
    tools._atomic_write_text(target, "new")
    assert (target.stat().st_mode & 0o777) == 0o644


def test_atomic_write_failure_leaves_existing_intact(tmp_path, monkeypatch):
    """If the rename fails after the temp file is written, the target file
    must remain byte-for-byte unchanged and no temp file may be left behind."""
    target = tmp_path / "a.txt"
    original = "original-content"
    target.write_text(original)

    _force_rename_failure(monkeypatch)

    with pytest.raises(OSError):
        tools._atomic_write_text(target, "should-not-appear")

    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["a.txt"], "no stray tempfiles must be left behind"
    assert target.read_text() == original, "existing file must be untouched"


# ---------------------------------------------------------------------------
# write_file / edit_file
# ---------------------------------------------------------------------------


def test_write_file_creates_parent_dirs(tmp_path):
    tools.set_workspace_root(tmp_path)
    result = tools.execute_tool(
        "write_file", {"path": "a/b/c.txt", "content": "nested"}
    )
    assert result == f"Successfully wrote to {tmp_path / 'a/b/c.txt'}"
    assert (tmp_path / "a/b/c.txt").read_text() == "nested"


def test_edit_file_replaces_uniquely(tmp_path):
    tools.set_workspace_root(tmp_path)
    f = tmp_path / "f.txt"
    f.write_text("one two three")
    result = tools.execute_tool(
        "edit_file", {"path": "f.txt", "old_string": "two", "new_string": "TWO"}
    )
    assert result == f"Successfully edited {f}"
    assert f.read_text() == "one TWO three"


def test_edit_file_rejects_non_unique(tmp_path):
    tools.set_workspace_root(tmp_path)
    f = tmp_path / "f.txt"
    f.write_text("dup dup dup")
    result = tools.execute_tool(
        "edit_file", {"path": "f.txt", "old_string": "dup", "new_string": "X"}
    )
    assert "appears 3 times" in result


def test_edit_file_failure_preserves_original(tmp_path, monkeypatch):
    tools.set_workspace_root(tmp_path)
    f = tmp_path / "f.txt"
    original = "keep me intact"
    f.write_text(original)

    _force_rename_failure(monkeypatch)
    result = tools.execute_tool(
        "edit_file", {"path": "f.txt", "old_string": "keep", "new_string": "X"}
    )
    assert "Error editing file" in result
    assert f.read_text() == original


def test_write_file_failure_preserves_original(tmp_path, monkeypatch):
    tools.set_workspace_root(tmp_path)
    f = tmp_path / "f.txt"
    original = "keep me intact"
    f.write_text(original)

    _force_rename_failure(monkeypatch)
    result = tools.execute_tool(
        "write_file", {"path": "f.txt", "content": "new"}
    )
    assert "Error writing file" in result
    assert f.read_text() == original


# ---------------------------------------------------------------------------
# NoteStore atomic flush + rollback
# ---------------------------------------------------------------------------


def _note_store(tmp_path):
    return tools.NoteStore(path=str(tmp_path / "notes.jsonl"))


def test_notes_add_persists_atomically(tmp_path):
    store = _note_store(tmp_path)
    idx = store.add("first note")
    assert idx == 0
    path = Path(store.path)
    assert path.exists()
    lines = [json.loads(l) for l in path.read_text().splitlines() if l]
    assert lines[0]["note"] == "first note"


def test_notes_flush_is_atomic_replace(tmp_path):
    """On success, the notes file is exactly the serialized notes (no
    partial writes, no trailing garbage)."""
    store = _note_store(tmp_path)
    for n in ("one", "two", "three"):
        store.add(n)
    lines = [json.loads(l) for l in Path(store.path).read_text().splitlines() if l]
    assert [e["note"] for e in lines] == ["one", "two", "three"]


def test_notes_add_failure_rolls_back_memory_and_leaves_file_intact(
    tmp_path, monkeypatch
):
    store = _note_store(tmp_path)
    store.add("seed note")
    path = Path(store.path)
    before = path.read_text()
    memory_before = [dict(e) for e in store._notes]

    _force_rename_failure(monkeypatch)
    with pytest.raises(OSError):
        store.add("will not persist")

    assert path.read_text() == before, "existing notes file must be untouched"
    assert [dict(e) for e in store._notes] == memory_before, (
        "in-memory state must match disk after failed add"
    )


def test_notes_update_failure_rolls_back(tmp_path, monkeypatch):
    store = _note_store(tmp_path)
    store.add("original")
    path = Path(store.path)
    before = path.read_text()

    _force_rename_failure(monkeypatch)
    with pytest.raises(OSError):
        store.update(0, "mutated")

    assert path.read_text() == before
    assert store._notes[0]["note"] == "original"


def test_notes_delete_failure_rolls_back(tmp_path, monkeypatch):
    store = _note_store(tmp_path)
    store.add("a")
    store.add("b")
    path = Path(store.path)
    before = path.read_text()
    count_before = len(store._notes)

    _force_rename_failure(monkeypatch)
    with pytest.raises(OSError):
        store.delete(0)

    assert path.read_text() == before
    assert len(store._notes) == count_before


# ---------------------------------------------------------------------------
# Tool wrappers surface persistence failure as an error
# ---------------------------------------------------------------------------


def test_tool_write_note_surfaces_persistence_failure(tmp_path, monkeypatch):
    # write_note is not routed through execute_tool; the live path is the
    # editor's SessionMemory.add_note, which wraps the store and falls back
    # to in-memory on persistence failure.
    from worker_bee.editor import SessionMemory

    store = tools.get_notes_store()
    memory = SessionMemory(notes_store=store)
    _force_rename_failure(monkeypatch)
    result = memory.add_note("will not persist")
    # Must NOT report a successful disk persist; it must signal the fallback.
    assert "persisted to" not in result, result
    assert "persistence error" in result, result


def test_tool_update_note_surfaces_persistence_failure(tmp_path, monkeypatch):
    store = tools.get_notes_store()
    store.add("a")
    _force_rename_failure(monkeypatch)
    result = tools.execute_tool(
        "update_note", {"index": 0, "note": "will not persist"}
    )
    assert "Error" in result, result


def test_tool_delete_note_surfaces_persistence_failure(tmp_path, monkeypatch):
    store = tools.get_notes_store()
    store.add("a")
    _force_rename_failure(monkeypatch)
    result = tools.execute_tool("delete_note", {"index": 0})
    assert "Error" in result, result
