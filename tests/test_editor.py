"""Tests for edit-session logging."""

import json
from datetime import datetime as real_datetime
from pathlib import Path

from worker_bee import editor
from worker_bee.llm import ChatResponse, TextBlock, ToolUseBlock
from worker_bee.tools import execute_tool, set_notes_file


class _FailingNotesStore:
    enabled = True
    path = "notes.jsonl"

    def list(self):
        return []

    def add(self, _note, name=None):
        raise OSError("disk full")


def test_log_records_include_schema_version(tmp_path):
    log = editor._init_log(tmp_path, "task", "model", False)
    try:
        editor._log_event(log, "test_event", detail="value")
        editor._log_event(log, "override_attempt", schema_version=999)
        records = [json.loads(line) for line in log["path"].read_text().splitlines()]
        assert len(records) == 3
        assert all(record["schema_version"] == 1 for record in records)
        assert records[0]["event"] == "session_start"
        assert records[1]["event"] == "test_event"
        assert records[2]["event"] == "override_attempt"
    finally:
        log["file"].close()


def test_note_persistence_failure_has_failure_status():
    memory = editor.SessionMemory(notes_store=_FailingNotesStore())
    result = memory.add_note("keep this note")

    assert isinstance(result, str)
    assert result.ok is False
    assert "persistence error" in result


def test_session_memory_refreshes_after_note_mutations(tmp_path):
    set_notes_file(str(tmp_path / "notes.jsonl"))
    try:
        store = editor.get_notes_store()
        store.add("first", name="note-first")
        store.add("second", name="note-second")
        memory = editor.SessionMemory(notes_store=store)

        execute_tool("update_note", {"name": "note-first", "note": "updated"})
        memory.refresh_notes()
        assert [e["note"] for e in memory._notes] == ["updated", "second"]

        execute_tool("delete_note", {"name": "note-first"})
        memory.refresh_notes()
        assert [e["note"] for e in memory._notes] == ["second"]
        assert "second" in memory.list_calls()
        assert "updated" not in memory.list_calls()
    finally:
        set_notes_file(None)


def test_run_edit_loop_refreshes_memory_after_note_mutation(tmp_path, monkeypatch):
    notes_path = tmp_path / "notes.jsonl"
    set_notes_file(str(notes_path))
    try:
        editor.get_notes_store().add("before", name="test-note")
        monkeypatch.setattr(editor, "DEFAULT_NOTES_PATH", str(notes_path))
        responses = iter([
            ChatResponse(
                content=[ToolUseBlock(
                    "tool-1", "update_note", {"name": "test-note", "note": "after"},
                )],
                stop_reason="tool_use",
            ),
            ChatResponse(
                content=[ToolUseBlock("tool-2", "list_memory", {})],
                stop_reason="tool_use",
            ),
            ChatResponse(content=[TextBlock("done")], stop_reason="end_turn"),
        ])

        def fake_dispatch(messages, **kwargs):
            return next(responses)

        monkeypatch.setattr(editor, "dispatch_chat", fake_dispatch)
        session = editor.run_edit_loop(
            "test", model="test", max_turns=3, log_dir=tmp_path,
            workspace_root=tmp_path,
        )

        assert session.completed is True
        memory_results = [
            step.tool_result for step in session.steps
            if step.role == "tool" and step.tool_name == "list_memory"
        ]
        assert len(memory_results) == 1
        memory_result = memory_results[0]
        assert memory_result is not None
        assert "[test-note] after" in memory_result
        assert "before" not in memory_result
    finally:
        set_notes_file(None)


def test_tool_result_event_records_explicit_status(tmp_path):
    log = editor._init_log(tmp_path, "task", "model", False)
    try:
        result = editor.ToolResult("Error: failed", ok=False)
        editor._log_event(log, "tool_result", tool="read_file", result=result,
                          ok=result.ok)
        log["file"].flush()
        record = json.loads(log["path"].read_text().splitlines()[-1])
        assert record["result"] == "Error: failed"
        assert record["ok"] is False
    finally:
        log["file"].close()


def test_run_edit_loop_returns_failed_tool_result_and_logs_status(tmp_path, monkeypatch):
    calls = []
    responses = iter([
        ChatResponse(
            content=[ToolUseBlock("tool-1", "read_file", {"path": "missing.txt"})],
            stop_reason="tool_use",
        ),
        ChatResponse(content=[TextBlock("done")], stop_reason="end_turn"),
    ])

    def fake_dispatch(messages, **kwargs):
        calls.append(messages)
        return next(responses)

    monkeypatch.setattr(editor, "dispatch_chat", fake_dispatch)
    session = editor.run_edit_loop(
        "test", model="test", max_turns=2, log_dir=tmp_path,
        workspace_root=tmp_path,
    )

    assert session.completed is True
    assert len(calls) == 2
    tool_message = calls[1][-1]
    assert tool_message["role"] == "user"
    result = tool_message["content"][0]
    assert result["content"].startswith("Error reading file:")
    log_records = [
        json.loads(line)
        for line in Path(session.log_path).read_text().splitlines()
    ]
    tool_events = [record for record in log_records if record["event"] == "tool_result"]
    assert len(tool_events) == 1
    assert tool_events[0]["ok"] is False
    assert tool_events[0]["result"].startswith("Error reading file:")


def test_init_log_allocates_distinct_files_for_same_timestamp(tmp_path, monkeypatch):
    class FixedDateTime:
        @classmethod
        def now(cls, tz=None):
            return real_datetime(2026, 9, 10, 19, 59, 0, 123456, tzinfo=tz)

    monkeypatch.setattr(editor, "datetime", FixedDateTime)

    first = editor._init_log(tmp_path, "first", "model", False)
    second = editor._init_log(tmp_path, "second", "model", False)
    try:
        first["file"].flush()
        second["file"].flush()
        assert first["path"] != second["path"]
        assert first["path"].name == "20260910-195900-123456.jsonl"
        assert second["path"].name == "20260910-195900-123456-1.jsonl"
        assert first["path"].read_text().count("session_start") == 1
        assert second["path"].read_text().count("session_start") == 1
    finally:
        first["file"].close()
        second["file"].close()


def test_init_log_retries_existing_timestamp_file(tmp_path, monkeypatch):
    (tmp_path / "20260910-195900-123456.jsonl").write_text("existing\n")

    class FixedDateTime:
        @classmethod
        def now(cls, tz=None):
            return real_datetime(2026, 9, 10, 19, 59, 0, 123456, tzinfo=tz)

    monkeypatch.setattr(editor, "datetime", FixedDateTime)
    log = editor._init_log(tmp_path, "task", "model", False)
    try:
        log["file"].flush()
        assert log["path"].name == "20260910-195900-123456-1.jsonl"
        assert "existing" not in log["path"].read_text()
    finally:
        log["file"].close()


def test_init_log_runs_retention_before_creating_session(tmp_path, monkeypatch):
    old = tmp_path / "old.jsonl"
    old.write_text("old")
    import os
    os.utime(old, (0, 0))
    monkeypatch.setattr(editor, "cleanup_logs", lambda path: old.unlink() or [old])

    log = editor._init_log(tmp_path, "task", "model", False)
    try:
        assert not old.exists()
        assert log["path"].exists()
    finally:
        log["file"].close()


def test_init_log_survives_retention_failure(tmp_path, monkeypatch):
    def fail(_path):
        raise OSError("cleanup race")

    monkeypatch.setattr(editor, "cleanup_logs", fail)
    log = editor._init_log(tmp_path, "task", "model", False)
    log["file"].close()
    assert log["path"].exists()
