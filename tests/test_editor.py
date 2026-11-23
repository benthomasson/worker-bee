"""Tests for edit-session logging."""

from datetime import datetime as real_datetime

from worker_bee import editor


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
