"""Tests for dated summary entry publication."""

from datetime import datetime, timezone

from worker_bee import summarizer


class _FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 11, 20, 12, 34, 56, tzinfo=tz or timezone.utc)


def test_write_entry_creates_missing_dated_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(summarizer, "datetime", _FixedDateTime)

    entry_path = summarizer._write_entry(
        "# Completed task\n\nSummary text.",
        [{"event": "session_start", "task": "Create summary entry"}],
        tmp_path / "entries",
    )

    expected = tmp_path / "entries" / "2026" / "11" / "20" / "create-summary-entry.md"
    assert entry_path == expected
    assert expected.read_text() == "# Completed task\n\nSummary text."
