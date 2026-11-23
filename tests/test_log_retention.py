from datetime import datetime, timezone
import os

import pytest

from worker_bee.log_retention import cleanup_logs


def _set_mtime(path, timestamp):
    path.touch()
    os.utime(path, (timestamp, timestamp))


def test_cleanup_removes_expired_logs_but_keeps_recent(tmp_path):
    now = datetime(2026, 1, 31, tzinfo=timezone.utc).timestamp()
    old = tmp_path / "old.jsonl"
    recent = tmp_path / "recent.jsonl"
    _set_mtime(old, now - 31 * 24 * 60 * 60)
    _set_mtime(recent, now - 2 * 24 * 60 * 60)

    deleted = cleanup_logs(tmp_path, max_age_days=30, now=now)

    assert deleted == [old]
    assert not old.exists()
    assert recent.exists()


def test_cleanup_trims_oldest_files_to_size_limit(tmp_path):
    now = 1_000_000.0
    oldest = tmp_path / "oldest.jsonl"
    newest = tmp_path / "newest.jsonl"
    oldest.write_text("a" * 8)
    newest.write_text("b" * 8)
    _set_mtime(oldest, now - 10)
    _set_mtime(newest, now - 1)

    deleted = cleanup_logs(tmp_path, max_age_days=1000, max_total_bytes=8, now=now)

    assert deleted == [oldest]
    assert not oldest.exists()
    assert newest.exists()


def test_cleanup_zero_age_removes_jsonl_files(tmp_path):
    now = 1_000_000.0
    log = tmp_path / "log.jsonl"
    _set_mtime(log, now)

    assert cleanup_logs(tmp_path, max_age_days=0, max_total_bytes=100, now=now + 1) == [log]
    assert not log.exists()


def test_cleanup_keeps_file_exactly_at_age_cutoff(tmp_path):
    now = 1_000_000.0
    log = tmp_path / "at-cutoff.jsonl"
    _set_mtime(log, now - 30 * 24 * 60 * 60)

    assert cleanup_logs(tmp_path, max_age_days=30, max_total_bytes=100, now=now) == []
    assert log.exists()


def test_cleanup_skips_symlink_to_regular_log(tmp_path):
    target = tmp_path / "target.jsonl"
    link = tmp_path / "link.jsonl"
    target.write_text("keep")
    link.symlink_to(target)

    assert cleanup_logs(tmp_path, max_age_days=0, max_total_bytes=0, now=1_000_001) == []
    assert target.exists()
    assert link.is_symlink()


def test_cleanup_rejects_negative_limits(tmp_path):
    with pytest.raises(ValueError, match="max_age_days"):
        cleanup_logs(tmp_path, max_age_days=-1)
    with pytest.raises(ValueError, match="max_total_bytes"):
        cleanup_logs(tmp_path, max_total_bytes=-1)


def test_cleanup_ignores_other_files_and_subdirectories(tmp_path):
    (tmp_path / "notes.txt").write_text("keep")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "nested.jsonl").write_text("keep")

    assert cleanup_logs(tmp_path, max_age_days=0, max_total_bytes=0, now=1_000_000) == []
    assert (tmp_path / "notes.txt").exists()
    assert (nested / "nested.jsonl").exists()


def test_cleanup_skips_files_that_fail_stat(tmp_path, monkeypatch):
    log = tmp_path / "log.jsonl"
    log.write_text("log")
    original_lstat = type(log).lstat

    def failing_lstat(path, *args, **kwargs):
        if path == log:
            raise OSError("inaccessible")
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(type(log), "lstat", failing_lstat)
    assert cleanup_logs(tmp_path, max_age_days=0, now=1_000_001) == []


def test_cleanup_tolerates_unlink_failure_during_size_trim(tmp_path, monkeypatch):
    oldest = tmp_path / "oldest.jsonl"
    newest = tmp_path / "newest.jsonl"
    oldest.write_text("a" * 8)
    newest.write_text("b" * 8)
    os.utime(oldest, (999_990, 999_990))
    os.utime(newest, (999_999, 999_999))
    original_unlink = type(oldest).unlink

    def failing_unlink(path, *args, **kwargs):
        if path == oldest:
            raise OSError("busy")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(type(oldest), "unlink", failing_unlink)
    assert cleanup_logs(tmp_path, max_age_days=1000, max_total_bytes=8, now=1_000_000) == [newest]
    assert oldest.exists()
