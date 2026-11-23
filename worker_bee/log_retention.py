"""Retention helpers for worker-bee session logs."""

from __future__ import annotations

import stat
import time
from pathlib import Path

DEFAULT_MAX_AGE_DAYS = 30
DEFAULT_MAX_TOTAL_BYTES = 100 * 1024 * 1024


def cleanup_logs(
    log_dir: str | Path,
    *,
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    now: float | None = None,
) -> list[Path]:
    """Delete expired logs and trim the oldest logs over the size limit.

    Only regular ``.jsonl`` files directly in *log_dir* are considered. Cleanup
    is best-effort: a file that disappears or cannot be removed is skipped.
    Returns the paths successfully deleted.  A file exactly at the age cutoff
    is retained; only files strictly older than the cutoff are expired.

    ``max_age_days`` and ``max_total_bytes`` must be non-negative.
    """
    if max_age_days < 0:
        raise ValueError("max_age_days must be non-negative")
    if max_total_bytes < 0:
        raise ValueError("max_total_bytes must be non-negative")

    directory = Path(log_dir)
    try:
        if not directory.is_dir():
            return []
        candidates = directory.glob("*.jsonl")
    except OSError:
        return []

    # Capture metadata while discovering files.  Every filesystem operation is
    # allowed to fail: another session may remove a file between any two calls.
    files: list[tuple[Path, object]] = []
    try:
        for path in candidates:
            try:
                metadata = path.lstat()
                if stat.S_ISREG(metadata.st_mode):
                    files.append((path, metadata))
            except (FileNotFoundError, OSError):
                continue
    except OSError:
        pass

    timestamp = time.time() if now is None else now
    cutoff = timestamp - max_age_days * 24 * 60 * 60
    deleted: list[Path] = []

    # Age retention is independent of the size limit.  Logs that are open in
    # another process are not detectable portably and remain eligible under
    # this policy; callers should use the age/size limits accordingly.
    remaining: list[tuple[Path, object]] = []
    for path, metadata in files:
        try:
            if metadata.st_mtime < cutoff:
                path.unlink()
                deleted.append(path)
            else:
                remaining.append((path, metadata))
        except (FileNotFoundError, OSError):
            pass

    # Refresh metadata before size accounting so disappearing or inaccessible
    # files are simply excluded rather than making session startup fail.
    sized: list[tuple[Path, object]] = []
    for path, _metadata in remaining:
        try:
            sized.append((path, path.stat()))
        except (FileNotFoundError, OSError):
            continue

    total = sum(metadata.st_size for _path, metadata in sized)
    if total > max_total_bytes:
        sized.sort(key=lambda item: item[1].st_mtime)
        for path, _metadata in sized:
            if total <= max_total_bytes:
                break
            try:
                metadata = path.stat()
                path.unlink()
                deleted.append(path)
                total -= metadata.st_size
            except (FileNotFoundError, OSError):
                pass

    return deleted
