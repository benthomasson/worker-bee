"""Shared workspace configuration with no optional dependencies."""

from __future__ import annotations

import os
from pathlib import Path

_workspace_root = Path.cwd().resolve()


def set_workspace_root(path: str | os.PathLike[str] | None) -> Path:
    """Set and return the canonical workspace root."""
    global _workspace_root
    root = Path.cwd() if path is None else Path(path)
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"workspace root is not a directory: {path}")
    _workspace_root = root
    return root


def get_workspace_root() -> Path:
    """Return the canonical workspace root."""
    return _workspace_root


def workspace_path(
    path: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve *path* and require it to remain inside the workspace.

    ``root`` allows callers to validate against a supplied workspace without
    changing the process-wide configuration.
    """
    workspace = _workspace_root if root is None else Path(root).expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(f"workspace root is not a directory: {root}")
    candidate = Path(path).expanduser()
    resolved = (candidate if candidate.is_absolute() else workspace / candidate).resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"path is outside workspace: {path}") from exc
    return resolved
