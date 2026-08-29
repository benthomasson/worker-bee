"""Extract issues from a reasons.db belief database."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def extract(db_path: str | Path, *, types: list[str] | None = None) -> list[dict]:
    """Query reasons.db for actionable issues.

    Returns a list of issue dicts, each with: id, type, belief_id,
    belief_text, source_files, description.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    issues: list[dict] = []
    all_types = types or ["gated", "contradiction", "stale", "unreviewed"]

    if "gated" in all_types:
        issues.extend(_find_gated(conn))
    if "contradiction" in all_types:
        issues.extend(_find_contradictions(conn))
    if "stale" in all_types:
        issues.extend(_find_stale(conn))
    if "unreviewed" in all_types:
        issues.extend(_find_unreviewed(conn))

    conn.close()
    return issues


def _find_gated(conn: sqlite3.Connection) -> list[dict]:
    """Find beliefs that are IN but depend on an OUT antecedent."""
    rows = conn.execute("""
        SELECT b.name, b.text, b.source
        FROM beliefs b
        JOIN justifications j ON j.belief_id = b.id
        JOIN justification_antecedents ja ON ja.justification_id = j.id
        JOIN beliefs ant ON ant.id = ja.antecedent_id
        WHERE b.status = 'in'
          AND ant.status = 'out'
    """).fetchall()

    return [
        {
            "id": f"gated-{i}",
            "type": "gated",
            "belief_id": row["name"],
            "belief_text": row["text"],
            "source_files": _parse_sources(row["source"]),
            "description": f"Belief is IN but has an OUT antecedent",
        }
        for i, row in enumerate(rows)
    ]


def _find_contradictions(conn: sqlite3.Connection) -> list[dict]:
    """Find nogood pairs where both beliefs are still IN."""
    rows = conn.execute("""
        SELECT n.id, b1.name AS name1, b1.text AS text1,
               b2.name AS name2, b2.text AS text2
        FROM nogoods n
        JOIN nogood_members nm1 ON nm1.nogood_id = n.id
        JOIN nogood_members nm2 ON nm2.nogood_id = n.id AND nm2.id > nm1.id
        JOIN beliefs b1 ON b1.id = nm1.belief_id
        JOIN beliefs b2 ON b2.id = nm2.belief_id
        WHERE b1.status = 'in' AND b2.status = 'in'
    """).fetchall()

    return [
        {
            "id": f"contradiction-{i}",
            "type": "contradiction",
            "belief_id": f"{row['name1']} vs {row['name2']}",
            "belief_text": f"{row['text1']} <-> {row['text2']}",
            "source_files": [],
            "description": f"Contradiction: both beliefs are IN",
        }
        for i, row in enumerate(rows)
    ]


def _find_stale(conn: sqlite3.Connection) -> list[dict]:
    """Find beliefs whose source files may have changed."""
    rows = conn.execute("""
        SELECT name, text, source, derived_at
        FROM beliefs
        WHERE status = 'in'
          AND source IS NOT NULL
          AND derived_at IS NOT NULL
    """).fetchall()

    stale = []
    for i, row in enumerate(rows):
        sources = _parse_sources(row["source"])
        for src in sources:
            p = Path(src)
            if p.exists() and p.stat().st_mtime > _iso_to_ts(row["derived_at"]):
                stale.append({
                    "id": f"stale-{i}",
                    "type": "stale",
                    "belief_id": row["name"],
                    "belief_text": row["text"],
                    "source_files": sources,
                    "description": f"Source file {src} modified after belief derivation",
                })
                break

    return stale


def _find_unreviewed(conn: sqlite3.Connection) -> list[dict]:
    """Find derived beliefs that haven't been reviewed."""
    rows = conn.execute("""
        SELECT name, text, source
        FROM beliefs
        WHERE status = 'in'
          AND derived_at IS NOT NULL
          AND reviewed_at IS NULL
    """).fetchall()

    return [
        {
            "id": f"unreviewed-{i}",
            "type": "unreviewed",
            "belief_id": row["name"],
            "belief_text": row["text"],
            "source_files": _parse_sources(row["source"]),
            "description": "Derived belief has not been reviewed",
        }
        for i, row in enumerate(rows)
    ]


def _parse_sources(source: str | None) -> list[str]:
    if not source:
        return []
    return [s.strip() for s in source.split(",") if s.strip()]


def _iso_to_ts(iso_str: str) -> float:
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0
