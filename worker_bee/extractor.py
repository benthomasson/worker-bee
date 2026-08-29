"""Extract issues from a reasons.db belief database."""

from __future__ import annotations

import json
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
    """Find nodes that are IN but have an OUT antecedent."""
    rows = conn.execute("""
        SELECT j.node_id, j.antecedents_json, n.text, n.source
        FROM justifications j
        JOIN nodes n ON n.id = j.node_id
        WHERE n.truth_value = 'IN'
    """).fetchall()

    issues = []
    seen = set()
    for row in rows:
        antecedents = json.loads(row["antecedents_json"])
        if not antecedents:
            continue

        placeholders = ",".join("?" for _ in antecedents)
        out_ants = conn.execute(
            f"SELECT id FROM nodes WHERE id IN ({placeholders}) AND truth_value = 'OUT'",
            antecedents,
        ).fetchall()

        if out_ants and row["node_id"] not in seen:
            seen.add(row["node_id"])
            out_names = [r["id"] for r in out_ants]
            issues.append({
                "id": f"gated-{len(issues)}",
                "type": "gated",
                "belief_id": row["node_id"],
                "belief_text": row["text"],
                "source_files": _parse_sources(row["source"]),
                "description": f"Node is IN but antecedent(s) {', '.join(out_names)} are OUT",
            })

    return issues


def _find_contradictions(conn: sqlite3.Connection) -> list[dict]:
    """Find nogoods where all member nodes are still IN."""
    rows = conn.execute("SELECT id, nodes_json FROM nogoods").fetchall()

    issues = []
    for row in rows:
        members = json.loads(row["nodes_json"])
        if len(members) < 2:
            continue

        placeholders = ",".join("?" for _ in members)
        in_count = conn.execute(
            f"SELECT COUNT(*) as c FROM nodes WHERE id IN ({placeholders}) AND truth_value = 'IN'",
            members,
        ).fetchone()["c"]

        if in_count == len(members):
            issues.append({
                "id": f"contradiction-{len(issues)}",
                "type": "contradiction",
                "belief_id": " vs ".join(members),
                "belief_text": "",
                "source_files": [],
                "description": f"Nogood {row['id']}: all {len(members)} members are IN",
            })

    return issues


def _find_stale(conn: sqlite3.Connection) -> list[dict]:
    """Find nodes whose source files may have changed since derivation."""
    rows = conn.execute("""
        SELECT id, text, source, source_hash, created_at
        FROM nodes
        WHERE truth_value = 'IN'
          AND source != ''
          AND created_at != ''
    """).fetchall()

    issues = []
    for row in rows:
        sources = _parse_sources(row["source"])
        for src in sources:
            p = Path(src)
            if p.exists() and p.stat().st_mtime > _iso_to_ts(row["created_at"]):
                issues.append({
                    "id": f"stale-{len(issues)}",
                    "type": "stale",
                    "belief_id": row["id"],
                    "belief_text": row["text"],
                    "source_files": sources,
                    "description": f"Source file {src} modified after node creation",
                })
                break

    return issues


def _find_unreviewed(conn: sqlite3.Connection) -> list[dict]:
    """Find derived nodes that haven't been reviewed."""
    rows = conn.execute("""
        SELECT id, text, source
        FROM nodes
        WHERE truth_value = 'IN'
          AND created_at != ''
          AND (reviewed_at IS NULL OR reviewed_at = '')
    """).fetchall()

    return [
        {
            "id": f"unreviewed-{i}",
            "type": "unreviewed",
            "belief_id": row["id"],
            "belief_text": row["text"],
            "source_files": _parse_sources(row["source"]),
            "description": "Derived node has not been reviewed",
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
