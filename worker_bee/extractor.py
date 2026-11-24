"""Extract issues from a reasons.db belief database."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path


def extract(db_path: str | Path, *, types: list[str] | None = None) -> list[dict]:
    """Query reasons.db for actionable issues.

    Returns a list of issue dicts, each with: id, type, belief_id,
    belief_text, source_files, description.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    issues: list[dict] = []
    all_types = types or ["gated", "contradiction", "stale", "unreviewed"]

    if "gated" in all_types:
        issues.extend(_find_gated(str(db_path)))
    if "contradiction" in all_types:
        issues.extend(_find_contradictions(str(db_path)))
    if "stale" in all_types:
        issues.extend(_find_stale(str(db_path)))
    if "unreviewed" in all_types:
        issues.extend(_find_unreviewed(str(db_path)))

    return issues


def filter_acceptance_gates(
    db_path: str | Path,
    issues: list[dict],
    *,
    require_reviewed: bool = True,
    require_verified: bool = True,
) -> list[dict]:
    """Keep only issues backed by accepted IN nodes.

    Database access belongs to extraction, so callers do not need to duplicate
    the node-state query.  Issues without one concrete belief ID (such as
    contradictions) cannot satisfy an acceptance gate and are excluded.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    if not require_reviewed and not require_verified:
        return issues
    # Contradiction issues use a display ID made from all of their members;
    # acceptance applies to every member rather than to that composite string.
    issue_members: dict[int, list[str]] = {}
    ids: set[str] = set()
    for index, issue in enumerate(issues):
        belief_id = issue.get("belief_id")
        if not isinstance(belief_id, str):
            continue
        if issue.get("type") == "contradiction":
            members = [member.strip() for member in belief_id.split(" vs ") if member.strip()]
        else:
            members = [belief_id]
        if members:
            issue_members[index] = members
            ids.update(members)
    if not ids:
        return []

    accepted: set[str] = set()
    # Keep each query below SQLite's usual host-parameter limit.  This also
    # avoids making acceptance depend on a connection-specific limit.
    batch_size = 900
    with sqlite3.connect(str(db_path)) as conn:
        id_list = list(ids)
        for start in range(0, len(id_list), batch_size):
            batch = id_list[start:start + batch_size]
            placeholders = ",".join("?" for _ in batch)
            rows = conn.execute(
                f"SELECT id, truth_value, reviewed_at, verified_at FROM nodes "
                f"WHERE id IN ({placeholders})", tuple(batch)
            ).fetchall()
            accepted.update(
                row[0] for row in rows
                if row[1] == "IN"
                and (not require_reviewed or _valid_timestamp(row[2]))
                and (not require_verified or _valid_timestamp(row[3]))
            )

    return [
        issue for index, issue in enumerate(issues)
        if index in issue_members
        and all(member in accepted for member in issue_members[index])
    ]


def _find_gated(db_path: str) -> list[dict]:
    """Find nodes that are IN but have an OUT antecedent."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT j.node_id, j.antecedents_json, n.text, n.source
        FROM justifications j
        JOIN nodes n ON n.id = j.node_id
        WHERE n.truth_value = 'IN'
          AND (n.verified_at IS NULL OR n.verified_at = '')
        ORDER BY j.rowid
    """).fetchall()

    issues = []
    issues_by_node = {}
    out_names_by_node = {}
    for row in rows:
        antecedents = json.loads(row["antecedents_json"])
        if not antecedents:
            continue

        placeholders = ",".join("?" for _ in antecedents)
        out_ids = {
            out_ant["id"]
            for out_ant in conn.execute(
                f"SELECT id FROM nodes WHERE id IN ({placeholders}) AND truth_value = 'OUT'",
                antecedents,
            )
        }

        node_id = row["node_id"]
        out_names = out_names_by_node.setdefault(node_id, [])
        for antecedent in antecedents:
            if antecedent in out_ids and antecedent not in out_names:
                out_names.append(antecedent)
        if not out_names:
            continue

        issue = issues_by_node.get(node_id)
        if issue is None:
            issue = {
                "id": f"gated-{len(issues)}",
                "type": "gated",
                "belief_id": node_id,
                "belief_text": row["text"],
                "source_files": _parse_sources(row["source"]),
                "description": "",
            }
            issues_by_node[node_id] = issue
            issues.append(issue)

        issue["description"] = (
            f"Node is IN but antecedent(s) {', '.join(out_names)} are OUT"
        )

    conn.close()
    return issues


def _find_contradictions(db_path: str) -> list[dict]:
    """Find nogoods where all member nodes are still IN."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
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

    conn.close()
    return issues


def _find_stale(db_path: str) -> list[dict]:
    """Find nodes whose source files may have changed since derivation."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, text, source, source_hash, created_at
        FROM nodes
        WHERE truth_value = 'IN'
          AND source != ''
          AND created_at != ''
    """).fetchall()
    conn.close()

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


def _find_unreviewed(db_path: str) -> list[dict]:
    """Find derived nodes that haven't been reviewed."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, text, source
        FROM nodes
        WHERE truth_value = 'IN'
          AND created_at != ''
          AND (reviewed_at IS NULL OR reviewed_at = '')
    """).fetchall()
    conn.close()

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


def _valid_timestamp(value: object) -> bool:
    """Return whether an acceptance field contains a valid ISO timestamp."""
    if not isinstance(value, str) or not value.strip():
        return False
    timestamp = value.strip()
    if "T" not in timestamp and " " not in timestamp:
        return False
    try:
        datetime.fromisoformat(timestamp)
    except ValueError:
        return False
    return True


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
