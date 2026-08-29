"""Tests for the extractor module."""

import json
import sqlite3

import pytest

from worker_bee.extractor import extract


@pytest.fixture
def reasons_db(tmp_path):
    """Create a minimal reasons.db with the real schema."""
    db_path = tmp_path / "reasons.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            truth_value TEXT NOT NULL DEFAULT 'IN',
            supporting_justification INTEGER DEFAULT NULL,
            source TEXT DEFAULT '',
            source_url TEXT DEFAULT '',
            source_hash TEXT DEFAULT '',
            text_hash TEXT DEFAULT '',
            date TEXT DEFAULT '',
            metadata_json TEXT DEFAULT '{}',
            created_at TEXT DEFAULT '',
            updated_at TEXT DEFAULT '',
            reviewed_at TEXT DEFAULT '',
            verified_at TEXT DEFAULT '',
            retracted_at TEXT DEFAULT ''
        );

        CREATE TABLE justifications (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL REFERENCES nodes(id),
            type TEXT NOT NULL,
            antecedents_json TEXT NOT NULL DEFAULT '[]',
            outlist_json TEXT NOT NULL DEFAULT '[]',
            label TEXT DEFAULT '',
            content_hash TEXT DEFAULT ''
        );

        CREATE TABLE nogoods (
            id TEXT PRIMARY KEY,
            nodes_json TEXT NOT NULL DEFAULT '[]',
            discovered TEXT DEFAULT '',
            resolution TEXT DEFAULT ''
        );
    """)
    conn.close()
    return db_path


def test_extract_empty_db(reasons_db):
    issues = extract(reasons_db)
    assert issues == []


def test_extract_missing_db(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract(tmp_path / "nonexistent.db")


def test_extract_gated(reasons_db):
    conn = sqlite3.connect(str(reasons_db))
    conn.execute(
        "INSERT INTO nodes (id, text, truth_value) VALUES (?, ?, ?)",
        ("node-a", "Node A is true", "IN"),
    )
    conn.execute(
        "INSERT INTO nodes (id, text, truth_value) VALUES (?, ?, ?)",
        ("node-b", "Node B is true", "OUT"),
    )
    conn.execute(
        "INSERT INTO justifications (node_id, type, antecedents_json) VALUES (?, ?, ?)",
        ("node-a", "SL", json.dumps(["node-b"])),
    )
    conn.commit()
    conn.close()

    issues = extract(reasons_db, types=["gated"])
    assert len(issues) == 1
    assert issues[0]["type"] == "gated"
    assert issues[0]["belief_id"] == "node-a"
    assert "node-b" in issues[0]["description"]


def test_extract_gated_skips_all_in(reasons_db):
    conn = sqlite3.connect(str(reasons_db))
    conn.execute(
        "INSERT INTO nodes (id, text, truth_value) VALUES (?, ?, ?)",
        ("node-a", "Node A", "IN"),
    )
    conn.execute(
        "INSERT INTO nodes (id, text, truth_value) VALUES (?, ?, ?)",
        ("node-b", "Node B", "IN"),
    )
    conn.execute(
        "INSERT INTO justifications (node_id, type, antecedents_json) VALUES (?, ?, ?)",
        ("node-a", "SL", json.dumps(["node-b"])),
    )
    conn.commit()
    conn.close()

    issues = extract(reasons_db, types=["gated"])
    assert len(issues) == 0


def test_extract_contradiction(reasons_db):
    conn = sqlite3.connect(str(reasons_db))
    conn.execute(
        "INSERT INTO nodes (id, text, truth_value) VALUES (?, ?, ?)",
        ("node-a", "Yes", "IN"),
    )
    conn.execute(
        "INSERT INTO nodes (id, text, truth_value) VALUES (?, ?, ?)",
        ("node-b", "No", "IN"),
    )
    conn.execute(
        "INSERT INTO nogoods (id, nodes_json) VALUES (?, ?)",
        ("nogood-a-b", json.dumps(["node-a", "node-b"])),
    )
    conn.commit()
    conn.close()

    issues = extract(reasons_db, types=["contradiction"])
    assert len(issues) == 1
    assert issues[0]["type"] == "contradiction"
    assert "node-a" in issues[0]["belief_id"]
    assert "node-b" in issues[0]["belief_id"]


def test_extract_contradiction_skips_resolved(reasons_db):
    conn = sqlite3.connect(str(reasons_db))
    conn.execute(
        "INSERT INTO nodes (id, text, truth_value) VALUES (?, ?, ?)",
        ("node-a", "Yes", "IN"),
    )
    conn.execute(
        "INSERT INTO nodes (id, text, truth_value) VALUES (?, ?, ?)",
        ("node-b", "No", "OUT"),
    )
    conn.execute(
        "INSERT INTO nogoods (id, nodes_json) VALUES (?, ?)",
        ("nogood-a-b", json.dumps(["node-a", "node-b"])),
    )
    conn.commit()
    conn.close()

    issues = extract(reasons_db, types=["contradiction"])
    assert len(issues) == 0


def test_extract_unreviewed(reasons_db):
    conn = sqlite3.connect(str(reasons_db))
    conn.execute(
        "INSERT INTO nodes (id, text, truth_value, created_at, reviewed_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("node-a", "Derived thing", "IN", "2026-01-01T00:00:00", ""),
    )
    conn.commit()
    conn.close()

    issues = extract(reasons_db, types=["unreviewed"])
    assert len(issues) == 1
    assert issues[0]["type"] == "unreviewed"


def test_extract_unreviewed_skips_reviewed(reasons_db):
    conn = sqlite3.connect(str(reasons_db))
    conn.execute(
        "INSERT INTO nodes (id, text, truth_value, created_at, reviewed_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("node-a", "Reviewed thing", "IN", "2026-01-01T00:00:00", "2026-01-02T00:00:00"),
    )
    conn.commit()
    conn.close()

    issues = extract(reasons_db, types=["unreviewed"])
    assert len(issues) == 0


def test_extract_type_filter(reasons_db):
    conn = sqlite3.connect(str(reasons_db))
    conn.execute(
        "INSERT INTO nodes (id, text, truth_value, created_at, reviewed_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("node-a", "Derived", "IN", "2026-01-01T00:00:00", ""),
    )
    conn.commit()
    conn.close()

    issues = extract(reasons_db, types=["gated"])
    assert len(issues) == 0
