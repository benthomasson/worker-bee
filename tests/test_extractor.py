"""Tests for the extractor module."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from worker_bee.extractor import extract


@pytest.fixture
def reasons_db(tmp_path):
    """Create a minimal reasons.db with test data."""
    db_path = tmp_path / "reasons.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE beliefs (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            text TEXT,
            status TEXT NOT NULL DEFAULT 'in',
            source TEXT,
            derived_at TEXT,
            reviewed_at TEXT
        );

        CREATE TABLE justifications (
            id INTEGER PRIMARY KEY,
            belief_id INTEGER REFERENCES beliefs(id)
        );

        CREATE TABLE justification_antecedents (
            id INTEGER PRIMARY KEY,
            justification_id INTEGER REFERENCES justifications(id),
            antecedent_id INTEGER REFERENCES beliefs(id)
        );

        CREATE TABLE nogoods (
            id INTEGER PRIMARY KEY
        );

        CREATE TABLE nogood_members (
            id INTEGER PRIMARY KEY,
            nogood_id INTEGER REFERENCES nogoods(id),
            belief_id INTEGER REFERENCES beliefs(id)
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
    conn.execute("INSERT INTO beliefs (id, name, text, status) VALUES (1, 'b1', 'first', 'in')")
    conn.execute("INSERT INTO beliefs (id, name, text, status) VALUES (2, 'b2', 'second', 'out')")
    conn.execute("INSERT INTO justifications (id, belief_id) VALUES (1, 1)")
    conn.execute("INSERT INTO justification_antecedents (id, justification_id, antecedent_id) VALUES (1, 1, 2)")
    conn.commit()
    conn.close()

    issues = extract(reasons_db, types=["gated"])
    assert len(issues) == 1
    assert issues[0]["type"] == "gated"
    assert issues[0]["belief_id"] == "b1"


def test_extract_contradiction(reasons_db):
    conn = sqlite3.connect(str(reasons_db))
    conn.execute("INSERT INTO beliefs (id, name, text, status) VALUES (1, 'b1', 'yes', 'in')")
    conn.execute("INSERT INTO beliefs (id, name, text, status) VALUES (2, 'b2', 'no', 'in')")
    conn.execute("INSERT INTO nogoods (id) VALUES (1)")
    conn.execute("INSERT INTO nogood_members (id, nogood_id, belief_id) VALUES (1, 1, 1)")
    conn.execute("INSERT INTO nogood_members (id, nogood_id, belief_id) VALUES (2, 1, 2)")
    conn.commit()
    conn.close()

    issues = extract(reasons_db, types=["contradiction"])
    assert len(issues) == 1
    assert issues[0]["type"] == "contradiction"
    assert "b1" in issues[0]["belief_id"]
    assert "b2" in issues[0]["belief_id"]


def test_extract_unreviewed(reasons_db):
    conn = sqlite3.connect(str(reasons_db))
    conn.execute(
        "INSERT INTO beliefs (id, name, text, status, derived_at, reviewed_at) "
        "VALUES (1, 'b1', 'derived thing', 'in', '2026-01-01T00:00:00', NULL)"
    )
    conn.commit()
    conn.close()

    issues = extract(reasons_db, types=["unreviewed"])
    assert len(issues) == 1
    assert issues[0]["type"] == "unreviewed"


def test_extract_type_filter(reasons_db):
    conn = sqlite3.connect(str(reasons_db))
    conn.execute(
        "INSERT INTO beliefs (id, name, text, status, derived_at, reviewed_at) "
        "VALUES (1, 'b1', 'derived', 'in', '2026-01-01T00:00:00', NULL)"
    )
    conn.commit()
    conn.close()

    issues = extract(reasons_db, types=["gated"])
    assert len(issues) == 0
