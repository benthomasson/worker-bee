"""Tests for the reviewer module."""

import json
import sqlite3

import pytest

from worker_bee.reviewer import _parse_review_response, ReviewResult


def test_parse_review_response_valid():
    response = """Here are my findings:

```json
[
  {
    "id": "node-a",
    "accurate": true,
    "well_scoped": true,
    "error_type": null,
    "comment": "Accurately describes the function"
  },
  {
    "id": "node-b",
    "accurate": false,
    "well_scoped": false,
    "error_type": "overgeneralized",
    "comment": "Source says sometimes, not always"
  }
]
```"""
    results = _parse_review_response(response)
    assert len(results) == 2

    assert results[0].id == "node-a"
    assert results[0].accurate is True
    assert results[0].error_type is None

    assert results[1].id == "node-b"
    assert results[1].accurate is False
    assert results[1].error_type == "overgeneralized"


def test_parse_review_response_no_json():
    results = _parse_review_response("No JSON here at all.")
    assert results == []


def test_parse_review_response_thinking_then_json():
    response = """<think>
Let me analyze these beliefs...
</think>

[
  {
    "id": "node-a",
    "accurate": true,
    "well_scoped": true,
    "error_type": null,
    "comment": "Correct"
  }
]"""
    results = _parse_review_response(response)
    assert len(results) == 1
    assert results[0].id == "node-a"
    assert results[0].accurate is True


def test_parse_review_response_missing_fields():
    response = '[{"id": "node-a"}]'
    results = _parse_review_response(response)
    assert len(results) == 1
    assert results[0].accurate is True
    assert results[0].well_scoped is True
    assert results[0].error_type is None
    assert results[0].comment == ""


@pytest.fixture
def reasons_db(tmp_path):
    """Create a reasons.db with unreviewed beliefs and source files."""
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
    """)

    src = tmp_path / "example.py"
    src.write_text("def hello():\n    return 'world'\n")

    conn.execute(
        "INSERT INTO nodes (id, text, truth_value, source, created_at) VALUES (?, ?, ?, ?, ?)",
        ("hello-returns-world", "hello() returns 'world'", "IN", str(src), "2026-01-01T00:00:00"),
    )
    conn.execute(
        "INSERT INTO nodes (id, text, truth_value, source, created_at, reviewed_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("already-reviewed", "something", "IN", str(src), "2026-01-01T00:00:00", "2026-01-02T00:00:00"),
    )
    conn.commit()
    conn.close()
    return db_path


def test_review_dry_run(reasons_db, capsys):
    from worker_bee.reviewer import review_unreviewed
    results = review_unreviewed(reasons_db, dry_run=True)
    assert results == []
    captured = capsys.readouterr()
    assert "hello-returns-world" in captured.out


def test_update_db_sets_reviewed_at(reasons_db):
    from worker_bee.reviewer import _update_db, ReviewResult

    conn = sqlite3.connect(str(reasons_db))
    conn.row_factory = sqlite3.Row

    results = [
        ReviewResult(
            id="hello-returns-world",
            accurate=True,
            well_scoped=True,
            error_type=None,
            comment="Correct",
        )
    ]
    _update_db(conn, results)

    row = conn.execute("SELECT reviewed_at, metadata_json FROM nodes WHERE id = ?",
                       ("hello-returns-world",)).fetchone()
    assert row["reviewed_at"] != ""
    metadata = json.loads(row["metadata_json"])
    assert metadata["review_result"]["accurate"] is True
    assert metadata["review_result"]["reviewer"] == "worker-bee"
    conn.close()


def test_update_db_retracts_inaccurate(reasons_db):
    from worker_bee.reviewer import _update_db, ReviewResult

    conn = sqlite3.connect(str(reasons_db))
    conn.row_factory = sqlite3.Row

    results = [
        ReviewResult(
            id="hello-returns-world",
            accurate=False,
            well_scoped=False,
            error_type="fabricated",
            comment="Not what the source says",
        )
    ]
    _update_db(conn, results, retract_inaccurate=True)

    row = conn.execute("SELECT truth_value, retracted_at FROM nodes WHERE id = ?",
                       ("hello-returns-world",)).fetchone()
    assert row["truth_value"] == "OUT"
    assert row["retracted_at"] != ""
    conn.close()


def test_update_db_no_retract_by_default(reasons_db):
    from worker_bee.reviewer import _update_db, ReviewResult

    conn = sqlite3.connect(str(reasons_db))
    conn.row_factory = sqlite3.Row

    results = [
        ReviewResult(
            id="hello-returns-world",
            accurate=False,
            well_scoped=False,
            error_type="fabricated",
            comment="Not what the source says",
        )
    ]
    _update_db(conn, results)

    row = conn.execute("SELECT truth_value FROM nodes WHERE id = ?",
                       ("hello-returns-world",)).fetchone()
    assert row["truth_value"] == "IN"
    conn.close()
