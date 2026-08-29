"""Tests for the verifier module."""

import json
import sqlite3

import pytest

from worker_bee.verifier import _parse_verify_response, _update_db, VerifyResult


def test_parse_verify_response_valid():
    response = """<think>Let me check...</think>

```json
{
  "verified": true,
  "confidence": "high",
  "evidence": "Line 13: vello = { default-features = false }",
  "comment": "The Cargo.toml confirms default-features = false."
}
```"""
    result = _parse_verify_response(response)
    assert result["verified"] is True
    assert result["confidence"] == "high"


def test_parse_verify_response_not_verified():
    response = '{"verified": false, "confidence": "medium", "evidence": "none", "comment": "not found"}'
    result = _parse_verify_response(response)
    assert result["verified"] is False


def test_parse_verify_response_no_json():
    result = _parse_verify_response("No JSON here.")
    assert result == {}


def test_parse_verify_response_ignores_non_verify_json():
    response = '{"name": "foo"}\n{"verified": true, "confidence": "high", "evidence": "x", "comment": "y"}'
    result = _parse_verify_response(response)
    assert result["verified"] is True


@pytest.fixture
def verify_db(tmp_path):
    db_path = tmp_path / "reasons.db"

    from reasons.storage import Storage
    store = Storage(str(db_path))

    store.conn.execute(
        "INSERT INTO nodes (id, text, truth_value, source, created_at) VALUES (?, ?, ?, ?, ?)",
        ("test-belief", "some claim", "IN", "summary.md", "2026-01-01T00:00:00"),
    )
    store.conn.commit()
    store.conn.close()
    return db_path


def test_update_db_sets_verified_at(verify_db):
    result = VerifyResult(
        belief_id="test-belief",
        belief_text="some claim",
        source="summary.md",
        code_refs=["Cargo.toml"],
        code_found=["Cargo.toml"],
        code_missing=[],
        verified=True,
        confidence="high",
        evidence="line 13",
        comment="confirmed",
    )
    _update_db(str(verify_db), result)

    conn = sqlite3.connect(str(verify_db))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT verified_at, metadata_json FROM nodes WHERE id = ?",
                       ("test-belief",)).fetchone()
    assert row["verified_at"] != ""
    metadata = json.loads(row["metadata_json"])
    assert metadata["verify_result"]["verified"] is True
    assert metadata["verify_result"]["confidence"] == "high"
    assert metadata["verify_result"]["verifier"] == "worker-bee"
    assert "Cargo.toml" in metadata["verify_result"]["code_files"]
    conn.close()


def test_update_db_not_verified(verify_db):
    result = VerifyResult(
        belief_id="test-belief",
        belief_text="some claim",
        source="summary.md",
        code_refs=[],
        code_found=[],
        code_missing=[],
        verified=False,
        confidence="medium",
        evidence="not found",
        comment="no support",
    )
    _update_db(str(verify_db), result)

    conn = sqlite3.connect(str(verify_db))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT verified_at, metadata_json FROM nodes WHERE id = ?",
                       ("test-belief",)).fetchone()
    assert row["verified_at"] != ""
    metadata = json.loads(row["metadata_json"])
    assert metadata["verify_result"]["verified"] is False
    conn.close()
