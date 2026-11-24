"""Tests for the verifier module."""

import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from worker_bee.verifier import (
    VERIFY_SYSTEM_PROMPT,
    VerifyResult,
    _parse_verify_response,
    _update_db,
)


def test_verify_prompt_requires_raw_json_comment_field():
    assert '"comment"' in VERIFY_SYSTEM_PROMPT
    assert '"justification"' not in VERIFY_SYSTEM_PROMPT
    assert "one raw JSON object" in VERIFY_SYSTEM_PROMPT
    assert "```" not in VERIFY_SYSTEM_PROMPT


def test_parse_verify_response_valid():
    response = json.dumps({
        "verified": True,
        "confidence": "high",
        "evidence": "Line 13: vello = { default-features = false }",
        "comment": "The Cargo.toml confirms default-features = false.",
    })
    result = _parse_verify_response(response)
    assert result["verified"] is True
    assert result["confidence"] == "high"


def test_parse_verify_response_not_verified():
    response = '{"verified": false, "confidence": "medium", "evidence": "none", "comment": "not found"}'
    result = _parse_verify_response(response)
    assert result["verified"] is False


def test_parse_verify_response_allows_json_whitespace():
    response = '\n\t {"verified": true, "confidence": "high", "evidence": "found", "comment": "confirmed"} \r\n'
    result = _parse_verify_response(response)
    assert result["verified"] is True


def test_parse_verify_response_no_json():
    result = _parse_verify_response("No JSON here.")
    assert result == {}


@pytest.mark.parametrize(
    "response",
    [
        '<think>reasoning</think> {"verified": true}',
        '```json\\n{"verified": true}\\n```',
        '{"verified": true} trailing text',
        '{"name": "foo"}\\n{"verified": true}',
    ],
)
def test_parse_verify_response_rejects_surrounding_content(response):
    assert _parse_verify_response(response) == {}


def test_verify_belief_returns_undecided_for_invalid_response_without_writing(verify_db):
    from worker_bee import verifier

    trace = SimpleNamespace(
        belief_id="test-belief", belief_text="some claim", source="summary.md",
        code_refs=[], code_found=[], code_missing=[], summary_text="",
    )
    session = SimpleNamespace(steps=[SimpleNamespace(
        text='{"verified": true, "confidence": [], "evidence": "bad", "comment": "bad"}'
    )])
    with patch.object(verifier, "trace_belief", return_value=trace), \
         patch.object(verifier, "_guess_project_dir", return_value=None), \
         patch.object(verifier, "_update_db") as update, \
         patch("worker_bee.editor.run_edit_loop", return_value=session):
        result = verifier.verify_belief(str(verify_db), "test-belief")

    assert result.verified is None
    assert result.evidence == ""
    assert result.comment == ""
    update.assert_not_called()

    conn = sqlite3.connect(str(verify_db))
    row = conn.execute(
        "SELECT verified_at, metadata_json FROM nodes WHERE id = ?", ("test-belief",)
    ).fetchone()
    assert row == (None, None)
    conn.close()


def test_verify_unverified_continues_after_invalid_response(verify_db):
    from worker_bee import verifier

    def make_trace(_db_path, belief_id, **_kwargs):
        return SimpleNamespace(
            belief_id=belief_id, belief_text=belief_id, source="summary.md",
            code_refs=[], code_found=[], code_missing=[], summary_text="",
        )

    sessions = [
        SimpleNamespace(steps=[SimpleNamespace(text="not JSON")]),
        SimpleNamespace(steps=[SimpleNamespace(
            text='{"verified": true, "confidence": "high", '
                 '"evidence": "found", "comment": "confirmed"}'
        )]),
    ]
    with patch.object(verifier, "_find_unverified_gated",
                      return_value=["first-belief", "second-belief"]), \
         patch.object(verifier, "trace_belief", side_effect=make_trace), \
         patch.object(verifier, "_guess_project_dir", return_value=None), \
         patch.object(verifier, "_update_db") as update, \
         patch("worker_bee.editor.run_edit_loop", side_effect=sessions) as run_loop:
        results = verifier.verify_unverified(str(verify_db))

    assert [result.belief_id for result in results] == [
        "first-belief", "second-belief",
    ]
    assert results[0].verified is None
    assert results[1].verified is True
    assert run_loop.call_count == 2
    update.assert_called_once()


def test_verify_belief_sends_unwrapped_response_instruction():
    from worker_bee import verifier

    trace = SimpleNamespace(
        belief_id="test-belief", belief_text="some claim", source="summary.md",
        code_refs=[], code_found=[], code_missing=[], summary_text="summary",
    )
    session = SimpleNamespace(steps=[SimpleNamespace(
        text='{"verified": true, "confidence": "high", "evidence": "found", "comment": "confirmed"}'
    )])
    with patch.object(verifier, "trace_belief", return_value=trace), \
         patch.object(verifier, "_guess_project_dir", return_value=None), \
         patch.object(verifier, "_update_db"), \
         patch("worker_bee.editor.run_edit_loop", return_value=session) as run_loop:
        verifier.verify_belief(str("reasons.db"), "test-belief")

    prompt = run_loop.call_args.args[0]
    assert "one raw JSON object" in prompt
    assert "no Markdown fences" in prompt
    assert "```" not in prompt


@pytest.mark.parametrize("value", ['"true"', "1", "null", "[]", "{}"])
def test_parse_verify_response_rejects_non_boolean_verdict(value):
    response = f'{{"verified": {value}, "evidence": "invalid"}}'
    assert _parse_verify_response(response) == {}


def test_parse_verify_response_rejects_invalid_verdict_with_later_valid_object():
    response = (
        '{"verified": "true"}\n'
        'The actual result is {"verified": false, "confidence": "low", '
        '"evidence": "unsupported", "comment": "not confirmed"}'
    )
    assert _parse_verify_response(response) == {}


@pytest.mark.parametrize(
    "field,value",
    [
        ("confidence", "[]"),
        ("confidence", '"certain"'),
        ("evidence", "{}"),
        ("comment", "123"),
    ],
)
def test_parse_verify_response_rejects_invalid_result_field_types(field, value):
    response = (
        '{"verified": true, "confidence": "high", "evidence": "found", '
        f'"comment": "confirmed", "{field}": {value}}}'
    )
    assert _parse_verify_response(response) == {}


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
    verify_result = metadata["verify_result"]
    assert set(verify_result) == {
        "verified", "confidence", "evidence", "comment", "code_files", "verifier",
    }
    assert verify_result["verified"] is True
    assert verify_result["confidence"] == "high"
    assert verify_result["comment"] == "confirmed"
    assert verify_result["verifier"] == "worker-bee"
    assert "Cargo.toml" in verify_result["code_files"]
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
    truth = conn.execute("SELECT truth_value FROM nodes WHERE id = ?",
                         ("test-belief",)).fetchone()[0]
    assert truth == "IN"
    conn.close()


def test_verify_belief_retracts_negative_verdict_when_requested(verify_db):
    from worker_bee import verifier

    trace = SimpleNamespace(
        belief_id="test-belief", belief_text="some claim", source="summary.md",
        code_refs=[], code_found=[], code_missing=[], summary_text="",
    )
    session = SimpleNamespace(steps=[SimpleNamespace(
        text='{"verified": false, "confidence": "low", "evidence": "not found", "comment": "no support"}'
    )])
    with patch.object(verifier, "trace_belief", return_value=trace), \
         patch.object(verifier, "_guess_project_dir", return_value=None), \
         patch.object(verifier, "_update_db"), \
         patch("worker_bee.editor.run_edit_loop", return_value=session), \
         patch("reasons.api.retract_node") as retract:
        verifier.verify_belief(str(verify_db), "test-belief", retract_inaccurate=True)

    retract.assert_called_once()
    assert retract.call_args.kwargs["db_path"] == str(verify_db)


def test_verify_belief_retracts_and_persists_metadata_end_to_end(verify_db):
    """The real reasons API changes truth and preserves verification metadata."""
    from reasons.api import add_justification, add_node, show_node
    from worker_bee import verifier

    add_node("dependent", "dependent claim", db_path=str(verify_db))
    add_justification("dependent", sl="test-belief", db_path=str(verify_db))

    trace = SimpleNamespace(
        belief_id="test-belief", belief_text="some claim", source="summary.md",
        code_refs=[], code_found=["source.py"], code_missing=[], summary_text="",
    )
    session = SimpleNamespace(steps=[SimpleNamespace(
        text='{"verified": false, "confidence": "high", '
             '"evidence": "not found", "comment": "no support"}'
    )])
    with patch.object(verifier, "trace_belief", return_value=trace), \
         patch.object(verifier, "_guess_project_dir", return_value=None), \
         patch("worker_bee.editor.run_edit_loop", return_value=session):
        verifier.verify_belief(str(verify_db), "test-belief", retract_inaccurate=True)

    node = show_node("test-belief", db_path=str(verify_db))
    dependent = show_node("dependent", db_path=str(verify_db))
    assert node["truth_value"] == "OUT"
    assert dependent["truth_value"] == "OUT"
    assert node["verified_at"]
    assert node["metadata"]["verify_result"] == {
        "verified": False,
        "confidence": "high",
        "evidence": "not found",
        "comment": "no support",
        "code_files": ["source.py"],
        "verifier": "worker-bee",
    }


def test_verify_belief_does_not_record_failed_retraction(verify_db):
    from worker_bee import verifier

    trace = SimpleNamespace(
        belief_id="test-belief", belief_text="some claim", source="summary.md",
        code_refs=[], code_found=[], code_missing=[], summary_text="",
    )
    session = SimpleNamespace(steps=[SimpleNamespace(
        text='{"verified": false, "confidence": "low", "evidence": "not found", "comment": "no support"}'
    )])
    with patch.object(verifier, "trace_belief", return_value=trace), \
         patch.object(verifier, "_guess_project_dir", return_value=None), \
         patch.object(verifier, "_update_db") as update, \
         patch("worker_bee.editor.run_edit_loop", return_value=session), \
         patch("reasons.api.retract_node", side_effect=RuntimeError("retraction failed")):
        with pytest.raises(RuntimeError, match="retraction failed"):
            verifier.verify_belief(str(verify_db), "test-belief", retract_inaccurate=True)

    update.assert_not_called()
