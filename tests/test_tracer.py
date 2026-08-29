"""Tests for the tracer module."""

import sqlite3

import pytest

from worker_bee.tracer import _extract_code_refs, _guess_project_dir


def test_extract_code_refs_file_markers():
    text = """## Topics
- [file] ../ia2-rust/src/app.rs — the app loop
- [file] `src/main.rs` — main entry
"""
    refs = _extract_code_refs(text)
    assert "../ia2-rust/src/app.rs" in refs
    assert "src/main.rs" in refs


def test_extract_code_refs_from_pattern():
    text = 'From `Cargo.toml` (line ~13):\n```toml\nvello = "0.7"\n```'
    refs = _extract_code_refs(text)
    assert "Cargo.toml" in refs


def test_extract_code_refs_inline_backticks():
    text = "The file `src/draw.rs` contains the draw primitives."
    refs = _extract_code_refs(text)
    assert "src/draw.rs" in refs


def test_extract_code_refs_bare_paths():
    text = "See ../ia2-rust/src/lib.rs for the full implementation."
    refs = _extract_code_refs(text)
    assert "../ia2-rust/src/lib.rs" in refs


def test_extract_code_refs_deduplicates():
    text = """From `Cargo.toml`:
```toml
[dependencies]
```
See `Cargo.toml` for details."""
    refs = _extract_code_refs(text)
    assert refs.count("Cargo.toml") == 1


def test_guess_project_dir(tmp_path):
    expert_dir = tmp_path / "wizard-rust-expert"
    expert_dir.mkdir()
    project_dir = tmp_path / "wizard-rust"
    project_dir.mkdir()
    db_path = expert_dir / "reasons.db"
    db_path.touch()

    result = _guess_project_dir(db_path)
    assert result == project_dir


def test_guess_project_dir_no_match(tmp_path):
    some_dir = tmp_path / "myproject"
    some_dir.mkdir()
    db_path = some_dir / "reasons.db"
    db_path.touch()

    result = _guess_project_dir(db_path)
    assert result is None


@pytest.fixture
def trace_db(tmp_path):
    """Create a reasons.db with a belief and a source summary."""
    db_path = tmp_path / "test-expert" / "reasons.db"
    db_path.parent.mkdir()

    from reasons.storage import Storage
    store = Storage(str(db_path))

    summary_dir = db_path.parent / "summaries"
    summary_dir.mkdir()
    summary = summary_dir / "topic-rendering.md"
    summary.write_text(
        "# Rendering\nFrom `Cargo.toml`:\n```toml\nvello = { version = \"0.7\" }\n```\n"
    )

    project_dir = tmp_path / "test"
    project_dir.mkdir()
    cargo = project_dir / "Cargo.toml"
    cargo.write_text('[dependencies]\nvello = { version = "0.7.0", default-features = false }\n')

    store.conn.execute(
        "INSERT INTO nodes (id, text, truth_value, source, created_at) VALUES (?, ?, ?, ?, ?)",
        ("test-belief", "vello uses version 0.7", "IN", str(summary), "2026-01-01T00:00:00"),
    )
    store.conn.commit()
    store.conn.close()
    return db_path


def test_trace_finds_code_refs(trace_db):
    from worker_bee.tracer import trace_belief

    project_dir = trace_db.parent.parent / "test"
    result = trace_belief(trace_db, "test-belief", project_dir=project_dir)
    assert result.belief_id == "test-belief"
    assert result.belief_text == "vello uses version 0.7"
    assert "Cargo.toml" in result.code_refs
    assert "Cargo.toml" in result.code_found
    assert result.code_sections != ""
    assert "default-features = false" in result.code_sections


def test_trace_missing_belief(trace_db):
    from worker_bee.tracer import trace_belief

    result = trace_belief(trace_db, "nonexistent")
    assert result.belief_id == "nonexistent"
    assert result.belief_text == ""
    assert result.code_refs == []
