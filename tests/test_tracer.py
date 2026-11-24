"""Tests for the tracer module."""

import sqlite3

import pytest

from worker_bee.tracer import _extract_code_refs, _guess_project_dir, _resolve_ref


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


def test_extract_code_refs_preserves_commas_in_paths():
    text = "See docs/report,final.md and [file] `docs/report,final.md`."
    refs = _extract_code_refs(text)
    assert refs == ["docs/report,final.md"]


def test_extract_code_refs_normalizes_equivalent_paths():
    refs = _extract_code_refs(
        "[file] src/./main.py and [file] src//main.py and [file] src/../file.py"
    )
    assert refs == ["src/main.py", "file.py"]


def test_extract_code_refs_does_not_truncate_longer_extension():
    refs = _extract_code_refs("See src/parser.cpp for the implementation.")
    assert refs == ["src/parser.cpp"]


def test_extract_code_refs_unquoted_markers_stop_at_whitespace():
    refs = _extract_code_refs("[file] src/./main.py and [file] src/other.py")
    assert refs == ["src/main.py", "src/other.py"]


def test_extract_code_refs_rejects_extension_like_suffixes():
    refs = _extract_code_refs("See src/main.py.bak, but use src/main.py.")
    assert refs == ["src/main.py"]


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


def test_resolve_ref_prefers_project_and_returns_absolute_path(tmp_path):
    project_dir = tmp_path / "project"
    summary_dir = tmp_path / "summary"
    project_dir.mkdir()
    summary_dir.mkdir()
    project_file = project_dir / "src" / "main.py"
    project_file.parent.mkdir()
    project_file.write_text("project")
    summary_file = summary_dir / "src" / "main.py"
    summary_file.parent.mkdir()
    summary_file.write_text("summary")

    result = _resolve_ref("src/./main.py", project_dir, summary_dir)

    assert result == project_file.resolve()
    assert result.is_absolute()


def test_resolve_ref_allows_traversal_and_canonicalizes_result(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("outside")

    result = _resolve_ref("../outside.py", project_dir, tmp_path / "summary")

    assert result == outside.resolve()


def test_resolve_ref_accepts_absolute_reference(tmp_path):
    referenced = tmp_path / "absolute.py"
    referenced.write_text("absolute")

    result = _resolve_ref(str(referenced), None, tmp_path / "summary")

    assert result == referenced.resolve()


def test_resolve_ref_returns_symlink_target(tmp_path):
    target = tmp_path / "target.py"
    target.write_text("target")
    link = tmp_path / "link.py"
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are not supported")

    result = _resolve_ref("link.py", None, tmp_path)

    assert result == target.resolve()


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


def test_trace_resolves_relative_normalized_code_refs(trace_db):
    summary = trace_db.parent / "summaries" / "topic-rendering.md"
    summary.write_text("[file] src/./main.py")
    project_dir = trace_db.parent.parent / "test"
    source = project_dir / "src" / "main.py"
    source.parent.mkdir()
    source.write_text("print('ok')")

    from worker_bee.tracer import trace_belief

    result = trace_belief(trace_db, "test-belief", project_dir=project_dir)

    assert result.code_refs == ["src/main.py"]
    assert result.code_found == ["src/main.py"]
    assert "print('ok')" in result.code_sections


def test_trace_missing_belief(trace_db):
    from worker_bee.tracer import trace_belief

    result = trace_belief(trace_db, "nonexistent")
    assert result.belief_id == "nonexistent"
    assert result.belief_text == ""
    assert result.code_refs == []
