"""Tests for the retract_belief and supersede_belief tools.

These tools are the LLM-facing complement to ``bee review --retract``: they
let the worker bee retract an outdated belief or replace it with a corrected
version, both through the ftl-reasons (``reasons``) API against the local
(writable) belief database.
"""

import pytest

import worker_bee.tools as tools
from worker_bee.tools import (
    BELIEF_TOOLS,
    set_belief_db,
    execute_tool,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def local_db(tmp_path):
    """A writable local (brain) belief database with a few beliefs."""
    from reasons.api import init_db, add_node

    db_path = str(tmp_path / "brain.db")
    init_db(db_path=db_path)
    add_node("notes-are-persisted", "Notes persist across sessions.",
             source="worker_bee/tools.py", db_path=db_path)
    add_node("legacy-blob-store", "Blob store keeps 30 days of history.",
             source="worker_bee/README.md", db_path=db_path)
    return db_path


@pytest.fixture
def hive_db(tmp_path):
    """A read-only hive belief database (shared project beliefs)."""
    from reasons.api import init_db, add_node

    db_path = str(tmp_path / "hive.db")
    init_db(db_path=db_path)
    add_node("hive-shared-fact", "The hive is shared and read-only.",
             source="worker_bee/README.md", db_path=db_path)
    return db_path


@pytest.fixture
def store(local_db, hive_db):
    """Point the global BeliefStore at the two layers and reset it after.

    ``set_belief_db(db_path, brain_path)`` → db_path is the read-only hive,
    brain_path is the writable local layer. So local=local_db, hive=hive_db.
    """
    set_belief_db(hive_db, brain_path=local_db)
    yield local_db
    # Reset to a clean default so other tests are unaffected.
    set_belief_db(None)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_retract_and_supersede_schemas_are_registered():
    names = [t["name"] for t in BELIEF_TOOLS]
    assert "retract_belief" in names
    assert "supersede_belief" in names

    retract = next(t for t in BELIEF_TOOLS if t["name"] == "retract_belief")
    assert retract["input_schema"]["required"] == ["id"]
    assert "reason" in retract["input_schema"]["properties"]

    supersede = next(t for t in BELIEF_TOOLS if t["name"] == "supersede_belief")
    assert supersede["input_schema"]["required"] == ["id", "text"]
    assert "new_id" in supersede["input_schema"]["properties"]


def test_dispatch_routes_new_tools():
    # Both tools must be wired into execute_tool (not "Unknown tool").
    # Use a no-store config so we exercise the routing path deterministically.
    set_belief_db(None)
    try:
        r1 = execute_tool("retract_belief", {"id": "whatever"})
        r2 = execute_tool("supersede_belief", {"id": "whatever", "text": "x"})
        assert "no belief database" in r1
        assert "no belief database" in r2
    finally:
        set_belief_db(None)


# ---------------------------------------------------------------------------
# retract_belief
# ---------------------------------------------------------------------------

def test_retract_belief_sets_out_and_records_reason(store):
    from reasons.api import show_node

    out = execute_tool("retract_belief",
                       {"id": "legacy-blob-store", "reason": "stale — now 90 days"})
    assert "OUT" in out
    assert "stale" in out

    node = show_node("legacy-blob-store", db_path=store)
    assert node["truth_value"] == "OUT"
    assert node["metadata"].get("retract_reason") == "stale — now 90 days"


def test_retract_belief_without_reason(store):
    from reasons.api import show_node

    out = execute_tool("retract_belief", {"id": "legacy-blob-store"})
    assert "OUT" in out

    node = show_node("legacy-blob-store", db_path=store)
    assert node["truth_value"] == "OUT"
    # No reason supplied → no retract_reason key is written.
    assert "retract_reason" not in (node["metadata"] or {})


def test_retract_belief_cascades_to_dependent(store):
    from reasons.api import add_node, add_justification, show_node

    # A consequence that depends on the premise we will retract.
    add_node("depends-on-legacy", "The API supports blob export.",
             db_path=store)
    add_justification("depends-on-legacy", sl="legacy-blob-store", db_path=store)
    assert show_node("depends-on-legacy", db_path=store)["truth_value"] == "IN"

    out = execute_tool("retract_belief", {"id": "legacy-blob-store"})

    # The premise goes OUT, and the dependent cascade is reported.
    assert show_node("legacy-blob-store", db_path=store)["truth_value"] == "OUT"
    assert show_node("depends-on-legacy", db_path=store)["truth_value"] == "OUT"
    assert "Cascaded OUT" in out and "depends-on-legacy" in out


def test_retract_belief_missing_id(store):
    out = execute_tool("retract_belief", {"id": "no-such-belief"})
    assert "not found" in out


def test_retract_belief_hive_only_is_refused(store):
    out = execute_tool("retract_belief", {"id": "hive-shared-fact"})
    assert "hive" in out
    from reasons.api import show_node
    # The hive belief is untouched.
    assert show_node("hive-shared-fact", db_path=tools._belief_store.hive_path)["truth_value"] == "IN"


# ---------------------------------------------------------------------------
# supersede_belief
# ---------------------------------------------------------------------------

def test_supersede_belief_replaces_old_with_new(store):
    from reasons.api import show_node

    out = execute_tool("supersede_belief", {
        "id": "legacy-blob-store",
        "text": "Blob store keeps 90 days of history.",
    })
    assert "90 days" not in out  # the text isn't echoed, only the ids
    assert "legacy-blob-store-v2" in out

    old = show_node("legacy-blob-store", db_path=store)
    new = show_node("legacy-blob-store-v2", db_path=store)
    assert old["truth_value"] == "OUT"
    assert new["truth_value"] == "IN"
    # Supersession recorded on both sides.
    assert old["metadata"].get("superseded_by") == "legacy-blob-store-v2"
    assert "legacy-blob-store" in (new["metadata"].get("supersedes") or [])


def test_supersede_belief_respects_explicit_new_id(store):
    from reasons.api import show_node

    out = execute_tool("supersede_belief", {
        "id": "legacy-blob-store",
        "text": "Blob store keeps 90 days of history.",
        "new_id": "blob-retention-v3",
    })
    assert "blob-retention-v3" in out
    assert show_node("blob-retention-v3", db_path=store)["truth_value"] == "IN"
    assert show_node("legacy-blob-store", db_path=store)["truth_value"] == "OUT"


def test_supersede_belief_auto_id_bumps_on_collision(store):
    from reasons.api import add_node, show_node

    # Occupy the default target id so supersede must fall back to -v3.
    add_node("legacy-blob-store-v2", "A squatter.", db_path=store)

    out = execute_tool("supersede_belief", {
        "id": "legacy-blob-store",
        "text": "Blob store keeps 90 days of history.",
    })
    # Should have skipped the squatted -v2 and used -v3.
    assert "legacy-blob-store-v3" in out
    assert show_node("legacy-blob-store-v3", db_path=store)["truth_value"] == "IN"
    # The squatter is left alone.
    assert show_node("legacy-blob-store-v2", db_path=store)["truth_value"] == "IN"
    assert show_node("legacy-blob-store", db_path=store)["truth_value"] == "OUT"


def test_supersede_belief_missing_id(store):
    out = execute_tool("supersede_belief", {"id": "no-such-belief", "text": "x"})
    assert "not found" in out


def test_supersede_belief_empty_text(store):
    out = execute_tool("supersede_belief", {"id": "legacy-blob-store", "text": "   "})
    assert "empty" in out


def test_supersede_belief_hive_only_is_refused(store):
    out = execute_tool("supersede_belief", {"id": "hive-shared-fact", "text": "x"})
    assert "hive" in out
    from reasons.api import show_node
    assert show_node("hive-shared-fact", db_path=tools._belief_store.hive_path)["truth_value"] == "IN"
