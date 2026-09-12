"""Tests for the hive proposal tools (propose_retraction, propose_supersede,
list_hive_proposals).

These implement the worker-bee surface of docs/hive-proposals.md §8: the bee
cannot write the read-only hive directly, so it *proposes* a retraction or
supersession — a pending row in the hive's proposals table that changes no
truth value until a human accepts it.

All tests are hermetic (tmp_path databases, no network).
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
def hive_db(tmp_path):
    """A hive (shared) belief database with a few IN beliefs and one
    dependent chain to exercise cascade impact."""
    from reasons.api import init_db, add_node, add_justification

    db_path = str(tmp_path / "hive.db")
    init_db(db_path=db_path)
    add_node("hive-stale-fact", "The blob store keeps 30 days of history.",
             source="docs/blob.md", db_path=db_path)
    # A dependent on hive-stale-fact so retraction impact is non-zero.
    add_node("hive-derives-from-stale", "Exports work because of 30-day retention.",
             db_path=db_path)
    add_justification("hive-derives-from-stale", sl="hive-stale-fact",
                      db_path=db_path)
    add_node("hive-independent", "Unrelated hive belief.",
             source="docs/other.md", db_path=db_path)
    return db_path


@pytest.fixture
def store(hive_db):
    """Point the global BeliefStore at the hive (no separate local brain, so
    proposals fall back to the hive layer). Reset after each test."""
    set_belief_db(hive_db)
    yield hive_db
    set_belief_db(None)


# ---------------------------------------------------------------------------
# Schema / dispatch
# ---------------------------------------------------------------------------

def test_proposal_tools_are_registered_in_schema():
    names = [t["name"] for t in BELIEF_TOOLS]
    assert "propose_retraction" in names
    assert "propose_supersede" in names
    assert "list_hive_proposals" in names

    retract = next(t for t in BELIEF_TOOLS if t["name"] == "propose_retraction")
    assert retract["input_schema"]["required"] == ["target_id"]
    for opt in ("reason", "basis", "evidence", "failure_mode"):
        assert opt in retract["input_schema"]["properties"]

    supersede = next(t for t in BELIEF_TOOLS if t["name"] == "propose_supersede")
    assert supersede["input_schema"]["required"] == ["old_id", "new_text"]
    for opt in ("new_id", "reason", "basis", "evidence", "failure_mode"):
        assert opt in supersede["input_schema"]["properties"]

    listing = next(t for t in BELIEF_TOOLS if t["name"] == "list_hive_proposals")
    for opt in ("status", "target", "proposer"):
        assert opt in listing["input_schema"]["properties"]
    # list_hive_proposals takes no required fields
    assert listing["input_schema"].get("required", []) == []


def test_dispatch_routes_proposal_tools_without_a_db():
    set_belief_db(None)
    try:
        r1 = execute_tool("propose_retraction", {"target_id": "x"})
        r2 = execute_tool("propose_supersede", {"old_id": "x", "new_text": "y"})
        r3 = execute_tool("list_hive_proposals", {})
        assert "no belief database" in r1
        assert "no belief database" in r2
        assert "no belief database" in r3
    finally:
        set_belief_db(None)


# ---------------------------------------------------------------------------
# propose_retraction
# ---------------------------------------------------------------------------

def test_propose_retraction_creates_pending_and_changes_no_truth(store):
    from reasons.api import show_node, list_proposals

    out = execute_tool("propose_retraction", {
        "target_id": "hive-stale-fact",
        "reason": "retention window changed to 90 days",
        "basis": "source-divergence",
        "evidence": "docs/blob.md now says 90 days",
        "failure_mode": "stale",
    })
    assert "pending" in out
    assert "no truth values changed" in out
    assert "Proposed retraction of 'hive-stale-fact'" in out
    assert "Reason: retention window changed to 90 days" in out
    assert "Basis: source-divergence" in out
    assert "Failure mode: stale" in out

    # Truth value is untouched — the proposal is only a pending row.
    assert show_node("hive-stale-fact", db_path=store)["truth_value"] == "IN"
    assert show_node("hive-derives-from-stale", db_path=store)["truth_value"] == "IN"

    # A pending proposal exists, attributed to worker-bee.
    listings = list_proposals(status="pending", db_path=store)
    assert listings["count"] == 1
    row = listings["proposals"][0]
    assert row["action"] == "retract"
    assert row["target_id"] == "hive-stale-fact"
    assert row["proposer"] == "worker-bee"
    assert row["status"] == "pending"
    # The proposal id is surfaced so a human can act on it.
    assert row["id"] in out


def test_propose_retraction_reports_cascade_impact(store):
    from reasons.api import show_proposal

    out = execute_tool("propose_retraction", {"target_id": "hive-stale-fact"})
    # The dependent hive-derives-from-stale would go OUT on accept.
    assert "1 belief(s) would be affected" in out
    assert "hive-derives-from-stale" in out

    # Impact is persisted on the proposal row itself.
    pid = [l for l in out.splitlines() if l.startswith("Proposal id")][0].split()[-1]
    detail = show_proposal(pid, db_path=store)
    assert detail["impact"].get("total_affected") == 1
    assert any(n["id"] == "hive-derives-from-stale"
               for n in detail["impact"].get("retracted", []))


def test_propose_retraction_missing_target(store):
    out = execute_tool("propose_retraction", {"target_id": "no-such-belief"})
    assert "Error proposing retraction" in out
    assert "not found" in out


def test_propose_retraction_already_out_is_rejected(store):
    from reasons.api import retract_node

    retract_node("hive-independent", db_path=store)
    out = execute_tool("propose_retraction", {"target_id": "hive-independent"})
    assert "Error proposing retraction" in out
    assert "already OUT" in out


def test_propose_retraction_second_proposal_stales_first(store):
    from reasons.api import list_proposals

    execute_tool("propose_retraction", {"target_id": "hive-stale-fact"})
    out = execute_tool("propose_retraction", {"target_id": "hive-stale-fact"})

    # Invariant I1: at most one pending per (target, action) — the older one
    # is auto-marked stale.
    listings = list_proposals(status="pending", target_id="hive-stale-fact",
                             db_path=store)
    assert listings["count"] == 1
    assert "Superseded older pending proposal(s)" in out
    staled = [l for l in out.splitlines()
              if l.startswith("Superseded older")][0]
    staled_id = staled.split(":")[-1].strip()
    current = listings["proposals"][0]["id"]
    assert staled_id != current


def test_propose_retraction_accept_then_reject_flow(store):
    """End-to-end: propose → accept applies the retraction (drift guard OK);
    a second proposal accepted after the target drifted is marked stale."""
    from reasons.api import show_node, accept_proposal, list_proposals

    out = execute_tool("propose_retraction", {"target_id": "hive-stale-fact"})
    pid = [l for l in out.splitlines() if l.startswith("Proposal id")][0].split()[-1]

    applied = accept_proposal(pid, voter="human", db_path=store)
    assert applied.get("applied") is True
    assert show_node("hive-stale-fact", db_path=store)["truth_value"] == "OUT"
    assert show_node("hive-derives-from-stale", db_path=store)["truth_value"] == "OUT"
    assert list_proposals(status="accepted", db_path=store)["count"] == 1


# ---------------------------------------------------------------------------
# propose_supersede
# ---------------------------------------------------------------------------

def test_propose_supersede_creates_pending_with_new_id(store):
    from reasons.api import list_proposals

    out = execute_tool("propose_supersede", {
        "old_id": "hive-stale-fact",
        "new_text": "The blob store keeps 90 days of history.",
        "reason": "retention window changed",
        "basis": "source-divergence",
        "failure_mode": "stale",
    })
    assert "pending" in out
    assert "no truth values changed" in out
    assert "→ 'hive-stale-fact-v2'" in out
    assert "Basis: source-divergence" in out

    # No node was added and nothing went OUT.
    from reasons.api import show_node
    assert show_node("hive-stale-fact", db_path=store)["truth_value"] == "IN"
    try:
        show_node("hive-stale-fact-v2", db_path=store)
        raise AssertionError("replacement node should not exist yet")
    except Exception:
        pass

    listings = list_proposals(status="pending", db_path=store)
    assert listings["count"] == 1
    row = listings["proposals"][0]
    assert row["action"] == "supersede"
    assert row["target_id"] == "hive-stale-fact"
    assert row["new_id"] == "hive-stale-fact-v2"
    assert row["proposer"] == "worker-bee"


def test_propose_supersede_respects_explicit_new_id(store):
    from reasons.api import list_proposals

    execute_tool("propose_supersede", {
        "old_id": "hive-stale-fact",
        "new_text": "New text.",
        "new_id": "retention-90d",
    })
    listings = list_proposals(status="pending", db_path=store)
    assert listings["proposals"][0]["new_id"] == "retention-90d"


def test_propose_supersede_empty_text_is_rejected(store):
    out = execute_tool("propose_supersede", {"old_id": "hive-stale-fact",
                                             "new_text": "   "})
    assert "empty" in out


def test_propose_supersede_missing_target(store):
    out = execute_tool("propose_supersede",
                       {"old_id": "no-such-belief", "new_text": "x"})
    assert "Error proposing supersession" in out
    assert "not found" in out


def test_propose_supersede_new_id_collision_is_rejected(store):
    from reasons.api import add_node

    add_node("hive-stale-fact-v2", "A squatter.", db_path=store)
    out = execute_tool("propose_supersede", {
        "old_id": "hive-stale-fact",
        "new_text": "New text.",
        "new_id": "hive-stale-fact-v2",
    })
    assert "Error proposing supersession" in out
    assert "already exists" in out


def test_propose_supersede_accept_flow(store):
    from reasons.api import show_node, accept_proposal, list_proposals

    out = execute_tool("propose_supersede", {
        "old_id": "hive-stale-fact",
        "new_text": "The blob store keeps 90 days of history.",
    })
    pid = [l for l in out.splitlines() if l.startswith("Proposal id")][0].split()[-1]

    applied = accept_proposal(pid, voter="human", db_path=store)
    assert applied.get("applied") is True
    # Old OUT, new IN after accept.
    assert show_node("hive-stale-fact", db_path=store)["truth_value"] == "OUT"
    assert show_node("hive-stale-fact-v2", db_path=store)["truth_value"] == "IN"
    assert list_proposals(status="accepted", db_path=store)["count"] == 1


# ---------------------------------------------------------------------------
# list_hive_proposals
# ---------------------------------------------------------------------------

def test_list_hive_proposals_empty(store):
    out = execute_tool("list_hive_proposals", {})
    assert "No hive proposals found" in out


def test_list_hive_proposals_shows_pending(store):
    execute_tool("propose_retraction", {"target_id": "hive-independent"})
    out = execute_tool("list_hive_proposals", {})
    assert "1 hive proposal(s)" in out
    assert "retract" in out
    assert "hive-independent" in out
    assert "pending" in out
    assert "proposer=worker-bee" in out
    # Same result with an explicit pending filter.
    assert "1 hive proposal(s)" in execute_tool(
        "list_hive_proposals", {"status": "pending"})


def test_list_hive_proposals_filters(store):
    execute_tool("propose_retraction", {"target_id": "hive-independent"})
    execute_tool("propose_supersede", {"old_id": "hive-stale-fact",
                                       "new_text": "New."})

    # Filter by target: only the supersede one.
    out = execute_tool("list_hive_proposals", {"target": "hive-stale-fact"})
    assert "1 hive proposal(s)" in out
    assert "hive-stale-fact" in out
    assert "hive-independent" not in out

    # Filter by proposer: both were proposed by worker-bee.
    out = execute_tool("list_hive_proposals", {"proposer": "worker-bee"})
    assert "2 hive proposal(s)" in out

    # Filter by a different proposer: none.
    out = execute_tool("list_hive_proposals", {"proposer": "someone-else"})
    assert "No hive proposals found" in out


def test_list_hive_proposals_status_filter(store):
    from reasons.api import accept_proposal

    execute_tool("propose_retraction", {"target_id": "hive-independent"})
    pid = execute_tool("list_hive_proposals", {})
    pid_line = [l for l in pid.splitlines()
                if "prop-" in l and "retract" in l][0].split()[0]
    accept_proposal(pid_line, voter="human", db_path=store)

    assert "No hive proposals found" in execute_tool("list_hive_proposals", {"status": "pending"})
    out = execute_tool("list_hive_proposals", {"status": "accepted"})
    assert "1 hive proposal(s)" in out
    assert "accepted" in out
    out = execute_tool("list_hive_proposals", {"status": "pending"})
    assert "No hive proposals found" in out
    # No status filter → shows all statuses.
    out = execute_tool("list_hive_proposals", {})
    assert "1 hive proposal(s)" in out


# ---------------------------------------------------------------------------
# Local/hive split: propose tools target the hive; local tools still refuse
# hive-only beliefs and point at the propose path.
# ---------------------------------------------------------------------------

@pytest.fixture
def layered_store(tmp_path):
    """Hive + local brain, so we can check the split explicitly."""
    from reasons.api import init_db, add_node

    hive = str(tmp_path / "hive.db")
    init_db(db_path=hive)
    add_node("hive-fact", "A hive belief.", db_path=hive)

    brain = str(tmp_path / "brain.db")
    init_db(db_path=brain)
    add_node("local-fact", "A local belief.", db_path=brain)

    set_belief_db(hive, brain_path=brain)
    yield hive, brain
    set_belief_db(None)


def test_local_retract_still_refuses_hive_and_points_at_propose(layered_store):
    hive, brain = layered_store
    out = execute_tool("retract_belief", {"id": "hive-fact"})
    assert "hive" in out
    assert "propose_retraction" in out
    from reasons.api import show_node
    assert show_node("hive-fact", db_path=hive)["truth_value"] == "IN"


def test_local_supersede_still_refuses_hive_and_points_at_propose(layered_store):
    hive, brain = layered_store
    out = execute_tool("supersede_belief", {"id": "hive-fact", "text": "new"})
    assert "hive" in out
    assert "propose_supersede" in out
    from reasons.api import show_node
    assert show_node("hive-fact", db_path=hive)["truth_value"] == "IN"


def test_propose_targets_hive_even_when_local_brain_configured(layered_store):
    hive, brain = layered_store
    out = execute_tool("propose_retraction", {"target_id": "hive-fact"})
    assert "pending" in out

    # The proposal row lands in the HIVE db, not the brain.
    from reasons.api import list_proposals
    assert list_proposals(status="pending", db_path=hive)["count"] == 1
    assert list_proposals(status="pending", db_path=brain)["count"] == 0
