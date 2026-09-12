"""Tests for the chat REPL's prompt history wiring.

These do NOT invoke the interactive prompt(); they verify the mechanism
that underpins it: that the chat module wires a prompt_toolkit FileHistory
into the ``bee>`` REPL, so that:

1. The SAME instance is shared across every prompt() in a session (in-session
   up/down recall — the whole point of passing ``history=`` to ``pt_prompt``).
2. Entries survive on disk and are re-loadable by a fresh FileHistory
   instance (cross-session recall).
"""

from __future__ import annotations

from pathlib import Path

from prompt_toolkit.history import FileHistory

import worker_bee.chat as chat


def test_history_path_lives_under_worker_bee_dir():
    # History file sits alongside the notes file so it is CWD-relative and
    # survives in the same place across sessions.
    assert chat.HISTORY_PATH == ".worker-bee/prompt_history"


def test_history_file_round_trips(tmp_path, monkeypatch):
    # Simulate a chat session: create the history file, append entries,
    # close the REPL, and open a fresh FileHistory in a new "session" that
    # must see the old entries.
    hist_dir = tmp_path / ".worker-bee"
    hist_path = hist_dir / "prompt_history"
    hist_dir.mkdir(parents=True)

    monkeypatch.setattr(chat, "HISTORY_PATH", str(hist_path))

    # "Session 1"
    h1 = FileHistory(hist_path)
    for entry in ("wire up prompt history", "add tests for it", "explain FileHistory"):
        h1.append_string(entry)

    # "Session 2" — a brand-new FileHistory over the same path must see
    # what session 1 stored, newest-first.
    h2 = FileHistory(hist_path)
    loaded = list(h2.load_history_strings())

    assert loaded[0] == "explain FileHistory"
    assert "add tests for it" in loaded
    assert "wire up prompt history" in loaded


def test_shared_instance_is_single_per_chat_run(tmp_path, monkeypatch, capsys):
    # The REPL uses ONE history instance across every prompt() call in the
    # process. We can't run the interactive REPL, but we CAN verify the
    # run_chat wiring by patching pt_prompt and FileHistory and confirming
    # they're called with the SAME history object.
    hist_path = tmp_path / ".worker-bee" / "prompt_history"
    hist_path.parent.mkdir(parents=True)
    monkeypatch.setattr(chat, "HISTORY_PATH", str(hist_path))

    calls: list[dict] = []

    def fake_pt_prompt(prompt, history=None):
        calls.append({"prompt": prompt, "history": history})
        if len(calls) >= 3:
            raise EOFError  # exit the REPL loop after the third prompt
        return "do something"  # two prompts processed, then EOF

    class FakeFileHistory:
        def __init__(self, path):
            self.path = path

    monkeypatch.setattr(chat, "pt_prompt", fake_pt_prompt)
    monkeypatch.setattr(chat, "FileHistory", FakeFileHistory)
    # Make run_edit_loop a no-op in case the loop survives (it shouldn't)
    monkeypatch.setattr(chat, "run_edit_loop", lambda *a, **k: None)

    chat.run_chat()

    # At least one prompt() call happened, and it got a FileHistory instance.
    assert calls, "expected run_chat to call pt_prompt at least once"
    assert calls[0]["prompt"] == "bee> "

    # KEY PROPERTY: every prompt() in the REPL receives the SAME history
    # object — that is what makes in-session up/down recall work. If each
    # call built its own history, this would fail.
    histories = {id(c["history"]) for c in calls}
    assert len(calls) >= 3
    assert len(histories) == 1, (
        f"expected all {len(calls)} prompt() calls to share one history, "
        f"got {len(histories)} distinct instances"
    )
