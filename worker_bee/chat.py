"""Interactive chat mode: REPL around the prompt command.

Each user prompt gets a fresh context window. Knowledge accumulates
in the brain db (beliefs) and notes file between rounds — the bee
starts fresh every time but can search_beliefs and list_notes to
recall what it learned before.
"""

from __future__ import annotations

import sys

from prompt_toolkit import prompt as pt_prompt
from pathlib import Path

from worker_bee.editor import run_edit_loop, PROMPT_SYSTEM_PREFIX


CHAT_SYSTEM_PREFIX = """\
You are a worker bee — a focused assistant with tools to read, edit, and
write files, search the codebase, run commands, and query a belief database.

You get a FRESH CONTEXT for every prompt. Your conversation history is NOT
preserved between prompts. To remember things across prompts:

- Use list_notes at the start of every task to recall what previous rounds
  recorded. Do NOT redo work that is already noted.
- Use write_note to record findings as you work. Notes are PERSISTENT —
  they survive across prompts and sessions.
- Use add_belief to record durable conclusions, discoveries, or claims.
  Notes are scratch; beliefs are knowledge.
- Use search_beliefs to recall prior beliefs.

Within a single prompt you have limited context. Take notes as you work:

- After reading each file, immediately use write_note to record what you
  learned. Do NOT read another file until you have noted your findings.
- Use list_memory and retrieve_memory to recall earlier findings that may
  have scrolled out of context within this prompt.

Work through your task step by step. Start by calling list_notes, then
use write_note to record your plan.

If you are unsure about something, use ask_user_question to ask the user
for clarification before proceeding.

When you are done, use add_belief to record any lasting conclusions.

## Task

"""


def run_chat(
    *,
    model: str = "ollama:qwen3.8:27b",
    max_turns: int = 20,
    dry_run: bool = False,
    verbose: bool = False,
    confirm: bool = False,
    num_ctx: int = 65536,
    db_path: str | None = None,
    brain_path: str | None = None,
    truncate_chars: int | None = None,
    ctx_limit_pct: float = 0.80,
    allow_questions: bool = True,
) -> None:
    """Run an interactive chat loop."""
    print("worker-bee chat (type 'exit' or Ctrl-D to quit)", file=sys.stderr)
    if db_path:
        print(f"  hive: {db_path}", file=sys.stderr)
    if brain_path:
        print(f"  brain: {brain_path}", file=sys.stderr)
    print(f"  notes: .worker-bee/notes.jsonl", file=sys.stderr)
    print(f"  model: {model}", file=sys.stderr)
    print(f"  context: {num_ctx} tokens", file=sys.stderr)
    print(file=sys.stderr)

    round_num = 0
    while True:
        try:
            task = pt_prompt("bee> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            break

        if not task:
            continue
        if task.lower() in ("exit", "quit", "q"):
            break

        round_num += 1
        print(f"\n--- Round {round_num} ---", file=sys.stderr)

        try:
            run_edit_loop(
                task,
                model=model,
                max_turns=max_turns,
                dry_run=dry_run,
                verbose=verbose,
                confirm=confirm,
                num_ctx=num_ctx,
                db_path=db_path,
                brain_path=brain_path,
                system_prefix=CHAT_SYSTEM_PREFIX,
                truncate_chars=truncate_chars,
                ctx_limit_pct=ctx_limit_pct,
                chat_mode=True,
                allow_questions=allow_questions,
            )
        except KeyboardInterrupt:
            print("\n  Interrupted.", file=sys.stderr)

        print(file=sys.stderr)
