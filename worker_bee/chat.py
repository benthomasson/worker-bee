"""Interactive chat mode: REPL around the prompt command.

Each user prompt gets a fresh context window. Knowledge accumulates
in the brain db between rounds — the bee starts fresh every time
but can search_beliefs to recall what it learned before.
"""

from __future__ import annotations

import sys
from pathlib import Path

from worker_bee.editor import run_edit_loop, PROMPT_SYSTEM_PREFIX


CHAT_SYSTEM_PREFIX = """\
You are a worker bee — a focused assistant with tools to read, edit, and
write files, search the codebase, run commands, and query a belief database.

You get a FRESH CONTEXT for every prompt. Your conversation history is NOT
preserved between prompts. To remember things across prompts:

- Use add_belief to record conclusions, discoveries, or claims. These
  persist in your belief database and you can search_beliefs next time.
- Use search_beliefs at the start of a task to recall prior findings.

Within a single prompt you have limited context. Take notes as you work:

- After reading each file, immediately use write_note to record what you
  learned. Do NOT read another file until you have noted your findings.
- Use list_memory and retrieve_memory to recall earlier findings that may
  have scrolled out of context.

Work through your task step by step. Start by using write_note to record
your plan.

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
) -> None:
    """Run an interactive chat loop."""
    print("worker-bee chat (type 'exit' or Ctrl-D to quit)", file=sys.stderr)
    if db_path:
        print(f"  hive: {db_path}", file=sys.stderr)
    if brain_path:
        print(f"  brain: {brain_path}", file=sys.stderr)
    print(f"  model: {model}", file=sys.stderr)
    print(f"  context: {num_ctx} tokens", file=sys.stderr)
    print(file=sys.stderr)

    round_num = 0
    while True:
        try:
            task = input("bee> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            break

        if not task:
            continue
        if task.lower() in ("exit", "quit", "q"):
            break

        round_num += 1
        print(f"\n--- Round {round_num} ---", file=sys.stderr)

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
        )

        print(file=sys.stderr)
