"""Post-mortem summarization of worker-bee session logs."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from worker_bee.dispatcher import dispatch

SUMMARIZE_PROMPT = """\
You are summarizing a worker-bee session log. The log records what an AI
worker bee did during a task: which files it read, what edits it made,
what commands it ran, and what it concluded.

Write a concise summary entry in markdown. Include:
- A short title as a markdown heading
- What the task was
- What the bee did (files read, edited, commands run)
- Key findings or changes made
- Whether the task completed successfully
- Any open questions or follow-up work needed

Be concise. Focus on what happened and what matters for future reference.

## Session log

{log_text}"""


def summarize_log(
    log_path: str | Path,
    *,
    model: str = "ollama:qwen3.8:27b",
    entry_dir: str | Path | None = None,
    dry_run: bool = False,
) -> str | None:
    """Read a session JSONL log and produce a summary entry."""
    log_path = Path(log_path)
    if not log_path.exists():
        print(f"Log file not found: {log_path}", file=sys.stderr)
        return None

    events = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    if not events:
        print("Empty log file.", file=sys.stderr)
        return None

    log_text = _condense_log(events)
    prompt = SUMMARIZE_PROMPT.format(log_text=log_text)

    if dry_run:
        from worker_bee.tokens import count_tokens
        token_count = count_tokens(prompt)
        print(f"Prompt (~{token_count:,} tokens):")
        print(prompt)
        return None

    print(f"Summarizing {log_path.name} ({len(events)} events)...", file=sys.stderr)
    resp = dispatch(prompt, model=model)
    summary = resp.text

    if entry_dir:
        entry_path = _write_entry(summary, events, entry_dir)
        print(f"Entry written: {entry_path}", file=sys.stderr)

    return summary


def _condense_log(events: list[dict]) -> str:
    """Condense JSONL events into a readable summary for the LLM."""
    lines = []

    for ev in events:
        event = ev.get("event", "")

        if event == "session_start":
            lines.append(f"Task: {ev.get('task', '?')}")
            lines.append(f"Model: {ev.get('model', '?')}")
            lines.append("")

        elif event == "text":
            turn = ev.get("turn", "?")
            text = ev.get("text", "")
            lines.append(f"[Turn {turn}] Assistant: {text[:500]}")
            if len(text) > 500:
                lines.append(f"  ... ({len(text) - 500} chars truncated)")
            lines.append("")

        elif event == "tool_call":
            turn = ev.get("turn", "?")
            tool = ev.get("tool", "?")
            tool_input = ev.get("input", {})
            summary = _summarize_tool_input(tool, tool_input)
            lines.append(f"[Turn {turn}] Tool: {tool}({summary})")

        elif event == "tool_result":
            tool = ev.get("tool", "?")
            result = ev.get("result", "")
            if tool in ("list_memory", "retrieve_memory", "write_note"):
                continue
            if len(result) > 200:
                lines.append(f"  Result: {result[:200]}... ({len(result)} chars)")
            else:
                lines.append(f"  Result: {result}")
            lines.append("")

        elif event == "error":
            lines.append(f"[Turn {ev.get('turn', '?')}] ERROR: {ev.get('error', '?')}")
            lines.append("")

        elif event == "context_limit":
            lines.append(f"[Turn {ev.get('turn', '?')}] CONTEXT LIMIT: "
                         f"{ev.get('estimated_tokens', '?')}/{ev.get('limit', '?')} tokens")
            lines.append("")

        elif event == "session_end":
            completed = ev.get("completed", False)
            turns = ev.get("turns", "?")
            status = "completed" if completed else "did not complete"
            lines.append(f"Session {status} after {turns} turns.")

    return "\n".join(lines)


def _summarize_tool_input(tool: str, tool_input: dict) -> str:
    """Short summary of tool input for the condensed log."""
    if tool in ("read_file", "write_file", "edit_file"):
        return tool_input.get("path", "?")
    if tool == "grep":
        return f"pattern={tool_input.get('pattern', '?')}"
    if tool == "glob":
        return f"pattern={tool_input.get('pattern', '?')}"
    if tool == "run_command":
        return tool_input.get("command", "?")[:100]
    if tool == "show_belief":
        return tool_input.get("id", "?")
    if tool == "search_beliefs":
        return tool_input.get("query", "?")
    return str(tool_input)[:80]


def _write_entry(summary: str, events: list[dict], entry_dir: str | Path) -> Path:
    """Write the summary as a dated entry file."""
    now = datetime.now(timezone.utc)
    date_dir = Path(entry_dir) / now.strftime("%Y/%m/%d")
    date_dir.mkdir(parents=True, exist_ok=True)

    task = ""
    for ev in events:
        if ev.get("event") == "session_start":
            task = ev.get("task", "session-summary")
            break

    import re
    slug = task[:60].lower().strip()
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'[\s_]+', '-', slug)
    slug = re.sub(r'-+', '-', slug).strip('-') or "session-summary"

    entry_path = date_dir / f"{slug}.md"
    entry_path.write_text(summary)
    return entry_path
