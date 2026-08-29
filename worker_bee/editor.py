"""Code-editing loop: propose and apply fixes via multi-turn tool use."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from worker_bee.dispatcher import dispatch_chat
from worker_bee.tools import TOOLS, execute_tool
from worker_bee.llm import TextBlock, ToolUseBlock

MAX_TURNS = 20

EDITOR_SYSTEM_PREFIX = """\
You are a code editor working on a software project. You have tools to read,
edit, and write files, search the codebase, and run commands.

Work through your task step by step:
1. Read the relevant files to understand the current code.
2. Plan your changes.
3. Make the edits using edit_file (preferred) or write_file.
4. Run tests or build commands to verify your changes work.
5. When done, summarize what you changed and why.

Be precise with edits. Always read a file before editing it. Prefer small,
targeted changes over rewriting entire files.

## Task

"""


@dataclass
class EditStep:
    turn: int
    role: str  # "assistant" or "tool"
    tool_name: str | None = None
    tool_input: dict | None = None
    tool_result: str | None = None
    text: str | None = None


@dataclass
class EditSession:
    task: str
    model: str
    steps: list[EditStep] = field(default_factory=list)
    turns_used: int = 0
    completed: bool = False
    log_path: str | None = None


LOG_DIR = Path(".worker-bee/logs")


def _estimate_tokens(text) -> int:
    """Rough token estimate: serialize to string and divide by 4."""
    if isinstance(text, str):
        return len(text) // 4
    return len(json.dumps(text, default=str)) // 4


def run_edit_loop(
    task: str,
    *,
    model: str = "ollama:qwen3.8:27b",
    max_turns: int = MAX_TURNS,
    dry_run: bool = False,
    verbose: bool = False,
    confirm: bool = False,
    log_dir: str | Path | None = None,
    num_ctx: int | None = None,
) -> EditSession:
    """Run a multi-turn code-editing conversation with tool use.

    The task is placed in the system message so Ollama preserves it
    when evicting old messages from the context window.
    """
    session = EditSession(task=task, model=model)
    system = EDITOR_SYSTEM_PREFIX + task
    messages: list[dict] = [{"role": "user", "content": "Begin."}]

    ctx_limit = num_ctx or 0
    ctx_warn_threshold = int(ctx_limit * 0.80) if ctx_limit else 0

    log = _init_log(log_dir or LOG_DIR, task, model, dry_run)
    session.log_path = str(log["path"])
    print(f"Log: {session.log_path}", file=sys.stderr)

    system_tokens = _estimate_tokens(system)
    messages_tokens = _estimate_tokens(messages)

    print(f"Task: {task[:200]}", file=sys.stderr)
    print(f"Model: {model}", file=sys.stderr)
    print(f"Max turns: {max_turns}", file=sys.stderr)
    if ctx_limit:
        print(f"Context window: {ctx_limit} tokens", file=sys.stderr)
    if verbose:
        print(f"System prompt: ~{system_tokens} tokens", file=sys.stderr)
        print(f"Messages: ~{messages_tokens} tokens", file=sys.stderr)
        print(f"Total: ~{system_tokens + messages_tokens} tokens", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    for turn in range(1, max_turns + 1):
        session.turns_used = turn
        print(f"\n--- Turn {turn}/{max_turns} ---", file=sys.stderr)

        messages_tokens = _estimate_tokens(messages)
        total_est = system_tokens + messages_tokens
        if verbose:
            print(f"  Tokens — system: ~{system_tokens}  messages: ~{messages_tokens}  total: ~{total_est}", file=sys.stderr)

        if ctx_warn_threshold:
            if total_est > ctx_warn_threshold:
                print(f"  Context {total_est}/{ctx_limit} tokens — stopping to avoid overflow.", file=sys.stderr)
                _log_event(log, "context_limit", turn=turn, estimated_tokens=total_est, limit=ctx_limit)
                break
            elif not verbose:
                print(f"  Context: ~{total_est} tokens ({total_est * 100 // ctx_limit}%)", file=sys.stderr)

        try:
            response = dispatch_chat(
                messages,
                system=system,
                model=model,
                tools=TOOLS,
                num_ctx=num_ctx,
            )
        except RuntimeError as e:
            err = str(e)
            print(f"\n  Error: {err}", file=sys.stderr)
            _log_event(log, "error", turn=turn, error=err)
            if "no user query" in err.lower() or "context" in err.lower():
                print("  Context window likely exhausted.", file=sys.stderr)
            break

        assistant_content = []
        for block in response.content:
            if isinstance(block, TextBlock) and block.text.strip():
                print(f"  {block.text[:200]}", file=sys.stderr)
                if verbose:
                    print(f"\n## Assistant (turn {turn})\n{block.text}")
                _log_event(log, "text", turn=turn, text=block.text)
                session.steps.append(EditStep(
                    turn=turn, role="assistant", text=block.text,
                ))
                assistant_content.append({
                    "type": "text", "text": block.text,
                })
            elif isinstance(block, ToolUseBlock):
                _print_tool_call(block)
                if verbose:
                    _print_tool_call_verbose(block, turn)
                _log_event(log, "tool_call", turn=turn,
                           tool=block.name, input=block.input)
                session.steps.append(EditStep(
                    turn=turn, role="assistant",
                    tool_name=block.name, tool_input=block.input,
                ))
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason != "tool_use":
            print(f"\n  Model finished (stop_reason: {response.stop_reason})", file=sys.stderr)
            session.completed = True
            break

        tool_results = []
        abort = False
        for block in response.content:
            if not isinstance(block, ToolUseBlock):
                continue

            if dry_run:
                result_text = "(dry-run: tool not executed)"
            elif confirm:
                if not verbose:
                    _print_tool_call_verbose(block, turn)
                choice = _prompt_confirm(block.name)
                if choice == "abort":
                    result_text = "(aborted by user)"
                    abort = True
                elif choice == "skip":
                    result_text = "(skipped by user)"
                else:
                    result_text = execute_tool(block.name, block.input)
            else:
                result_text = execute_tool(block.name, block.input)

            if verbose:
                _print_tool_result_verbose(block.name, result_text, turn)

            _log_event(log, "tool_result", turn=turn,
                       tool=block.name, result=result_text)
            session.steps.append(EditStep(
                turn=turn, role="tool",
                tool_name=block.name, tool_result=result_text,
            ))
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_text,
            })

        messages.append({"role": "user", "content": tool_results})

        if abort:
            print(f"\n  Session aborted by user.", file=sys.stderr)
            break

    if not session.completed:
        print(f"\n  Reached max turns ({max_turns})", file=sys.stderr)

    _log_event(log, "session_end",
               turns=session.turns_used, completed=session.completed)
    _print_summary(session)
    return session


def _print_tool_call(block: ToolUseBlock) -> None:
    """Print a compact tool call summary to stderr."""
    if block.name == "edit_file":
        path = block.input.get("path", "?")
        old = block.input.get("old_string", "")
        preview = old[:80].replace("\n", "\\n")
        print(f"  -> edit_file({path}, {preview!r}...)", file=sys.stderr)
    elif block.name == "write_file":
        print(f"  -> write_file({block.input.get('path', '?')})", file=sys.stderr)
    elif block.name == "read_file":
        print(f"  -> read_file({block.input.get('path', '?')})", file=sys.stderr)
    elif block.name == "grep":
        print(f"  -> grep({block.input.get('pattern', '?')})", file=sys.stderr)
    elif block.name == "glob":
        print(f"  -> glob({block.input.get('pattern', '?')})", file=sys.stderr)
    elif block.name == "run_command":
        print(f"  -> run_command({block.input.get('command', '?')[:100]})", file=sys.stderr)
    else:
        print(f"  -> {block.name}({block.input})", file=sys.stderr)


def _prompt_confirm(tool_name: str) -> str:
    """Prompt the user to confirm a tool call. Returns 'yes', 'skip', or 'abort'."""
    try:
        answer = input(f"  Execute {tool_name}? [Y/n/abort] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        return "abort"
    if answer in ("", "y", "yes"):
        return "yes"
    if answer in ("a", "abort", "q", "quit"):
        return "abort"
    return "skip"


def _print_tool_call_verbose(block: ToolUseBlock, turn: int) -> None:
    """Print full tool call details to stdout."""
    print(f"\n## Tool call: {block.name} (turn {turn})")
    print(f"```json")
    print(json.dumps(block.input, indent=2))
    print(f"```")


def _print_tool_result_verbose(tool_name: str, result: str, turn: int) -> None:
    """Print tool result to stdout."""
    print(f"\n## Tool result: {tool_name} (turn {turn})")
    if len(result) > 2000:
        print(f"```")
        print(result[:2000])
        print(f"... ({len(result) - 2000} chars truncated)")
        print(f"```")
    else:
        print(f"```")
        print(result)
        print(f"```")


def _print_summary(session: EditSession) -> None:
    """Print a summary of the edit session."""
    edits = [s for s in session.steps if s.role == "assistant" and s.tool_name in ("edit_file", "write_file")]
    reads = [s for s in session.steps if s.role == "assistant" and s.tool_name == "read_file"]
    commands = [s for s in session.steps if s.role == "assistant" and s.tool_name == "run_command"]

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Session summary: {session.turns_used} turns, "
          f"{len(reads)} reads, {len(edits)} edits, {len(commands)} commands",
          file=sys.stderr)
    if edits:
        files = sorted(set(s.tool_input.get("path", "?") for s in edits))
        print(f"  Files modified: {', '.join(files)}", file=sys.stderr)
    if session.log_path:
        print(f"  Log: {session.log_path}", file=sys.stderr)


def _init_log(log_dir: str | Path, task: str, model: str, dry_run: bool) -> dict:
    """Create a session log file and return the log handle."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    filename = now.strftime("%Y%m%d-%H%M%S") + ".jsonl"
    log_path = log_dir / filename

    log = {"path": log_path, "file": open(log_path, "a")}
    _log_event(log, "session_start",
               task=task, model=model, dry_run=dry_run)
    return log


def _log_event(log: dict, event: str, **data) -> None:
    """Append a JSON line to the session log."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        **data,
    }
    f = log.get("file")
    if f and not f.closed:
        f.write(json.dumps(entry, default=str) + "\n")
        f.flush()
