"""Assemble self-contained prompts for worker bees."""

from __future__ import annotations

from pathlib import Path

from worker_bee.tokens import count_tokens


DEFAULT_TOKEN_BUDGET = 56000


def build_prompt(issue: dict, *, token_budget: int = DEFAULT_TOKEN_BUDGET) -> str:
    """Build a self-contained prompt for a single issue.

    Returns a prompt string ready to send to a model.
    """
    parts = [
        _system_section(issue),
        _belief_section(issue),
        _source_section(issue, token_budget),
        _task_section(issue),
    ]
    return "\n\n".join(p for p in parts if p)


def _system_section(issue: dict) -> str:
    return (
        "You are a focused code analyst. You will be given a belief about a codebase "
        "and the relevant source code. Your job is to analyze whether the belief is "
        "correct and, if it identifies a problem, propose a fix.\n\n"
        "Respond with:\n"
        "1. ANALYSIS: Is the belief accurate? Explain briefly.\n"
        "2. FIX: If a fix is needed, provide a minimal diff or corrected code.\n"
        "3. CONFIDENCE: high / medium / low"
    )


def _belief_section(issue: dict) -> str:
    lines = [f"## Belief: {issue['belief_id']}"]
    lines.append(f"Type: {issue['type']}")
    lines.append(f"Description: {issue['description']}")
    if issue.get("belief_text"):
        lines.append(f"\n{issue['belief_text']}")
    return "\n".join(lines)


def _source_section(issue: dict, token_budget: int) -> str:
    sources = issue.get("source_files", [])
    if not sources:
        return ""

    parts = []
    tokens_used = 0

    for src in sources:
        p = Path(src)
        if not p.exists():
            parts.append(f"## Source: {src}\n[file not found]")
            continue

        content = p.read_text(errors="replace")
        content_tokens = count_tokens(content)
        if tokens_used + content_tokens > token_budget:
            chars_remaining = (token_budget - tokens_used) * 3
            content = content[:chars_remaining] + "\n... [truncated]"
            tokens_used = token_budget
        else:
            tokens_used += content_tokens

        parts.append(f"## Source: {src}\n```\n{content}\n```")

        if tokens_used >= token_budget:
            break

    return "\n\n".join(parts)


def _task_section(issue: dict) -> str:
    tasks = {
        "gated": (
            "This belief is marked as IN (accepted) but depends on another belief "
            "that is OUT (retracted). Analyze the source code to determine:\n"
            "- Is this belief still valid on its own merits?\n"
            "- If not, what code change would make it valid, or should it be retracted?"
        ),
        "contradiction": (
            "Two contradictory beliefs are both marked as IN. Analyze the source "
            "code to determine which belief is correct and which should be retracted."
        ),
        "stale": (
            "This belief was derived from source code that has since changed. "
            "Re-examine the source and determine if the belief is still accurate."
        ),
        "unreviewed": (
            "This belief was automatically derived and has not been reviewed. "
            "Verify whether the belief accurately describes the source code."
        ),
    }
    task = tasks.get(issue["type"], "Analyze this belief against the source code.")
    return f"## Task\n{task}"
