"""Trace a belief back to source code."""

from __future__ import annotations

import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

from worker_bee.tokens import count_tokens

TOKEN_BUDGET = 56_000


@dataclass
class TraceResult:
    belief_id: str
    belief_text: str
    source: str
    summary_text: str
    code_refs: list[str]
    code_found: list[str]
    code_missing: list[str]
    code_sections: str
    token_count: int = 0


def trace_belief(
    db_path: str | Path,
    belief_id: str,
    *,
    project_dir: str | Path | None = None,
) -> TraceResult:
    """Look up a belief, read its source summary, and load referenced code files."""
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    row = conn.execute(
        "SELECT id, text, source, truth_value FROM nodes WHERE id = ?",
        (belief_id,),
    ).fetchone()
    conn.close()

    if not row:
        print(f"Belief not found: {belief_id}", file=sys.stderr)
        return TraceResult(
            belief_id=belief_id, belief_text="", source="", summary_text="",
            code_refs=[], code_found=[], code_missing=[], code_sections="",
        )

    print(f"Belief: {row['id']}", file=sys.stderr)
    print(f"  Claim: {row['text']}", file=sys.stderr)
    print(f"  Source: {row['source']}", file=sys.stderr)
    print(f"  Status: {row['truth_value']}", file=sys.stderr)

    summary_path = Path(row["source"])
    if not summary_path.is_absolute():
        summary_path = db_path.parent / summary_path

    if not summary_path.exists():
        print(f"  Source file not found: {summary_path}", file=sys.stderr)
        return TraceResult(
            belief_id=row["id"], belief_text=row["text"], source=row["source"],
            summary_text="", code_refs=[], code_found=[], code_missing=[],
            code_sections="",
        )

    summary_text = summary_path.read_text(errors="replace")
    code_refs = _extract_code_refs(summary_text)
    print(f"  Code references found: {len(code_refs)}", file=sys.stderr)
    for ref in code_refs:
        print(f"    - {ref}", file=sys.stderr)

    if project_dir is None:
        project_dir = _guess_project_dir(db_path)
        if project_dir:
            print(f"  Project dir (guessed): {project_dir}", file=sys.stderr)
    else:
        project_dir = Path(project_dir)

    code_sections, found, missing = _load_code_refs(code_refs, project_dir, summary_path.parent)

    if found:
        print(f"  Code files loaded: {len(found)}", file=sys.stderr)
    if missing:
        print(f"  Code files missing: {len(missing)}", file=sys.stderr)
        for m in missing:
            print(f"    - {m}", file=sys.stderr)

    token_count = count_tokens(summary_text) + count_tokens(code_sections)
    print(f"  Content: ~{token_count:,} tokens", file=sys.stderr)

    return TraceResult(
        belief_id=row["id"],
        belief_text=row["text"],
        source=row["source"],
        summary_text=summary_text,
        code_refs=code_refs,
        code_found=found,
        code_missing=missing,
        code_sections=code_sections,
        token_count=token_count,
    )


def _extract_code_refs(text: str) -> list[str]:
    """Extract file path references from a summary document."""
    refs: list[str] = []
    seen: set[str] = set()

    def _add(path: str) -> None:
        path = path.strip().rstrip(",:;)")
        if path and path not in seen:
            seen.add(path)
            refs.append(path)

    for m in re.finditer(r'\[file\]\s*`?([^`\n]+)`?', text):
        _add(m.group(1))

    for m in re.finditer(r'[Ff]rom\s+`([^`]+\.\w+)`', text):
        _add(m.group(1))

    extensions = r'\.(?:rs|py|toml|js|ts|jsx|tsx|json|yaml|yml|md|go|c|h|cpp|hpp|java|rb|sh)'
    for m in re.finditer(rf'`([^`\s]+{extensions})`', text):
        candidate = m.group(1)
        if '(' not in candidate and len(candidate) < 200:
            _add(candidate)

    for m in re.finditer(rf'(?:^|\s)((?:\.\./)?(?:[\w.-]+/)*[\w.-]+{extensions})', text, re.MULTILINE):
        candidate = m.group(1)
        if len(candidate) < 200 and '(' not in candidate:
            _add(candidate)

    filtered: list[str] = []
    for ref in refs:
        if "/" not in ref and any(r.endswith("/" + ref) for r in refs):
            continue
        filtered.append(ref)

    return filtered


def _guess_project_dir(db_path: Path) -> Path | None:
    """Guess the project directory from the expert DB path.

    Convention: if db is in /path/to/foo-expert/, the project is /path/to/foo/.
    """
    name = db_path.parent.name
    if name.endswith("-expert"):
        project_name = name.removesuffix("-expert")
        candidate = db_path.parent.parent / project_name
        if candidate.is_dir():
            return candidate
    return None


def _load_code_refs(
    refs: list[str],
    project_dir: Path | None,
    summary_dir: Path,
) -> tuple[str, list[str], list[str]]:
    """Load referenced code files, respecting token budget."""
    found: list[str] = []
    missing: list[str] = []
    sections: list[str] = []
    tokens_used = 0
    budget = TOKEN_BUDGET - 4000

    for ref in refs:
        path = _resolve_ref(ref, project_dir, summary_dir)
        if path is None or not path.exists():
            missing.append(ref)
            continue

        try:
            content = path.read_text(errors="replace")
        except Exception:
            missing.append(ref)
            continue

        file_tokens = count_tokens(content)
        if tokens_used + file_tokens > budget:
            if file_tokens > 10_000:
                content = _truncate_to_tokens(content, max(2000, budget - tokens_used))
                file_tokens = count_tokens(content)
                if tokens_used + file_tokens > budget:
                    missing.append(f"{ref} (too large)")
                    continue
            else:
                missing.append(f"{ref} (budget exceeded)")
                continue

        sections.append(f"### {ref}\n```\n{content}\n```")
        tokens_used += file_tokens
        found.append(ref)

    return "\n\n".join(sections), found, missing


def _resolve_ref(ref: str, project_dir: Path | None, summary_dir: Path) -> Path | None:
    """Resolve a code reference to an absolute path."""
    p = Path(ref)

    if p.is_absolute() and p.exists():
        return p

    if project_dir:
        candidate = project_dir / ref
        if candidate.exists():
            return candidate
        candidate = project_dir / ref.lstrip("./")
        if candidate.exists():
            return candidate

    candidate = summary_dir / ref
    if candidate.exists():
        return candidate

    return None


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens."""
    lines = text.splitlines(keepends=True)
    result: list[str] = []
    tokens = 0
    for line in lines:
        line_tokens = count_tokens(line)
        if tokens + line_tokens > max_tokens:
            break
        result.append(line)
        tokens += line_tokens
    result.append(f"\n... (truncated, {len(lines) - len(result)} lines omitted)")
    return "".join(result)
