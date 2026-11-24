"""Verify beliefs against source code via agentic tool loop."""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from worker_bee.tracer import trace_belief


VERIFY_SYSTEM_PROMPT = """\
You are verifying a belief from a Truth Maintenance System (TMS) against
actual source code. The belief was originally derived from a summary
document, not from the code itself. Your job is to check whether the
source code actually supports the claim.

You have tools to explore the codebase: read_file, grep, and glob.
Use them to find the relevant source code and build your evidence.

## Belief
ID: {belief_id}
Claim: {belief_text}

## Source Summary (where the belief was derived)
{summary}

{initial_hints}

## Task
1. Use your tools to find and read the source code relevant to this belief.
   Start with the files mentioned above, then follow imports and references
   to build a complete picture.
2. Determine whether the belief claim is supported by the actual source code.
3. When you have enough evidence, output your verdict as one raw JSON object
   and nothing else (no Markdown fences or surrounding explanation):

{{
  "verified": true or false,
  "confidence": "high", "medium", or "low",
  "evidence": "the specific code or pattern that supports or contradicts the claim",
  "comment": "detailed reasoning chain: what you looked at, what you found, and why it confirms or refutes the claim"
}}

Rules:
- "verified" means the source code confirms the claim is still true today,
  not just that it sounds plausible.
- Read the actual implementation files. Do not guess based on file names alone.
- Quote specific lines, function signatures, or patterns as evidence.
- If the belief claims something is missing (no validation, no locking, etc.),
  search thoroughly before concluding it really is missing.
- Your comment should trace your investigation: which files you read,
  what you found in each, and how the pieces connect to your verdict."""


READ_ONLY_TOOLS = [
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file. For large files, use offset and limit "
            "to read specific line ranges. Files over 500 lines are automatically "
            "truncated — use offset/limit to read further."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to read",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-based, default: 1)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (default: 500)",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "grep",
        "description": (
            "Search for a regex pattern across files in a directory. "
            "Returns matching lines with file paths and line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "The directory to search in (defaults to current directory)",
                    "default": ".",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "glob",
        "description": (
            "Find files matching a glob pattern (e.g. '**/*.py' for all Python files). "
            "Returns a list of matching file paths."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The glob pattern to match",
                },
                "path": {
                    "type": "string",
                    "description": "The directory to search in (defaults to current directory)",
                    "default": ".",
                },
            },
            "required": ["pattern"],
        },
    },
]


@dataclass
class VerifyResult:
    belief_id: str
    belief_text: str
    source: str
    code_refs: list[str]
    code_found: list[str]
    code_missing: list[str]
    verified: bool | None = None
    confidence: str = ""
    evidence: str = ""
    comment: str = ""


def verify_unverified(
    db_path: str | Path,
    *,
    project_dir: str | Path | None = None,
    model: str = "ollama:qwen3.8:27b",
    limit: int | None = None,
    dry_run: bool = False,
    verbose: bool = False,
    max_turns: int = 20,
    num_ctx: int | None = None,
    retract_inaccurate: bool = False,
) -> list[VerifyResult]:
    """Verify unverified gated beliefs against source code.

    Verification is evidence-only by default. Set ``retract_inaccurate`` to
    enforce a negative verdict by retracting the belief.
    """
    db_path = Path(db_path)
    belief_ids = _find_unverified_gated(str(db_path))

    if limit is not None:
        belief_ids = belief_ids[:limit]

    if not belief_ids:
        print("No unverified gated beliefs found.", file=sys.stderr)
        return []

    print(f"Found {len(belief_ids)} unverified gated belief(s) to verify.", file=sys.stderr)
    for bid in belief_ids:
        print(f"  - {bid}", file=sys.stderr)

    results: list[VerifyResult] = []
    for i, bid in enumerate(belief_ids, 1):
        print(f"\n[{i}/{len(belief_ids)}] Verifying {bid}...", file=sys.stderr)
        try:
            result = verify_belief(
                db_path, bid,
                project_dir=project_dir,
                model=model,
                dry_run=dry_run,
                verbose=verbose,
                max_turns=max_turns,
                num_ctx=num_ctx,
                retract_inaccurate=retract_inaccurate,
            )
            results.append(result)
        except Exception as e:
            print(f"  WARN: {bid} failed: {e}", file=sys.stderr)

    if not dry_run and results:
        verified = sum(1 for r in results if r.verified)
        not_verified = sum(1 for r in results if r.verified is False)
        print(f"\n  Verified {verified}, not verified {not_verified} "
              f"out of {len(results)} checked.", file=sys.stderr)

    return results


def _find_unverified_gated(db_path: str) -> list[str]:
    """Find gated beliefs that haven't been verified yet."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT j.node_id, j.antecedents_json
        FROM justifications j
        JOIN nodes n ON n.id = j.node_id
        WHERE n.truth_value = 'IN'
          AND (n.verified_at IS NULL OR n.verified_at = '')
    """).fetchall()

    gated: list[str] = []
    seen: set[str] = set()
    for row in rows:
        antecedents = json.loads(row["antecedents_json"])
        if not antecedents:
            continue

        placeholders = ",".join("?" for _ in antecedents)
        out_count = conn.execute(
            f"SELECT COUNT(*) as c FROM nodes WHERE id IN ({placeholders}) AND truth_value = 'OUT'",
            antecedents,
        ).fetchone()["c"]

        if out_count > 0 and row["node_id"] not in seen:
            seen.add(row["node_id"])
            gated.append(row["node_id"])

    conn.close()
    return gated


def verify_belief(
    db_path: str | Path,
    belief_id: str,
    *,
    project_dir: str | Path | None = None,
    model: str = "ollama:qwen3.8:27b",
    dry_run: bool = False,
    verbose: bool = False,
    max_turns: int = 20,
    num_ctx: int | None = None,
    retract_inaccurate: bool = False,
) -> VerifyResult:
    """Verify a belief using an agentic tool loop.

    A negative verdict remains evidence-only unless ``retract_inaccurate`` is
    explicitly enabled.
    """
    from worker_bee.editor import run_edit_loop

    trace = trace_belief(db_path, belief_id, project_dir=project_dir)

    hints = ""
    if trace.code_refs:
        hints = "## Hints — files referenced in the source summary\n"
        hints += "These may be relevant starting points for your investigation:\n"
        for ref in trace.code_refs:
            found = "(found)" if ref in trace.code_found else "(not resolved)"
            hints += f"- {ref} {found}\n"

    summary = trace.summary_text or "(No source summary available. Use your tools to explore the codebase and find the relevant code.)"

    task = VERIFY_SYSTEM_PROMPT.format(
        belief_id=trace.belief_id,
        belief_text=trace.belief_text,
        summary=summary,
        initial_hints=hints,
    )

    if dry_run:
        print(f"\n{'='*60}")
        print(f"Verify: {trace.belief_id}")
        print(f"{'='*60}")
        print(task)
        return VerifyResult(
            belief_id=trace.belief_id, belief_text=trace.belief_text,
            source=trace.source, code_refs=trace.code_refs,
            code_found=trace.code_found, code_missing=trace.code_missing,
        )

    project_dir = project_dir or _guess_project_dir(db_path)

    session = run_edit_loop(
        task,
        model=model,
        max_turns=max_turns,
        num_ctx=num_ctx,
        verbose=verbose,
        workspace_root=project_dir,
        system_prefix="",
        custom_tools=READ_ONLY_TOOLS,
    )

    result = _extract_verdict(session)

    # A missing or malformed model response is undecided.  Do not index into
    # the parser result or write verification state for an invalid response.
    verify = VerifyResult(
        belief_id=trace.belief_id,
        belief_text=trace.belief_text,
        source=trace.source,
        code_refs=trace.code_refs,
        code_found=trace.code_found,
        code_missing=trace.code_missing,
        verified=result.get("verified"),
        confidence=result.get("confidence", ""),
        evidence=result.get("evidence", ""),
        comment=result.get("comment", ""),
    )

    status = (
        "VERIFIED" if verify.verified is True
        else "NOT VERIFIED" if verify.verified is False
        else "UNDECIDED"
    )
    print(f"\n  Result: {status}", file=sys.stderr)
    print(f"  Evidence: {verify.evidence}", file=sys.stderr)
    print(f"  Comment: {verify.comment}", file=sys.stderr)

    if verify.verified is not None:
        # Enforce an explicitly requested negative verdict before recording the
        # verification as complete.  If retraction fails, _update_db is not
        # reached and the database cannot claim that the operation completed.
        if verify.verified is False and retract_inaccurate:
            from reasons.api import retract_node
            retract_node(
                verify.belief_id,
                reason=f"worker-bee verification: {verify.comment}",
                db_path=str(db_path),
            )
        _update_db(str(db_path), verify)

    return verify


def _guess_project_dir(db_path: str | Path) -> str | None:
    """Guess the project directory from the db path."""
    db_path = Path(db_path).resolve()
    for parent in db_path.parents:
        if (parent / ".git").exists():
            return str(parent)
    return None


def _extract_verdict(session) -> dict:
    """Extract the JSON verdict from the session's final text output."""
    for step in reversed(session.steps):
        text = getattr(step, "text", None)
        if not text:
            continue
        result = _parse_verify_response(text)
        if result:
            return result
    return {}


def _update_db(db_path: str, result: VerifyResult) -> None:
    """Write verification result back to reasons.db via ftl-reasons API."""
    from reasons.api import _with_network

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with _with_network(db_path, write=True) as net:
        if result.belief_id not in net.nodes:
            return

        node = net.nodes[result.belief_id]
        node.verified_at = now
        node.updated_at = now
        if node.metadata is None:
            node.metadata = {}
        node.metadata["verify_result"] = {
            "verified": result.verified,
            "confidence": result.confidence,
            "evidence": result.evidence,
            "comment": result.comment,
            "code_files": result.code_found,
            "verifier": "worker-bee",
        }


def _parse_verify_response(response: str) -> dict:
    """Parse a verification response as one complete JSON object.

    The response boundary is deliberately strict: model prose, Markdown
    fences, and trailing text must not be able to smuggle an unrelated JSON
    object into the verification result.  ``json.loads`` permits surrounding
    JSON whitespace while rejecting any other surrounding content.
    """
    try:
        obj = json.loads(response)
    except (TypeError, json.JSONDecodeError):
        return {}

    # ``bool`` must be checked explicitly because it is a subclass of ``int``
    # in Python; only JSON true/false are valid verdicts.  Validate every
    # persisted field here so malformed model output cannot reach VerifyResult.
    if not isinstance(obj, dict):
        return {}
    if not isinstance(obj.get("verified"), bool):
        return {}
    confidence = obj.get("confidence")
    if not isinstance(confidence, str) or confidence not in {"high", "medium", "low"}:
        return {}
    if not isinstance(obj.get("evidence"), str):
        return {}
    if not isinstance(obj.get("comment"), str):
        return {}
    return obj
