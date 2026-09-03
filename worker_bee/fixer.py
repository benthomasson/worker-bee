"""Fix verified issues by chaining trace -> verify result -> edit loop."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from worker_bee.editor import run_edit_loop, EditSession
from worker_bee.tracer import trace_belief, TraceResult


FIX_TASK_TEMPLATE = """\
## Issue
Belief `{belief_id}` was verified against the source code and found to be: {status}.

**Claim:** {belief_text}

**Verification result:**
- Confidence: {confidence}
- Evidence: {evidence}
- Comment: {comment}

## Relevant files
{file_list}

## Source summary
The belief was derived from this summary document ({source}):

{summary_excerpt}

## Task
{task_instruction}

Work in the project at: {project_dir}"""


def fix_belief(
    db_path: str | Path,
    belief_id: str,
    *,
    project_dir: str | Path | None = None,
    model: str = "ollama:qwen3.8:27b",
    max_turns: int = 20,
    dry_run: bool = False,
    verbose: bool = False,
    confirm: bool = False,
    num_ctx: int | None = None,
    brain_path: str | None = None,
) -> EditSession:
    """Fix a verified issue by running a code-editing loop with full context."""
    db_path = Path(db_path)

    trace = trace_belief(db_path, belief_id, project_dir=project_dir)
    verify_result = _get_verify_result(str(db_path), belief_id)

    if project_dir is None:
        from worker_bee.tracer import _guess_project_dir
        project_dir = _guess_project_dir(db_path)

    task = _build_fix_task(trace, verify_result, project_dir)

    if dry_run or verbose:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Fix task for: {belief_id}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(task, file=sys.stderr)
        if dry_run:
            return EditSession(task=task, model=model)

    return run_edit_loop(
        task,
        model=model,
        max_turns=max_turns,
        dry_run=False,
        verbose=verbose,
        confirm=confirm,
        num_ctx=num_ctx,
        db_path=str(db_path),
        brain_path=brain_path,
    )


def _get_verify_result(db_path: str, belief_id: str) -> dict:
    """Look up the verify_result from node metadata."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT metadata_json FROM nodes WHERE id = ?", (belief_id,),
    ).fetchone()
    conn.close()

    if not row:
        return {}
    metadata = json.loads(row["metadata_json"] or "{}")
    return metadata.get("verify_result", {})


def _build_fix_task(
    trace: TraceResult,
    verify_result: dict,
    project_dir: Path | None,
) -> str:
    """Build the task prompt from trace and verify data."""
    verified = verify_result.get("verified")
    if verified is True:
        status = "VERIFIED (the code supports this claim)"
        task_instruction = (
            "This belief is verified as correct. If there is still an issue "
            "implied by the belief (e.g., a missing feature, a limitation), "
            "propose a code change to address it. Read the relevant files first, "
            "then make targeted edits. Run tests after editing."
        )
    elif verified is False:
        status = "NOT VERIFIED (the code does not support this claim)"
        task_instruction = (
            "This belief was found to be inaccurate based on the source code. "
            "Examine the relevant files to understand the actual behavior. "
            "If the code has a bug that the belief exposed, fix it. "
            "If the belief is simply wrong and the code is correct, no code "
            "change is needed — just explain what the code actually does."
        )
    else:
        status = "UNVERIFIED (no verification result available)"
        task_instruction = (
            "This belief has not been verified yet. Read the relevant files, "
            "determine whether the claim is accurate, and fix any issues you find. "
            "Run tests after editing."
        )

    if trace.code_found:
        file_list = "\n".join(f"- {f}" for f in trace.code_found)
    else:
        file_list = "(no code files found — explore the project to find relevant files)"

    summary_excerpt = trace.summary_text[:2000] if trace.summary_text else "(no summary available)"
    if len(trace.summary_text or "") > 2000:
        summary_excerpt += "\n... (truncated)"

    return FIX_TASK_TEMPLATE.format(
        belief_id=trace.belief_id,
        belief_text=trace.belief_text,
        status=status,
        confidence=verify_result.get("confidence", "unknown"),
        evidence=verify_result.get("evidence", "none"),
        comment=verify_result.get("comment", "none"),
        file_list=file_list,
        source=trace.source or "unknown",
        summary_excerpt=summary_excerpt,
        task_instruction=task_instruction,
        project_dir=project_dir or ".",
    )
