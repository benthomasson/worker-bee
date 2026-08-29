"""Verify beliefs against source code via LLM."""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from worker_bee.dispatcher import dispatch
from worker_bee.tokens import count_tokens
from worker_bee.tracer import TraceResult, trace_belief, TOKEN_BUDGET

VERIFY_PROMPT = """\
You are verifying a belief from a Truth Maintenance System (TMS) against
actual source code. The belief was originally derived from a summary
document, not from the code itself. Your job is to check whether the
source code actually supports the claim.

## Belief
ID: {belief_id}
Claim: {belief_text}

## Source Summary (where the belief was derived)
{summary}

## Source Code
{code_sections}

## Task
Verify whether the belief claim is supported by the actual source code above.

Return ONLY a JSON object:

```json
{{
  "verified": true or false,
  "confidence": "high" or "medium" or "low",
  "evidence": "the specific code or pattern that supports or contradicts the claim",
  "comment": "one-sentence explanation of your verdict"
}}
```

Rules:
- "verified" means the source code confirms the claim, not just that it sounds plausible.
- If key files are missing from the source code section, set confidence to "low".
- Quote specific lines or patterns as evidence.
- Be precise: "default-features = false" is a verifiable fact, not an opinion."""


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
) -> list[VerifyResult]:
    """Verify unverified gated beliefs against source code."""
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
) -> VerifyResult:
    """Trace a belief to source code and verify it via LLM."""
    trace = trace_belief(db_path, belief_id, project_dir=project_dir)

    if not trace.summary_text:
        return VerifyResult(
            belief_id=trace.belief_id, belief_text=trace.belief_text,
            source=trace.source, code_refs=trace.code_refs,
            code_found=trace.code_found, code_missing=trace.code_missing,
        )

    prompt = VERIFY_PROMPT.format(
        belief_id=trace.belief_id,
        belief_text=trace.belief_text,
        summary=trace.summary_text,
        code_sections=trace.code_sections if trace.code_sections else "(no source code files found)",
    )

    token_count = count_tokens(prompt)
    print(f"  Prompt: ~{token_count:,} tokens", file=sys.stderr)

    if token_count > TOKEN_BUDGET:
        print(f"  WARNING: prompt exceeds {TOKEN_BUDGET:,} token budget", file=sys.stderr)

    if dry_run or verbose:
        print(f"\n{'='*60}")
        print(f"Verify: {trace.belief_id} (~{token_count:,} tokens)")
        print(f"{'='*60}")
        print(prompt)
        if dry_run:
            return VerifyResult(
                belief_id=trace.belief_id, belief_text=trace.belief_text,
                source=trace.source, code_refs=trace.code_refs,
                code_found=trace.code_found, code_missing=trace.code_missing,
            )

    print(f"  Dispatching to {model}...", file=sys.stderr)
    resp = dispatch(prompt, model=model)
    result = _parse_verify_response(resp.text)

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

    status = "VERIFIED" if verify.verified else "NOT VERIFIED"
    print(f"\n  Result: {status} (confidence: {verify.confidence})", file=sys.stderr)
    print(f"  Evidence: {verify.evidence}", file=sys.stderr)
    print(f"  Comment: {verify.comment}", file=sys.stderr)

    if verify.verified is not None:
        _update_db(str(db_path), verify)

    return verify


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
        node.metadata["verify_result"] = {
            "verified": result.verified,
            "confidence": result.confidence,
            "evidence": result.evidence,
            "comment": result.comment,
            "code_files": result.code_found,
            "verifier": "worker-bee",
        }


def _parse_verify_response(response: str) -> dict:
    """Extract verification result JSON from LLM response."""
    import json
    decoder = json.JSONDecoder()
    for i, ch in enumerate(response):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(response, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "verified" in obj:
            return obj
    return {}
