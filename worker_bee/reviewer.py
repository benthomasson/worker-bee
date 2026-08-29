"""Review unreviewed beliefs against their source documents."""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from worker_bee.dispatcher import dispatch

REVIEW_BATCH_SIZE = 5

REVIEW_PROMPT = """\
You are auditing beliefs in a Truth Maintenance System (TMS).
Each belief below was extracted from a source document by an earlier LLM pass.
The system stored the claim but did not verify whether the source actually
says what the belief claims.

For each belief, the source document text is provided. Evaluate two axes:

1. **Accurate**: Does the source material actually state or clearly imply
   this claim? Watch for:
   - Details added that the source never mentions (misread_source)
   - Claims that go beyond what the source supports (overgeneralized)
   - Information fabricated with no basis in the source (fabricated)
   - Claims the source does not address at all (unsupported)

2. **Well-scoped**: Is the claim appropriately scoped? A belief that says
   "X always does Y" when the source says "X sometimes does Y" is
   overgeneralized even if the core idea is present.

Return ONLY a JSON array. For each belief, one object:

```json
[
  {{
    "id": "belief-id",
    "accurate": true,
    "well_scoped": true,
    "error_type": null,
    "comment": "brief explanation"
  }}
]
```

Rules:
- Return one object per belief reviewed, in the same order as presented.
- "error_type" must be one of: "misread_source", "overgeneralized",
  "fabricated", "unsupported", or null if the belief is accurate.
- "comment" should be a single sentence explaining the most important finding.
- If accurate is false, error_type MUST be set (not null).
- If accurate is true, error_type MUST be null.
- Be rigorous: a claim that sounds reasonable but isn't in the source is NOT accurate.
- Focus on what the source document says, not on general knowledge.

## Beliefs to review

{beliefs}"""


@dataclass
class ReviewResult:
    id: str
    accurate: bool
    well_scoped: bool
    error_type: str | None
    comment: str


def review_unreviewed(
    db_path: str | Path,
    *,
    model: str = "ollama:qwen3:27b",
    batch_size: int = REVIEW_BATCH_SIZE,
    limit: int | None = None,
    dry_run: bool = False,
    retract_inaccurate: bool = False,
) -> list[ReviewResult]:
    """Review unreviewed beliefs against their source documents.

    Returns a list of ReviewResult for each belief reviewed.
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT id, text, source
        FROM nodes
        WHERE truth_value = 'IN'
          AND created_at != ''
          AND (reviewed_at IS NULL OR reviewed_at = '')
          AND source != ''
        ORDER BY id
    """).fetchall()

    if limit is not None:
        rows = rows[:limit]

    if not rows:
        print("No unreviewed beliefs with sources found.", file=sys.stderr)
        conn.close()
        return []

    print(f"Found {len(rows)} unreviewed belief(s) to review.", file=sys.stderr)

    source_cache: dict[str, str] = {}
    all_results: list[ReviewResult] = []

    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]

    for batch_num, batch in enumerate(batches, 1):
        print(f"  Reviewing batch {batch_num}/{len(batches)} "
              f"({len(batch)} beliefs)...", file=sys.stderr)

        beliefs_text = "\n\n".join(
            _format_belief(row, source_cache, db_path.parent)
            for row in batch
        )
        prompt = REVIEW_PROMPT.format(beliefs=beliefs_text)

        if dry_run:
            print(f"\n{'='*60}")
            print(f"Batch {batch_num}/{len(batches)}")
            print(f"{'='*60}")
            print(prompt)
            continue

        try:
            resp = dispatch(prompt, model=model)
            results = _parse_review_response(resp.text)
            all_results.extend(results)

            for r in results:
                status = "PASS" if r.accurate else f"FAIL ({r.error_type})"
                print(f"    {r.id}: {status} — {r.comment}", file=sys.stderr)

        except Exception as e:
            print(f"  WARN: batch {batch_num} failed: {e}", file=sys.stderr)

    if not dry_run and all_results:
        _update_db(conn, all_results, retract_inaccurate=retract_inaccurate)

    conn.close()
    return all_results


def _format_belief(row: sqlite3.Row, source_cache: dict, base_dir: Path) -> str:
    """Format one belief with its source text for LLM review."""
    lines = [f"### {row['id']}"]
    lines.append(f"Claim: {row['text']}")

    source = row["source"]
    lines.append(f"Source reference: {source}")
    lines.append("")

    content = _load_source(source, source_cache, base_dir)
    lines.append("Source document content:")
    lines.append("```")
    lines.append(content)
    lines.append("```")

    return "\n".join(lines)


def _load_source(source: str, cache: dict, base_dir: Path) -> str:
    """Load source file content, caching for reuse within a batch."""
    if source in cache:
        return cache[source]

    p = Path(source)
    if not p.is_absolute():
        p = base_dir / p

    if not p.exists():
        cache[source] = "(source not available)"
    else:
        try:
            cache[source] = p.read_text(errors="replace")
        except Exception:
            cache[source] = "(error reading source)"

    return cache[source]


def _parse_review_response(response: str) -> list[ReviewResult]:
    """Extract review results JSON array from LLM response."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(response):
        if ch != "[":
            continue
        try:
            items, _ = decoder.raw_decode(response, i)
        except json.JSONDecodeError:
            continue
        if not isinstance(items, list):
            continue
        results = []
        for item in items:
            if not isinstance(item, dict) or "id" not in item:
                continue
            results.append(ReviewResult(
                id=item["id"],
                accurate=item.get("accurate", True),
                well_scoped=item.get("well_scoped", True),
                error_type=item.get("error_type"),
                comment=item.get("comment", ""),
            ))
        if results:
            return results
    return []


def _update_db(
    conn: sqlite3.Connection,
    results: list[ReviewResult],
    *,
    retract_inaccurate: bool = False,
) -> None:
    """Write review results back to reasons.db."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    reviewed = 0
    retracted = 0

    for r in results:
        row = conn.execute("SELECT id, metadata_json FROM nodes WHERE id = ?", (r.id,)).fetchone()
        if not row:
            continue

        metadata = json.loads(row["metadata_json"] or "{}")
        metadata["review_result"] = {
            "accurate": r.accurate,
            "well_scoped": r.well_scoped,
            "error_type": r.error_type,
            "comment": r.comment,
            "reviewer": "worker-bee",
        }

        conn.execute(
            "UPDATE nodes SET reviewed_at = ?, updated_at = ?, metadata_json = ? WHERE id = ?",
            (now, now, json.dumps(metadata), r.id),
        )
        reviewed += 1

        if retract_inaccurate and not r.accurate:
            conn.execute(
                "UPDATE nodes SET truth_value = 'OUT', retracted_at = ? WHERE id = ?",
                (now, r.id),
            )
            retracted += 1

    conn.commit()

    print(f"\n  Updated {reviewed} belief(s) in reasons.db.", file=sys.stderr)
    if retracted:
        print(f"  Retracted {retracted} inaccurate belief(s).", file=sys.stderr)
