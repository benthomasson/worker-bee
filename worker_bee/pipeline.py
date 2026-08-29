"""Run the full extract-prompt-dispatch pipeline."""

from __future__ import annotations

import sys

from worker_bee.extractor import extract
from worker_bee.prompter import build_prompt
from worker_bee.dispatcher import dispatch


def run_pipeline(
    db_path: str,
    *,
    model: str = "ollama:qwen3:27b",
    dry_run: bool = False,
) -> list[dict]:
    """Run the first-milestone pipeline: extract → prompt → dispatch → print."""
    issues = extract(db_path)

    if not issues:
        print("No issues found.", file=sys.stderr)
        return []

    print(f"Found {len(issues)} issue(s).", file=sys.stderr)

    results = []
    for issue in issues:
        prompt = build_prompt(issue)

        if dry_run:
            print(f"\n{'='*60}")
            print(f"Issue: [{issue['type']}] {issue['belief_id']}")
            print(f"{'='*60}")
            print(prompt)
            results.append({"issue": issue, "prompt": prompt, "response": None})
            continue

        print(f"\nDispatching: [{issue['type']}] {issue['belief_id']}...", file=sys.stderr)
        resp = dispatch(prompt, model=model)
        print(f"\n{'='*60}")
        print(f"Issue: [{issue['type']}] {issue['belief_id']}")
        print(f"Tokens: {resp.prompt_tokens} prompt, {resp.completion_tokens} completion")
        print(f"{'='*60}")
        print(resp.text)
        results.append({"issue": issue, "prompt": prompt, "response": resp})

    return results
