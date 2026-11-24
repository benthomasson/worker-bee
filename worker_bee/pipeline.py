"""Run the full extract-prompt-dispatch pipeline."""

from __future__ import annotations

import sys

from worker_bee.extractor import extract
from worker_bee.prompter import build_prompt
from worker_bee.dispatcher import configure_rate_limit, dispatch_batch


def run_pipeline(
    db_path: str,
    *,
    model: str = "ollama:qwen3.8:27b",
    dry_run: bool = False,
    max_workers: int = 1,
    requests_per_second: float | None = None,
) -> list[dict]:
    """Run the first-milestone pipeline: extract → prompt → dispatch → print.

    A rate supplied for this invocation is cleared when the pipeline exits, so
    it does not unexpectedly affect later calls in the same process.
    """
    if requests_per_second is not None:
        configure_rate_limit(model, requests_per_second)
    try:
        return _run_pipeline(
            db_path,
            model=model,
            dry_run=dry_run,
            max_workers=max_workers,
        )
    finally:
        if requests_per_second is not None:
            configure_rate_limit(model, None)


def _run_pipeline(
    db_path: str,
    *,
    model: str,
    dry_run: bool,
    max_workers: int,
) -> list[dict]:
    issues = extract(db_path)

    if not issues:
        print("No issues found.", file=sys.stderr)
        return []

    print(f"Found {len(issues)} issue(s).", file=sys.stderr)

    prompts = [(issue, build_prompt(issue)) for issue in issues]

    if dry_run:
        from worker_bee.tokens import count_tokens

        results = []
        for issue, prompt in prompts:
            token_count = count_tokens(prompt)
            print(f"\n{'='*60}")
            print(f"Issue: [{issue['type']}] {issue['belief_id']} (~{token_count:,} tokens)")
            print(f"{'='*60}")
            print(prompt)
            results.append({"issue": issue, "prompt": prompt, "response": None})
        return results

    print(f"\nDispatching {len(prompts)} prompt(s) (max_workers={max_workers})...", file=sys.stderr)
    results = []
    responses = dispatch_batch(prompts, model=model, max_workers=max_workers)
    for (issue, prompt), (_, resp) in zip(prompts, responses):
        if resp.error:
            print(f"\nDispatch failed: [{issue['type']}] {issue['belief_id']}: {resp.error}", file=sys.stderr)
        print(f"\n{'='*60}")
        print(f"Issue: [{issue['type']}] {issue['belief_id']}")
        if resp.error is None:
            print(f"Tokens: {resp.prompt_tokens} prompt, {resp.completion_tokens} completion")
        print(f"{'='*60}")
        print(resp.text)
        results.append({"issue": issue, "prompt": prompt, "response": resp})

    return results
