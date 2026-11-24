"""Run the full extract-prompt-dispatch pipeline."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import fcntl

from worker_bee.extractor import extract, filter_acceptance_gates
from worker_bee.prompter import build_prompt
from worker_bee.dispatcher import configure_rate_limit, dispatch_batch, Response


def run_pipeline(
    db_path: str,
    *,
    model: str = "ollama:qwen3.8:27b",
    dry_run: bool = False,
    max_workers: int = 1,
    requests_per_second: float | None = None,
    progress_file: str | Path | None = None,
    resume: bool = False,
    require_reviewed: bool = True,
    require_verified: bool = True,
) -> list[dict]:
    """Run the first-milestone pipeline: extract → prompt → dispatch → print.

    A rate supplied for this invocation is cleared when the pipeline exits, so
    it does not unexpectedly affect later calls in the same process.  Normal
    runs use ``<db_path>.progress.json`` and resume it automatically; dry runs
    do not create a checkpoint.  Progress mode checkpoints items serially after
    each response; failed responses are recorded but retried on resume.
    """
    # Keep restart safety at the pipeline API boundary, rather than relying on
    # each caller to remember to configure a checkpoint.  An explicitly supplied
    # progress path retains its existing --resume requirement.
    if progress_file is None and not dry_run and not resume:
        progress_file = f"{db_path}.progress.json"
        resume = True
    if resume and progress_file is None:
        raise ValueError("--resume requires --progress-file")
    if dry_run and progress_file is not None:
        raise ValueError("--dry-run cannot be combined with --progress-file")
    if requests_per_second is not None:
        configure_rate_limit(model, requests_per_second)
    try:
        with _progress_lock(progress_file):
            if progress_file is not None and not resume and Path(progress_file).exists():
                raise ValueError("progress file already exists; use --resume to continue it")
            return _run_pipeline(
                db_path,
                model=model,
                dry_run=dry_run,
                max_workers=max_workers,
                progress_file=progress_file,
                resume=resume,
                require_reviewed=require_reviewed,
                require_verified=require_verified,
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
    progress_file: str | Path | None,
    resume: bool,
    require_reviewed: bool,
    require_verified: bool,
) -> list[dict]:
    extracted_issues = extract(db_path)
    extracted_issues = filter_acceptance_gates(
        db_path,
        extracted_issues,
        require_reviewed=require_reviewed,
        require_verified=require_verified,
    )
    keys = [_issue_key(issue) for issue in extracted_issues]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate issue keys in extracted issues")
    progress_path = Path(progress_file) if progress_file is not None else None
    saved = _load_progress(progress_path) if resume and progress_path else {}
    all_prompts = [(issue, build_prompt(issue)) for issue in extracted_issues]

    # A checkpoint is reusable only when it describes the exact current issue
    # and prompt, and contains a successful response.  This prevents a changed
    # belief from silently receiving an answer generated for an older version.
    prompts = []
    for issue, prompt in all_prompts:
        item = saved.get(_issue_key(issue))
        if (item is None or item["issue"] != issue or item["prompt"] != prompt
                or item["response"]["error"] is not None):
            prompts.append((issue, prompt))

    if not all_prompts:
        print("No issues found.", file=sys.stderr)
        return []
    if not prompts and progress_path:
        return [_result_from_saved(saved[_issue_key(issue)]) for issue in extracted_issues]

    print(f"Found {len(prompts)} issue(s).", file=sys.stderr)

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
    if progress_path is None:
        responses = dispatch_batch(prompts, model=model, max_workers=max_workers)
    else:
        # Process one item at a time so a durable checkpoint is written after
        # every response. Failed items remain retryable on the next invocation.
        responses = []
        for issue, prompt in prompts:
            response = dispatch_batch([(issue, prompt)], model=model, max_workers=1)[0][1]
            result = {"issue": issue, "prompt": prompt, "response": response}
            # Record both successes and failures.  A failure remains retryable
            # because resume excludes error responses from the completed set.
            saved[_issue_key(issue)] = _serialize_result(result)
            _write_progress(progress_path, saved)
            responses.append((issue, response))
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

    if progress_path is not None and saved:
        # Include completed earlier items in current extraction order, ignoring
        # stale checkpoint entries for issues no longer present.
        current = {_issue_key(result["issue"]): result for result in results}
        results = []
        for issue in extracted_issues:
            key = _issue_key(issue)
            if key in current:
                results.append(current[key])
            elif key in saved:
                results.append(_result_from_saved(saved[key]))
    return results


def _issue_key(issue: dict) -> str:
    return f"{issue.get('type', '')}:{issue.get('belief_id', '')}"


def _serialize_result(result: dict) -> dict:
    response = result["response"]
    return {
        "issue": result["issue"],
        "prompt": result["prompt"],
        "response": {
            "text": response.text,
            "model": response.model,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "error": response.error,
        },
    }


def _result_from_saved(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != {"issue", "prompt", "response"}:
        raise ValueError("invalid progress item")
    issue, prompt, response = value["issue"], value["prompt"], value["response"]
    if not isinstance(issue, dict) or not isinstance(prompt, str) or not isinstance(response, dict):
        raise ValueError("invalid progress item")
    fields = {"text", "model", "prompt_tokens", "completion_tokens", "error"}
    if set(response) != fields:
        raise ValueError("invalid progress response")
    if (not isinstance(response["text"], str) or not isinstance(response["model"], str)
            or type(response["prompt_tokens"]) is not int
            or type(response["completion_tokens"]) is not int
            or (response["error"] is not None and not isinstance(response["error"], str))):
        raise ValueError("invalid progress response")
    return {"issue": issue, "prompt": prompt, "response": Response(**response)}


@contextmanager
def _progress_lock(path: str | Path | None):
    """Serialize pipeline owners that share a checkpoint file."""
    if path is None:
        yield
        return
    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    lock_path = checkpoint.with_name(f".{checkpoint.name}.lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _load_progress(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid progress file JSON: {exc.msg}") from exc
    except OSError as exc:
        raise ValueError(f"cannot read progress file: {exc}") from exc
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("items"), dict):
        raise ValueError("invalid progress file (expected version 1)")
    items = data["items"]
    for key, value in items.items():
        if not isinstance(key, str):
            raise ValueError("invalid progress item key")
        try:
            item = _result_from_saved(value)
        except ValueError as exc:
            raise ValueError(f"invalid progress item {key!r}: {exc}") from exc
        if _issue_key(item["issue"]) != key:
            raise ValueError(f"progress item key does not match issue: {key!r}")
    return items


def _write_progress(path: Path, items: dict) -> None:
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"version": 1, "items": items}, indent=2, sort_keys=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise ValueError(f"cannot write progress file: {exc}") from exc
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
