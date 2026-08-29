"""Command-line interface for worker-bee."""

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="worker-bee",
        description="Belief-driven orchestrator for small-context worker bees",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")

    subparsers = parser.add_subparsers(dest="command")

    sub_extract = subparsers.add_parser("extract", help="Extract issues from a belief database")
    sub_extract.add_argument("db", help="Path to reasons.db")

    sub_review = subparsers.add_parser("review", help="Review unreviewed beliefs against source documents")
    sub_review.add_argument("db", help="Path to reasons.db")
    sub_review.add_argument("--model", default="ollama:qwen3.8:27b", help="Model to use (e.g. ollama:qwen3.8:27b, claude, api:claude-sonnet-4-20250514)")
    sub_review.add_argument("--batch-size", type=int, default=5, help="Beliefs per LLM call (default: 5)")
    sub_review.add_argument("--limit", type=int, default=None, help="Max beliefs to review")
    sub_review.add_argument("--dry-run", action="store_true", help="Print prompts without dispatching")
    sub_review.add_argument("--verbose", "-v", action="store_true", help="Print prompts before dispatching")
    sub_review.add_argument("--retract", action="store_true", help="Retract beliefs found to be inaccurate")

    sub_trace = subparsers.add_parser("trace", help="Trace a belief back to its source code references")
    sub_trace.add_argument("db", help="Path to reasons.db")
    sub_trace.add_argument("belief_id", help="ID of the belief to trace")
    sub_trace.add_argument("--project-dir", default=None, help="Path to the source project (guessed from db path if omitted)")

    sub_verify = subparsers.add_parser("verify", help="Verify a belief against source code via LLM")
    sub_verify.add_argument("db", help="Path to reasons.db")
    sub_verify.add_argument("belief_id", help="ID of the belief to verify")
    sub_verify.add_argument("--project-dir", default=None, help="Path to the source project (guessed from db path if omitted)")
    sub_verify.add_argument("--model", default="ollama:qwen3.8:27b", help="Model to use")
    sub_verify.add_argument("--dry-run", action="store_true", help="Print prompt without dispatching")
    sub_verify.add_argument("--verbose", "-v", action="store_true", help="Print prompt before dispatching")

    sub_run = subparsers.add_parser("run", help="Run the full pipeline on a belief database")
    sub_run.add_argument("db", help="Path to reasons.db")
    sub_run.add_argument("--model", default="ollama:qwen3.8:27b", help="Model to use (e.g. ollama:qwen3.8:27b, claude, api:claude-sonnet-4-20250514)")
    sub_run.add_argument("--dry-run", action="store_true", help="Print prompts without dispatching")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "extract":
        from worker_bee.extractor import extract
        issues = extract(args.db)
        for issue in issues:
            print(f"[{issue['type']}] {issue['belief_id']}: {issue['description']}")
        return 0

    if args.command == "review":
        from worker_bee.reviewer import review_unreviewed
        results = review_unreviewed(
            args.db,
            model=args.model,
            batch_size=args.batch_size,
            limit=args.limit,
            dry_run=args.dry_run,
            verbose=args.verbose,
            retract_inaccurate=args.retract,
        )
        if results:
            accurate = sum(1 for r in results if r.accurate)
            inaccurate = sum(1 for r in results if not r.accurate)
            print(f"\nSummary: {accurate} accurate, {inaccurate} inaccurate out of {len(results)} reviewed.")
        return 0

    if args.command == "trace":
        from worker_bee.tracer import trace_belief
        trace = trace_belief(
            args.db,
            args.belief_id,
            project_dir=args.project_dir,
        )
        if trace.code_found:
            print(f"\nCode files loaded:")
            for f in trace.code_found:
                print(f"  {f}")
        if trace.code_missing:
            print(f"\nCode files missing:")
            for f in trace.code_missing:
                print(f"  {f}")
        if not trace.code_found and not trace.code_missing:
            print("\nNo code references found in source summary.")
        return 0

    if args.command == "verify":
        from worker_bee.verifier import verify_belief
        result = verify_belief(
            args.db,
            args.belief_id,
            project_dir=args.project_dir,
            model=args.model,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        if result.verified is not None:
            status = "VERIFIED" if result.verified else "NOT VERIFIED"
            print(f"\n{result.belief_id}: {status} ({result.confidence})")
            print(f"  {result.comment}")
        return 0

    if args.command == "run":
        from worker_bee.pipeline import run_pipeline
        run_pipeline(args.db, model=args.model, dry_run=args.dry_run)
        return 0

    return 0


def _version() -> str:
    from worker_bee import __version__
    return __version__


if __name__ == "__main__":
    sys.exit(main())
