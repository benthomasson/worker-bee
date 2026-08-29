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

    sub_run = subparsers.add_parser("run", help="Run the full pipeline on a belief database")
    sub_run.add_argument("db", help="Path to reasons.db")
    sub_run.add_argument("--model", default="qwen3:27b", help="Ollama model to use")
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
