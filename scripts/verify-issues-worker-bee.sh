#!/bin/bash
source "$(dirname "$0")/load-env.sh"
uv run worker-bee verify ~/git/worker-bee-expert/reasons.db \
    --project-dir ~/git/worker-bee \
    --model openai:gpt-5.6-luna \
    "$@"
