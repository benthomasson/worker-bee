#!/bin/bash
source "$(dirname "$0")/load-env.sh"
uv run worker-bee fix ~/git/worker-bee-expert/reasons.db "$1" \
    --project-dir ~/git/worker-bee \
    --model openai:gpt-5.6-luna \
    --max-turns 1000 \
    "${@:2}"
