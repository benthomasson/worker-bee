#!/bin/bash
source "$(dirname "$0")/load-env.sh"
reasonsforge code review \
    --reasons-db ~/git/worker-bee-expert/reasons.db \
    --model openai:gpt-5.6-luna \
    --base main \
    --branch $(git branch --show-current)
