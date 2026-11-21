#!/bin/bash
source "$(dirname "$0")/load-env.sh"
reasonsforge code review \
    --reasons-db ~/git/worker-bee-expert/reasons.db \
    --repo . \
    --model claude \
    --base main \
    --branch $(git branch --show-current)
