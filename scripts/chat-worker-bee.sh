#!/bin/bash
source "$(dirname "$0")/load-env.sh"
uv run worker-bee chat --db ~/git/worker-bee-expert/reasons.db --brain brain.db --model openai:gpt-5.6-luna
