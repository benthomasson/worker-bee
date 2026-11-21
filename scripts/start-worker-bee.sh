#!/bin/bash
source "$(dirname "$0")/load-env.sh"
uv run worker-bee chat --mcp http://127.0.0.1:6519/mcp --model openai:gpt-5.6-luna --mcp-domain personal
