#!/bin/bash
# load-env.sh — load environment variables from .env files
#
# Source this from other scripts:
#   source "$(dirname "$0")/load-env.sh"
#
# Searches for .env in: project root (parent of scripts/), then ~/.config/worker-bee/.env
# Format: KEY=VALUE (one per line, # comments, blank lines ignored)
# Existing env vars are NOT overwritten.

_load_env_file() {
    local envfile="$1"
    [[ -f "$envfile" ]] || return 1
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%%#*}"        # strip comments
        line="${line#"${line%%[![:space:]]*}"}"  # trim leading whitespace
        line="${line%"${line##*[![:space:]]}"}"  # trim trailing whitespace
        [[ -z "$line" ]] && continue
        [[ "$line" == *=* ]] || continue
        local key="${line%%=*}"
        local val="${line#*=}"
        # Remove surrounding quotes
        val="${val#\"}" ; val="${val%\"}"
        val="${val#\'}" ; val="${val%\'}"
        # Don't overwrite existing env vars
        if [[ -z "${!key+x}" ]]; then
            export "$key=$val"
        fi
    done < "$envfile"
}

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJECT_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"

if [[ -f "$_PROJECT_ROOT/.env" ]]; then
    _load_env_file "$_PROJECT_ROOT/.env"
elif [[ -f "$HOME/.config/worker-bee/.env" ]]; then
    _load_env_file "$HOME/.config/worker-bee/.env"
fi

unset -f _load_env_file
unset _SCRIPT_DIR _PROJECT_ROOT
