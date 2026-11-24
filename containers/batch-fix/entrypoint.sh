#!/bin/bash

# Worker-bee batch-fix container — runs fix-belief or batch-fix-beliefs
# inside a confined container with limited credentials.
#
# TASK_TYPE determines what runs:
#   fix-belief       — fix a single belief
#   batch-fix        — fix multiple beliefs sequentially
#   verify           — verify a single belief
#
# Required:
#   OPENAI_API_KEY   — LLM API key
#   REPO_URL         — git URL for the worker-bee target repo
#   EXPERT_URL       — git URL for the expert knowledge base repo
#
# Optional:
#   BELIEF_ID        — single belief ID (for fix-belief/verify)
#   BELIEF_IDS       — space-separated belief IDs (for batch-fix)
#   MODEL            — LLM model (default: openai:gpt-5.6-luna)
#   MAX_TURNS        — max LLM turns per step (default: 1000)
#   NUM_CTX          — context window size (default: 65536)
#   MAX_REVIEW_ROUNDS — max review rounds (default: 3)
#   MAX_VERIFY_ROUNDS — max verify rounds (default: 3)
#   GH_TOKEN         — GitHub token for private repos and pushing
#   PUSH_RESULTS     — if "true", push fix commits to remote

export PATH="/home/bee/.local/bin:$HOME/.local/bin:$PATH"

# When running as a different UID (mount mode with --user), /home/bee
# directories are unwritable.  Fall back to /tmp for venv and cache.
if [ -w /home/bee ]; then
    export UV_PROJECT_ENVIRONMENT="/home/bee/.venvs/repo"
else
    export UV_PROJECT_ENVIRONMENT="/tmp/uv-venv-$$"
    export UV_CACHE_DIR="/tmp/uv-cache-$$"
fi

TASK_TYPE="${TASK_TYPE:-batch-fix}"
MODEL="${MODEL:-openai:gpt-5.6-luna}"
MAX_TURNS="${MAX_TURNS:-1000}"
NUM_CTX="${NUM_CTX:-65536}"
MAX_REVIEW_ROUNDS="${MAX_REVIEW_ROUNDS:-3}"
MAX_VERIFY_ROUNDS="${MAX_VERIFY_ROUNDS:-3}"

WORK_DIR="/home/bee/workspace"
REPO_DIR="$WORK_DIR/repo"
EXPERT_DIR="$WORK_DIR/expert"
DB_PATH="$EXPERT_DIR/reasons.db"

# ─── Helpers ───

log() { echo ""; echo "══════════════════════════════════════════════════"; echo "  $*"; echo "══════════════════════════════════════════════════"; }
step() { echo "── $* ──"; }

fail() {
    echo "FATAL: $*" >&2
    exit 1
}

trap 'echo "[FATAL] Script failed at line $LINENO with exit code $?"' ERR
set -eE

# ─── Validate ───

if [ -z "$OPENAI_API_KEY" ]; then
    fail "OPENAI_API_KEY is required"
fi

if [ -z "$REPO_URL" ] && [ ! -d "$REPO_DIR/.git" ]; then
    fail "REPO_URL is required (or mount repo at $REPO_DIR)"
fi

if [ -z "$EXPERT_URL" ] && [ ! -d "$EXPERT_DIR/.git" ]; then
    fail "EXPERT_URL is required (or mount expert repo at $EXPERT_DIR)"
fi

# ─── Clone repos ───

clone_or_detect_repo() {
    local url="$1"
    local dest="$2"
    local name="$3"

    if [ -d "$dest/.git" ]; then
        step "Using mounted $name at $dest"
        return
    fi

    if [ -z "$url" ]; then
        fail "$name not mounted and no URL provided"
    fi

    step "Cloning $name"
    local clone_url="$url"
    if [ -n "$GH_TOKEN" ]; then
        clone_url=$(echo "$url" | sed "s|https://|https://${GH_TOKEN}@|")
    fi
    git clone "$clone_url" "$dest"
}

log "Setup"

clone_or_detect_repo "${REPO_URL:-}" "$REPO_DIR" "target repo"
clone_or_detect_repo "${EXPERT_URL:-}" "$EXPERT_DIR" "expert knowledge base"

if [ ! -f "$DB_PATH" ]; then
    fail "No reasons.db found at $DB_PATH"
fi

# If the database isn't writable (UID mismatch in mount mode), copy it
# to a temp location so worker-bee can retract/update beliefs.
ORIG_DB_PATH="$DB_PATH"
if [ ! -w "$DB_PATH" ]; then
    step "Database not writable — copying to temp location"
    TEMP_DB="/tmp/reasons-$$.db"
    cp "$DB_PATH" "$TEMP_DB"
    # Copy WAL/SHM if they exist
    [ -f "${DB_PATH}-wal" ] && cp "${DB_PATH}-wal" "${TEMP_DB}-wal"
    [ -f "${DB_PATH}-shm" ] && cp "${DB_PATH}-shm" "${TEMP_DB}-shm"
    DB_PATH="$TEMP_DB"
fi

# ─── Configure git push credentials ───

if [ -n "$GH_TOKEN" ] && [ -n "$REPO_URL" ]; then
    cd "$REPO_DIR"
    git remote set-url origin "$(echo "$REPO_URL" | sed "s|https://|https://${GH_TOKEN}@|")"
fi

# ═══════════════════════════════════════════
# Task: fix-belief
# ═══════════════════════════════════════════

run_fix_belief() {
    if [ -z "$BELIEF_ID" ]; then
        fail "BELIEF_ID is required for fix-belief task"
    fi

    log "Fixing belief: $BELIEF_ID"

    cd "$REPO_DIR"
    bash scripts/fix-belief.sh "$BELIEF_ID" \
        --model "$MODEL" \
        --db "$DB_PATH" \
        --repo "$REPO_DIR" \
        --max-turns "$MAX_TURNS" \
        --num-ctx "$NUM_CTX" \
        --max-review-rounds "$MAX_REVIEW_ROUNDS" \
        --max-verify-rounds "$MAX_VERIFY_ROUNDS" \
        2>&1

    if [ "$PUSH_RESULTS" = "true" ] && [ -n "$GH_TOKEN" ]; then
        step "Pushing results"
        cd "$REPO_DIR" && git push origin main 2>&1 || true
        cd "$EXPERT_DIR" && git push origin main 2>&1 || true
    fi
}

# ═══════════════════════════════════════════
# Task: batch-fix
# ═══════════════════════════════════════════

run_batch_fix() {
    if [ -z "$BELIEF_IDS" ]; then
        fail "BELIEF_IDS is required for batch-fix task (space-separated)"
    fi

    log "Batch fixing beliefs: $BELIEF_IDS"

    cd "$REPO_DIR"
    # shellcheck disable=SC2086
    bash scripts/batch-fix-beliefs.sh $BELIEF_IDS \
        --model "$MODEL" \
        --db "$DB_PATH" \
        --repo "$REPO_DIR" \
        --max-turns "$MAX_TURNS" \
        --num-ctx "$NUM_CTX" \
        2>&1

    if [ "$PUSH_RESULTS" = "true" ] && [ -n "$GH_TOKEN" ]; then
        step "Pushing results"
        cd "$REPO_DIR" && git push origin main 2>&1 || true
        cd "$EXPERT_DIR" && git push origin main 2>&1 || true
    fi
}

# ═══════════════════════════════════════════
# Task: verify
# ═══════════════════════════════════════════

run_verify() {
    if [ -z "$BELIEF_ID" ]; then
        fail "BELIEF_ID is required for verify task"
    fi

    log "Verifying belief: $BELIEF_ID"

    SCRATCH_DIR="/tmp/worker-bee-$$"
    mkdir -p "$SCRATCH_DIR"
    cd "$SCRATCH_DIR"
    worker-bee verify "$DB_PATH" "$BELIEF_ID" \
        --project-dir "$REPO_DIR" \
        --model "$MODEL" \
        --max-turns "$MAX_TURNS" \
        --num-ctx "$NUM_CTX" \
        --retract \
        2>&1
}

# ═══════════════════════════════════════════
# Dispatch
# ═══════════════════════════════════════════

case "$TASK_TYPE" in
    fix-belief)
        run_fix_belief
        ;;
    batch-fix)
        run_batch_fix
        ;;
    verify)
        run_verify
        ;;
    *)
        fail "Unknown task type: $TASK_TYPE"
        ;;
esac

# Copy the database back if we used a temp copy
if [ "$DB_PATH" != "$ORIG_DB_PATH" ] && [ -f "$DB_PATH" ]; then
    if cp "$DB_PATH" "$ORIG_DB_PATH" 2>/dev/null; then
        step "Database copied back to $ORIG_DB_PATH"
        [ -f "${DB_PATH}-wal" ] && cp "${DB_PATH}-wal" "${ORIG_DB_PATH}-wal" 2>/dev/null
        [ -f "${DB_PATH}-shm" ] && cp "${DB_PATH}-shm" "${ORIG_DB_PATH}-shm" 2>/dev/null
    else
        step "WARNING: Could not copy database back — results in $DB_PATH inside container only"
    fi
fi
