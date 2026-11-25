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

# ─── Fix helpers ───

has_changes() {
    cd "$REPO_DIR"
    [[ -n "$(git diff --name-only)" ]] || \
    [[ -n "$(git diff --cached --name-only)" ]] || \
    [[ -n "$(git ls-files --others --exclude-standard)" ]]
}

commit_changes() {
    local msg="$1"
    cd "$REPO_DIR"
    git add -A
    git reset HEAD -- .worker-bee/ 2>/dev/null || true
    git diff --cached --quiet && return 0
    git commit -m "$msg" || true
}

model_email() {
    local model="$1"
    case "$model" in
        openai:*)   echo "noreply@openai.com" ;;
        claude*|anthropic:*|api:*) echo "noreply@anthropic.com" ;;
        ollama:*)   echo "noreply@ollama.com" ;;
        *)          echo "noreply@worker-bee" ;;
    esac
}

abandon_branch() {
    local branch="$1"
    local msg="$2"
    cd "$REPO_DIR"
    if has_changes; then
        commit_changes "FAILED: $msg"
    fi
    git checkout main
    echo "Returned to main. Fix branch $branch preserved for inspection."
}

get_test_command() {
    local config="$EXPERT_DIR/.forge/code/config.json"
    if [ -f "$config" ]; then
        python3 -c "import json,sys; c=json.load(open(sys.argv[1])); print(c.get('test_command',''))" "$config" 2>/dev/null || echo ""
    else
        echo ""
    fi
}

run_tests() {
    local test_cmd
    test_cmd=$(get_test_command)
    if [ -z "$test_cmd" ]; then
        log "No test_command configured — skipping tests"
        return 0
    fi
    step "Running tests: $test_cmd"
    cd "$REPO_DIR"
    if eval "$test_cmd"; then
        log "Tests passed"
        return 0
    else
        log "Tests FAILED"
        return 1
    fi
}

# ═══════════════════════════════════════════
# Task: fix-belief (single belief pipeline)
# ═══════════════════════════════════════════
#
# Pipeline:
#   1. Verify the belief is real
#   2. Create branch and fix it
#   3. Review-fix loop
#   4. Verify-fix (with PARTIAL gap re-fix)
#   5. Squash-merge to main on FIXED
#

run_fix_belief() {
    if [ -z "$BELIEF_ID" ]; then
        fail "BELIEF_ID is required for fix-belief task"
    fi

    local BRANCH="fix-${BELIEF_ID}"
    local SCRATCH_DIR="/tmp/worker-bee-fix-$$"
    mkdir -p "$SCRATCH_DIR"

    # ── step 1: verify ──
    log "Step 1: Verify belief is real"

    local VERIFY_OUTPUT
    VERIFY_OUTPUT=$(cd "$SCRATCH_DIR" && worker-bee verify "$DB_PATH" "$BELIEF_ID" \
        --project-dir "$REPO_DIR" \
        --model "$MODEL" \
        --max-turns "$MAX_TURNS" \
        --num-ctx "$NUM_CTX" \
        --retract 2>&1) || true
    echo "$VERIFY_OUTPUT"

    if echo "$VERIFY_OUTPUT" | grep -q "NOT VERIFIED"; then
        echo ""
        echo "Belief is NOT VERIFIED and retracted by verifier."
        return 0
    fi

    if ! echo "$VERIFY_OUTPUT" | grep -q "VERIFIED"; then
        fail "Could not determine verification result"
    fi

    step "Belief verified as real — proceeding to fix"

    # ── step 2: create branch and fix ──
    log "Step 2: Create branch and initial fix"

    cd "$REPO_DIR"
    if has_changes; then
        git stash push -u -m "fix-belief: stash before $BELIEF_ID"
    fi

    local CURRENT
    CURRENT=$(git branch --show-current)
    if [[ "$CURRENT" != "$BRANCH" ]]; then
        git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
    fi

    local BASE_COMMIT
    BASE_COMMIT=$(git rev-parse HEAD)

    step "Running worker-bee fix"
    cd "$SCRATCH_DIR"
    worker-bee fix "$DB_PATH" "$BELIEF_ID" \
        --project-dir "$REPO_DIR" \
        --model "$MODEL" \
        --max-turns "$MAX_TURNS" \
        --num-ctx "$NUM_CTX" 2>&1 || true

    if ! has_changes; then
        abandon_branch "$BRANCH" "worker-bee fix made no changes for $BELIEF_ID"
        return 1
    fi

    commit_changes "fix $BELIEF_ID"

    # ── step 3+4: review-fix loop then verify-fix ──
    local verify_round review_round
    for verify_round in $(seq 1 "$MAX_VERIFY_ROUNDS"); do
        log "Verify round $verify_round/$MAX_VERIFY_ROUNDS"

        # ── review-fix loop ──
        for review_round in $(seq 1 "$MAX_REVIEW_ROUNDS"); do
            step "Review round $review_round/$MAX_REVIEW_ROUNDS"

            cd "$REPO_DIR"
            local REVIEW_EXIT=0
            reasonsforge code review \
                --reasons-db "$DB_PATH" \
                --repo "$REPO_DIR" \
                --model "$MODEL" \
                --base main \
                --branch "$BRANCH" \
                --fixing-belief "$BELIEF_ID" \
                2>&1 || REVIEW_EXIT=$?

            local GATE
            case "$REVIEW_EXIT" in
                0) GATE="PASS" ;;
                1) GATE="CONCERN" ;;
                2) GATE="BLOCK" ;;
                *) GATE="ERROR($REVIEW_EXIT)" ;;
            esac

            step "Review gate: $GATE (exit $REVIEW_EXIT)"

            if [[ "$GATE" == "PASS" ]]; then
                echo "Review passed!"
                break
            fi

            if [[ "$GATE" == ERROR* ]]; then
                echo "WARNING: Unexpected review exit code: $REVIEW_EXIT"
                break
            fi

            if [[ "$review_round" -eq "$MAX_REVIEW_ROUNDS" ]]; then
                echo "Max review rounds reached (gate: $GATE), proceeding to verify-fix"
                break
            fi

            step "Feeding review findings back to worker-bee edit"

            local REVIEW_DIR REPORT
            REVIEW_DIR=$(ls -td "$REPO_DIR/reviews/$BRANCH"/2* 2>/dev/null | head -1)
            REPORT="${REVIEW_DIR:+$REVIEW_DIR/report.md}"
            if [[ -z "$REPORT" ]] || [[ ! -f "$REPORT" ]]; then
                echo "WARNING: No review report found for edit task, skipping"
                break
            fi

            local TASK="Address the following code review findings for the fix of belief '$BELIEF_ID'.

The review report is below. Fix all BLOCK and CONCERN items.
Do not introduce new functionality beyond what is needed to address the findings.

--- REVIEW REPORT ---
$(cat "$REPORT")
--- END REVIEW REPORT ---"

            cd "$REPO_DIR"
            worker-bee edit "$TASK" \
                --db "$DB_PATH" \
                --model "$MODEL" \
                --max-turns "$MAX_TURNS" \
                --num-ctx "$NUM_CTX" 2>&1 || true

            if has_changes; then
                commit_changes "review fixes #$review_round for $BELIEF_ID"
            else
                echo "No changes from review-fix, proceeding"
                break
            fi
        done

        if has_changes; then
            commit_changes "additional files for $BELIEF_ID"
        fi

        # ── verify-fix ──
        step "Running verify-fix"

        local HEAD_COMMIT VERIFY_JSON VERDICT
        HEAD_COMMIT=$(cd "$REPO_DIR" && git rev-parse HEAD)
        VERIFY_JSON="/tmp/verify-fix-${BELIEF_ID}-${verify_round}.json"
        reasonsforge --db "$DB_PATH" code verify-fix "$BELIEF_ID" \
            --commit "$BASE_COMMIT..$HEAD_COMMIT" \
            --repo "$REPO_DIR" \
            --model "$MODEL" \
            --output-json "$VERIFY_JSON" 2>&1 || true

        if [[ ! -f "$VERIFY_JSON" ]]; then
            fail "verify-fix produced no structured output"
        fi

        VERDICT=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['verdict'])" "$VERIFY_JSON" 2>/dev/null || echo "UNKNOWN")
        step "verify-fix verdict: $VERDICT"

        if [[ "$VERDICT" == "FIXED" ]]; then
            log "SUCCESS: Belief retracted"
            python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(f\"  Verdict: {d['verdict']}\")
print(f\"  Reason: {d['reason']}\")
" "$VERIFY_JSON" || true

            if ! run_tests; then
                log "Tests failed — fix branch $BRANCH preserved for inspection"
                cd "$REPO_DIR" && git checkout main
                return 1
            fi

            step "Squash-merging $BRANCH into main"
            cd "$REPO_DIR"
            local COMMIT_COUNT
            COMMIT_COUNT=$(git rev-list --count main.."$BRANCH")
            git checkout main
            git merge --squash "$BRANCH"
            git commit -m "Fix $BELIEF_ID

$(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get('reason', ''))
" "$VERIFY_JSON" 2>/dev/null)

Squashed $COMMIT_COUNT commit(s) from $BRANCH.

Co-authored-by: worker-bee $MODEL <$(model_email "$MODEL")>"
            git branch -D "$BRANCH"
            echo "Merged to main and deleted $BRANCH."

            if [ "$PUSH_RESULTS" = "true" ] && [ -n "$GH_TOKEN" ]; then
                step "Pushing results"
                cd "$REPO_DIR" && git push origin main 2>&1 || true
                cd "$EXPERT_DIR" && git push origin main 2>&1 || true
            fi
            return 0
        fi

        if [[ "$VERDICT" == "PARTIAL" ]]; then
            if [[ "$verify_round" -eq "$MAX_VERIFY_ROUNDS" ]]; then
                log "PARTIAL after $MAX_VERIFY_ROUNDS verify rounds — stopping"
                python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1])),indent=2))" "$VERIFY_JSON" || true
                abandon_branch "$BRANCH" "PARTIAL after $MAX_VERIFY_ROUNDS verify rounds for $BELIEF_ID"
                return 1
            fi

            step "Fix is PARTIAL — reading gap from structured output"

            local GAP_DESC
            GAP_DESC=$(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
gap = d.get('gap_description', '')
reason = d.get('reason', '')
print(gap or reason)
" "$VERIFY_JSON")

            local GAP_FILES_ARGS=()
            while IFS= read -r gf; do
                [[ -n "$gf" ]] && GAP_FILES_ARGS+=(--gap-files "$gf")
            done < <(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
for f in d.get('affected_files', []):
    print(f)
" "$VERIFY_JSON")

            cd "$SCRATCH_DIR"
            worker-bee fix "$DB_PATH" "$BELIEF_ID" \
                --project-dir "$REPO_DIR" \
                --model "$MODEL" \
                --max-turns "$MAX_TURNS" \
                --num-ctx "$NUM_CTX" \
                --gap "$GAP_DESC" \
                "${GAP_FILES_ARGS[@]}" \
                2>&1 || true

            if has_changes; then
                commit_changes "fix remaining gap for $BELIEF_ID (round $verify_round)"
            else
                echo "No changes from gap fix attempt"
            fi

            continue
        fi

        # Neither FIXED nor PARTIAL
        log "verify-fix verdict: $VERDICT"
        python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1])),indent=2))" "$VERIFY_JSON" 2>/dev/null || true
        abandon_branch "$BRANCH" "verify-fix returned $VERDICT for $BELIEF_ID"
        return 1
    done

    log "Exhausted all verify rounds without full fix"
    abandon_branch "$BRANCH" "exhausted $MAX_VERIFY_ROUNDS verify rounds for $BELIEF_ID"
    return 1
}

# ═══════════════════════════════════════════
# Task: batch-fix
# ═══════════════════════════════════════════

run_batch_fix() {
    if [ -z "$BELIEF_IDS" ]; then
        fail "BELIEF_IDS is required for batch-fix task (space-separated)"
    fi

    log "Batch fixing beliefs: $BELIEF_IDS"

    local FIXED=() RETRACTED=() FAILED=() SKIPPED=()
    local beliefs
    # shellcheck disable=SC2206
    read -ra beliefs <<< "$BELIEF_IDS"
    local total=${#beliefs[@]}
    local count=0

    for belief in "${beliefs[@]}"; do
        count=$((count + 1))
        log "Batch: $belief ($count/$total)"

        BELIEF_ID="$belief"
        local EXIT_CODE=0
        local LOG_FILE="/tmp/batch-${belief}.log"
        run_fix_belief 2>&1 | tee "$LOG_FILE" || EXIT_CODE=$?

        if [[ $EXIT_CODE -eq 0 ]]; then
            if grep -q "SUCCESS: Belief retracted" "$LOG_FILE"; then
                FIXED+=("$belief")
                echo "  -> FIXED"
            elif grep -q "NOT VERIFIED" "$LOG_FILE"; then
                RETRACTED+=("$belief")
                echo "  -> AUTO-RETRACTED"
            else
                SKIPPED+=("$belief")
                echo "  -> SKIPPED"
            fi
        else
            FAILED+=("$belief")
            echo "  -> FAILED (exit $EXIT_CODE)"
        fi
    done

    # ── summary ──
    log "Batch Summary"
    echo "  Total:          $total"
    echo "  Fixed:          ${#FIXED[@]}"
    echo "  Auto-retracted: ${#RETRACTED[@]}"
    echo "  Failed:         ${#FAILED[@]}"
    echo "  Skipped:        ${#SKIPPED[@]}"

    if [[ ${#FIXED[@]} -gt 0 ]]; then
        echo "  Fixed:"
        for b in "${FIXED[@]}"; do echo "    + $b"; done
    fi
    if [[ ${#RETRACTED[@]} -gt 0 ]]; then
        echo "  Auto-retracted:"
        for b in "${RETRACTED[@]}"; do echo "    - $b"; done
    fi
    if [[ ${#FAILED[@]} -gt 0 ]]; then
        echo "  Failed:"
        for b in "${FAILED[@]}"; do echo "    x $b"; done
    fi

    if [ "$PUSH_RESULTS" = "true" ] && [ -n "$GH_TOKEN" ]; then
        step "Pushing results"
        cd "$REPO_DIR" && git push origin main 2>&1 || true
        cd "$EXPERT_DIR" && git push origin main 2>&1 || true
    fi

    [[ ${#FAILED[@]} -eq 0 ]]
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
