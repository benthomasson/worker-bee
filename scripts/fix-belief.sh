#!/bin/bash
#
# fix-belief.sh — closed-loop pipeline to fix a gated belief
#
# Usage: ./fix-belief.sh <belief-id> [--model MODEL] [--max-review-rounds N] [--max-verify-rounds N]
#
# Pipeline:
#   1. Verify the belief is real (worker-bee verify)
#   2. Create a branch and fix it (worker-bee fix)
#   3. Commit, review, fix review findings (review-fix loop)
#   4. Verify the fix resolves the belief (reasonsforge code verify-fix)
#   5. If PARTIAL, extract the gap and run worker-bee edit, then repeat from 3
#   6. If FIXED, done — belief retracted and dependents propagated
#
set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────
BELIEF_ID=""
MODEL="openai:gpt-5.6-luna"
MAX_REVIEW_ROUNDS=3
MAX_VERIFY_ROUNDS=3
DB=~/git/worker-bee-expert/reasons.db
REPO=~/git/worker-bee
MAX_TURNS=1000

# ── parse arguments ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --max-review-rounds) MAX_REVIEW_ROUNDS="$2"; shift 2 ;;
        --max-verify-rounds) MAX_VERIFY_ROUNDS="$2"; shift 2 ;;
        --db) DB="$2"; shift 2 ;;
        --repo) REPO="$2"; shift 2 ;;
        --max-turns) MAX_TURNS="$2"; shift 2 ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            if [[ -z "$BELIEF_ID" ]]; then
                BELIEF_ID="$1"
            else
                echo "Unexpected argument: $1" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$BELIEF_ID" ]]; then
    echo "Usage: $0 <belief-id> [--model MODEL] [--max-review-rounds N] [--max-verify-rounds N]" >&2
    exit 1
fi

# ── secrets ───────────────────────────────────────────────────────────
source "$(dirname "$0")/load-env.sh"

# ── helpers ───────────────────────────────────────────────────────────
log() { echo ""; echo "══════════════════════════════════════════════════"; echo "  $*"; echo "══════════════════════════════════════════════════"; }
step() { echo "── $* ──"; }
fail() { echo "FAILED: $*" >&2; exit 1; }

BRANCH="fix-${BELIEF_ID}"
BASE_COMMIT=""

has_changes() {
    cd "$REPO"
    # staged or unstaged modifications to tracked files, or new untracked files
    [[ -n "$(git diff --name-only)" ]] || [[ -n "$(git diff --cached --name-only)" ]] || [[ -n "$(git ls-files --others --exclude-standard)" ]]
}

commit_changes() {
    local msg="$1"
    cd "$REPO"
    # Only stage source and test files — skip generated artifacts (reviews/, costs/, etc.)
    git add worker_bee/ tests/ .gitignore pyproject.toml 2>/dev/null || true
    git diff --cached --quiet && return 0
    git commit -m "$msg" || true
}

latest_review_dir() {
    ls -td "$REPO/reviews/$BRANCH"/2* 2>/dev/null | head -1
}

# ── step 1: verify ───────────────────────────────────────────────────
log "Step 1: Verify belief is real"

VERIFY_OUTPUT=$(cd "$REPO" && uv run worker-bee verify "$DB" "$BELIEF_ID" \
    --project-dir "$REPO" \
    --model "$MODEL" 2>&1) || true
echo "$VERIFY_OUTPUT"

if echo "$VERIFY_OUTPUT" | grep -q "NOT VERIFIED"; then
    CONFIDENCE=$(echo "$VERIFY_OUTPUT" | grep -o "confidence: [a-z]*" | head -1 | cut -d' ' -f2)
    echo ""
    echo "Belief is NOT VERIFIED (confidence: ${CONFIDENCE:-unknown})."

    if [[ "$CONFIDENCE" == "high" ]]; then
        step "Auto-retracting belief (high confidence)"
        EVIDENCE=$(echo "$VERIFY_OUTPUT" | grep "Evidence:" | head -1 | sed 's/.*Evidence: //')
        reasonsforge retract "$BELIEF_ID" \
            --reason "Auto-retracted: NOT VERIFIED (high). ${EVIDENCE}" \
            2>&1 || true
    else
        echo "Confidence is not high — skipping auto-retraction."
        echo "To retract manually: reasonsforge retract $BELIEF_ID"
    fi

    exit 0
fi

if ! echo "$VERIFY_OUTPUT" | grep -q "VERIFIED"; then
    fail "Could not determine verification result"
fi

step "Belief verified as real — proceeding to fix"

# ── step 2: create branch and fix ─────────────────────────────────────
log "Step 2: Create branch and initial fix"

cd "$REPO"
# stash any existing work
if has_changes; then
    git stash push -u -m "fix-belief: stash before $BELIEF_ID"
fi

# create or switch to fix branch
CURRENT=$(git branch --show-current)
if [[ "$CURRENT" != "$BRANCH" ]]; then
    git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
fi

BASE_COMMIT=$(git rev-parse HEAD)

step "Running worker-bee fix"
cd "$REPO"
uv run worker-bee fix "$DB" "$BELIEF_ID" \
    --project-dir "$REPO" \
    --model "$MODEL" \
    --max-turns "$MAX_TURNS" 2>&1 || true

if ! has_changes; then
    fail "worker-bee fix made no changes"
fi

commit_changes "fix $BELIEF_ID"

# ── step 3+4: review-fix loop then verify-fix, repeat ─────────────────
for verify_round in $(seq 1 "$MAX_VERIFY_ROUNDS"); do
    log "Verify round $verify_round/$MAX_VERIFY_ROUNDS"

    # ── review-fix loop ──
    for review_round in $(seq 1 "$MAX_REVIEW_ROUNDS"); do
        step "Review round $review_round/$MAX_REVIEW_ROUNDS"

        cd "$REPO"
        REVIEW_EXIT=0
        reasonsforge code review \
            --reasons-db "$DB" \
            --repo "$REPO" \
            --model "$MODEL" \
            --base main \
            --branch "$BRANCH" \
            --fixing-belief "$BELIEF_ID" \
            2>&1 || REVIEW_EXIT=$?

        # Exit codes: 0=PASS, 1=CONCERN, 2=BLOCK
        case "$REVIEW_EXIT" in
            0) GATE="PASS" ;;
            1) GATE="CONCERN" ;;
            2) GATE="BLOCK" ;;
            *) GATE="ERROR($REVIEW_EXIT)" ;;
        esac

        echo ""
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

        REVIEW_DIR=$(latest_review_dir)
        REPORT="${REVIEW_DIR:+$REVIEW_DIR/report.md}"
        if [[ -z "$REPORT" ]] || [[ ! -f "$REPORT" ]]; then
            echo "WARNING: No review report found for edit task, skipping"
            break
        fi

        TASK="Address the following code review findings for the fix of belief '$BELIEF_ID'.

The review report is below. Fix all BLOCK and CONCERN items.
Do not introduce new functionality beyond what is needed to address the findings.

--- REVIEW REPORT ---
$(cat "$REPORT")
--- END REVIEW REPORT ---"

        cd "$REPO"
        uv run worker-bee edit "$TASK" \
            --db "$DB" \
            --model "$MODEL" \
            --max-turns "$MAX_TURNS" 2>&1 || true

        if has_changes; then
            commit_changes "review fixes #$review_round for $BELIEF_ID"
        else
            echo "No changes from review-fix, proceeding"
            break
        fi
    done

    # commit any remaining uncommitted files (e.g. new test files)
    if has_changes; then
        commit_changes "additional files for $BELIEF_ID"
    fi

    # ── verify-fix ──
    step "Running verify-fix"

    HEAD_COMMIT=$(cd "$REPO" && git rev-parse HEAD)
    VERIFY_JSON="/tmp/verify-fix-${BELIEF_ID}-${verify_round}.json"
    reasonsforge --db "$DB" code verify-fix "$BELIEF_ID" \
        --commit "$BASE_COMMIT..$HEAD_COMMIT" \
        --repo "$REPO" \
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
        exit 0
    fi

    if [[ "$VERDICT" == "PARTIAL" ]]; then
        if [[ "$verify_round" -eq "$MAX_VERIFY_ROUNDS" ]]; then
            log "PARTIAL after $MAX_VERIFY_ROUNDS verify rounds — stopping"
            python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1])),indent=2))" "$VERIFY_JSON" || true
            exit 1
        fi

        step "Fix is PARTIAL — reading gap from structured output"

        GAP_DESC=$(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
gap = d.get('gap_description', '')
reason = d.get('reason', '')
print(gap or reason)
" "$VERIFY_JSON")

        # Build --gap-files arguments
        GAP_FILES_ARGS=()
        while IFS= read -r gf; do
            [[ -n "$gf" ]] && GAP_FILES_ARGS+=(--gap-files "$gf")
        done < <(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
for f in d.get('affected_files', []):
    print(f)
" "$VERIFY_JSON")

        cd "$REPO"
        uv run worker-bee fix "$DB" "$BELIEF_ID" \
            --project-dir "$REPO" \
            --model "$MODEL" \
            --max-turns "$MAX_TURNS" \
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
    log "verify-fix did not return FIXED or PARTIAL"
    echo "$VERIFY_FIX_OUTPUT"
    exit 1
done

log "Exhausted all verify rounds without full fix"
exit 1
