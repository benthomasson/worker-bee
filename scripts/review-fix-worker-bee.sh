#!/bin/bash
#
# Usage: ./review-fix-worker-bee.sh <belief-id> [review-dir]
#
# Runs reasonsforge code review on the current branch, then feeds
# any BLOCK/CONCERN findings back to worker-bee edit for remediation.
#
# If review-dir is omitted, runs a fresh review first.
# If review-dir is provided, skips the review and uses that report.

source "$(dirname "$0")/load-env.sh"

BELIEF_ID="${1:?Usage: $0 <belief-id> [review-dir]}"
REVIEW_DIR="${2:-}"
MODEL="openai:gpt-5.6-luna"
DB=~/git/worker-bee-expert/reasons.db
BRANCH=$(cd ~/git/worker-bee && git branch --show-current)

cd ~/git/worker-bee

# Step 1: Run code review if no review dir provided
if [ -z "$REVIEW_DIR" ]; then
    echo "=== Running code review ==="
    reasonsforge code review \
        --reasons-db "$DB" \
        --repo . \
        --model "$MODEL" \
        --base main \
        --branch "$BRANCH" \
        2>&1 | tee /tmp/review-output.txt

    # Find the latest review directory
    REVIEW_DIR=$(ls -td reviews/*/2* 2>/dev/null | head -1)
    if [ -z "$REVIEW_DIR" ]; then
        echo "No review directory found"
        exit 1
    fi
fi

REPORT="$REVIEW_DIR/report.md"
if [ ! -f "$REPORT" ]; then
    echo "Report not found: $REPORT"
    exit 1
fi

# Step 2: Check if there are actionable findings
if grep -q '\[PASS\]' "$REPORT" && ! grep -q 'BLOCK\|CONCERN' "$REPORT"; then
    echo "=== Review passed, no fixes needed ==="
    exit 0
fi

echo "=== Review found issues, running worker-bee edit ==="

# Step 3: Build the edit task from the review report
TASK=$(cat <<TASK_EOF
Address the following code review findings for the fix of belief '$BELIEF_ID'.

The review report is below. Fix all BLOCK and CONCERN items.
Do not introduce new functionality beyond what is needed to address the findings.

--- REVIEW REPORT ---
$(cat "$REPORT")
--- END REVIEW REPORT ---
TASK_EOF
)

# Step 4: Run worker-bee edit with the review findings as the task
uv run worker-bee edit "$TASK" \
    --db "$DB" \
    --model "$MODEL" \
    --max-turns 1000
