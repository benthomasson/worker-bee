#!/bin/bash
#
# batch-fix-beliefs.sh — run fix-belief.sh on multiple beliefs sequentially
#
# Usage:
#   ./batch-fix-beliefs.sh belief-1 belief-2 belief-3 [--options...]
#   ./batch-fix-beliefs.sh --file beliefs.txt [--options...]
#
# Options are passed through to fix-belief.sh (--model, --db, --repo, etc.)
# Outputs are saved to .worker-bee/batch/<timestamp>/ for later inspection.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BELIEFS=()
FIX_ARGS=()
BELIEF_FILE=""

# ── parse arguments ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --file)
            BELIEF_FILE="$2"; shift 2 ;;
        --model|--max-review-rounds|--max-verify-rounds|--db|--repo|--max-turns|--num-ctx)
            FIX_ARGS+=("$1" "$2"); shift 2 ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            BELIEFS+=("$1"); shift ;;
    esac
done

# Load beliefs from file if specified
if [[ -n "$BELIEF_FILE" ]]; then
    while IFS= read -r line; do
        line="${line%%#*}"   # strip comments
        line="${line// /}"   # strip whitespace
        [[ -n "$line" ]] && BELIEFS+=("$line")
    done < "$BELIEF_FILE"
fi

if [[ ${#BELIEFS[@]} -eq 0 ]]; then
    echo "Usage: $0 <belief-id>... [--file beliefs.txt] [--model MODEL] [--db DB] [--repo REPO]" >&2
    exit 1
fi

# ── output directory ──────────────────────────────────────────────────
TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
# Determine repo for output dir
REPO_DIR=""
for i in "${!FIX_ARGS[@]}"; do
    if [[ "${FIX_ARGS[$i]}" == "--repo" ]]; then
        REPO_DIR="${FIX_ARGS[$((i+1))]}"
    fi
done
REPO_DIR="${REPO_DIR:-$HOME/git/worker-bee}"
BATCH_DIR="${REPO_DIR}/.worker-bee/batch/${TIMESTAMP}"
mkdir -p "$BATCH_DIR"

echo "Batch fix: ${#BELIEFS[@]} belief(s)"
echo "Output dir: $BATCH_DIR"
echo "Beliefs: ${BELIEFS[*]}"
echo ""

# ── run each belief ───────────────────────────────────────────────────
FIXED=()
RETRACTED=()
FAILED=()
SKIPPED=()

for belief in "${BELIEFS[@]}"; do
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Batch: $belief ($(( ${#FIXED[@]} + ${#RETRACTED[@]} + ${#FAILED[@]} + ${#SKIPPED[@]} + 1 ))/${#BELIEFS[@]})"
    echo "════════════════════════════════════════════════════════════════"

    LOG_FILE="${BATCH_DIR}/${belief}.log"
    EXIT_CODE=0

    bash "$SCRIPT_DIR/fix-belief.sh" "$belief" "${FIX_ARGS[@]}" \
        2>&1 | tee "$LOG_FILE" || EXIT_CODE=$?

    # Classify result
    if [[ $EXIT_CODE -eq 0 ]]; then
        if grep -q "SUCCESS: Belief retracted" "$LOG_FILE"; then
            FIXED+=("$belief")
            echo "  → FIXED"
        elif grep -q "NOT VERIFIED" "$LOG_FILE"; then
            RETRACTED+=("$belief")
            echo "  → AUTO-RETRACTED"
        else
            SKIPPED+=("$belief")
            echo "  → SKIPPED (exited 0 but no fix/retract)"
        fi
    else
        FAILED+=("$belief")
        echo "  → FAILED (exit $EXIT_CODE)"
    fi
done

# ── summary ───────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Batch Summary"
echo "════════════════════════════════════════════════════════════════"
echo "  Total:         ${#BELIEFS[@]}"
echo "  Fixed:         ${#FIXED[@]}"
echo "  Auto-retracted: ${#RETRACTED[@]}"
echo "  Failed:        ${#FAILED[@]}"
echo "  Skipped:       ${#SKIPPED[@]}"
echo ""

if [[ ${#FIXED[@]} -gt 0 ]]; then
    echo "  Fixed:"
    for b in "${FIXED[@]}"; do echo "    ✓ $b"; done
fi
if [[ ${#RETRACTED[@]} -gt 0 ]]; then
    echo "  Auto-retracted:"
    for b in "${RETRACTED[@]}"; do echo "    ⊘ $b"; done
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  Failed:"
    for b in "${FAILED[@]}"; do echo "    ✗ $b"; done
fi
if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    echo "  Skipped:"
    for b in "${SKIPPED[@]}"; do echo "    - $b"; done
fi

echo ""
echo "  Logs: $BATCH_DIR/"

# Write machine-readable summary
cat > "$BATCH_DIR/summary.json" <<ENDJSON
{
  "timestamp": "$TIMESTAMP",
  "total": ${#BELIEFS[@]},
  "fixed": [$(printf '"%s",' "${FIXED[@]}" 2>/dev/null | sed 's/,$//')],
  "retracted": [$(printf '"%s",' "${RETRACTED[@]}" 2>/dev/null | sed 's/,$//')],
  "failed": [$(printf '"%s",' "${FAILED[@]}" 2>/dev/null | sed 's/,$//')],
  "skipped": [$(printf '"%s",' "${SKIPPED[@]}" 2>/dev/null | sed 's/,$//')]
}
ENDJSON

# Exit non-zero if any failed
[[ ${#FAILED[@]} -eq 0 ]]
