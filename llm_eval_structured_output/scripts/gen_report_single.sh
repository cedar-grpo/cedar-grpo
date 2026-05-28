#!/usr/bin/env bash
# =============================================================================
# gen_report_single.sh
# -------------
# Generates single-checkpoint reports for multiple datasets and problem IDs.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# >>>  EDIT THESE  <<<
# ---------------------------------------------------------------------------
DATASETS=("defeasible_nli" "musr_murder" "musr_object" "musr_team" "neulr_abductive" "strategyqa")
PROBLEM_IDS=("0" "1" "2")
CHECKPOINT=2560 # Single checkpoint

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR="results/latex_reports"
LOG_DIR="results/llm_logs"
CHECKPOINTS_DIR="checkpoints"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

for ds in "${DATASETS[@]}"; do
    for pid in "${PROBLEM_IDS[@]}"; do
        echo "Generating report for: Dataset=$ds | Problem ID=$pid | Checkpoint=$CHECKPOINT"
        python scripts/generate_latex_report.py \
            --dataset         "$ds" \
            --problem_id      "$pid" \
            --checkpoint_a    "$CHECKPOINT" \
            --output          "$OUTPUT_DIR" \
            --log_dir         "$LOG_DIR" \
            --checkpoints_dir "$CHECKPOINTS_DIR" || echo "Failed to generate report for $ds $pid. Skipping."
    done
    echo "--------------------------------------------------------"
done
