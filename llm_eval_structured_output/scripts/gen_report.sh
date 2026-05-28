#!/usr/bin/env bash
# =============================================================================
# gen_report.sh
# -------------
# Convenience wrapper around scripts/generate_latex_report.py.
#
# Edit the variables below, then run from the project root:
#   bash scripts/gen_report.sh
#
# Compile the output with (run twice for correct table of contents):
#   pdflatex <output_file.tex>
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# >>>  EDIT THESE  <<<
# ---------------------------------------------------------------------------
DATASET="MedQA"      # e.g. medqa, art, copa_guess_effect, goemotion
PROBLEM_ID=7         # problem / sample ID from the JSON data
CHECKPOINT_A=0       # lower checkpoint (0 = raw_model)
CHECKPOINT_B=2560    # upper checkpoint

# ---------------------------------------------------------------------------
# Paths  (relative to project root – rarely need changing)
# ---------------------------------------------------------------------------
OUTPUT_DIR="results/latex_reports"
LOG_DIR="results/llm_logs"
CHECKPOINTS_DIR="checkpoints"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python scripts/generate_latex_report.py \
    --dataset         "$DATASET" \
    --problem_id      "$PROBLEM_ID" \
    --checkpoint_a    "$CHECKPOINT_A" \
    --checkpoint_b    "$CHECKPOINT_B" \
    --output          "$OUTPUT_DIR" \
    --log_dir         "$LOG_DIR" \
    --checkpoints_dir "$CHECKPOINTS_DIR"
