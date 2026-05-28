#!/usr/bin/env bash
# =============================================================================
# gen_slides.sh
# -------------
# Convenience wrapper around scripts/generate_latex_slides.py.
#
# Edit the variables below, then run from the project root:
#   bash scripts/gen_slides.sh
#
# Compile the output with:
#   pdflatex <output_file.tex>
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# >>>  EDIT THESE  <<<
# ---------------------------------------------------------------------------
DATASET="MedQA"   # e.g. medqa, art, copa_guess_effect, musr_murder
PROBLEM_ID=9                 # problem / sample ID from the JSON data
CHECKPOINT_A=0                # lower checkpoint (0 = raw_model)
CHECKPOINT_B=2560             # upper checkpoint
METRIC="observation_coverage"  # backtracking | branchiness | uncertainty_markers
                              # uncertainty_language | detail_coverage
                              # observation_coverage | prior

# ---------------------------------------------------------------------------
# Paths  (relative to project root – rarely need changing)
# ---------------------------------------------------------------------------
OUTPUT_DIR="results2/latex_slides"
LOG_DIR="results2/llm_logs"
CHECKPOINTS_DIR="checkpoints"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python scripts/generate_latex_slides.py \
    --dataset         "$DATASET" \
    --problem_id      "$PROBLEM_ID" \
    --checkpoint_a    "$CHECKPOINT_A" \
    --checkpoint_b    "$CHECKPOINT_B" \
    --metric          "$METRIC" \
    --output          "$OUTPUT_DIR" \
    --log_dir         "$LOG_DIR" \
    --checkpoints_dir "$CHECKPOINTS_DIR"
