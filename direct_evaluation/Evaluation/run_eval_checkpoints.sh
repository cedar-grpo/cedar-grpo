#!/bin/bash

scripts=(
    "evaluate_strategyqa_raw_vs_finetuned.py"
    "evaluate_defeasible_nli_raw_vs_finetuned.py"
    "evaluate_neulr_abductive_raw_vs_finetuned.py"
    "evaluate_copa_raw_vs_finetuned_guess_effect.py"
    "evaluate_art_raw_vs_finetuned.py"
    "evaluate_goEmotion_raw_vs_finetuned.py"
    "evaluate_medqa_raw_vs_finetuned.py"
    "evaluate_musr_murder_mystery_raw_vs_finetuned.py"
    "evaluate_musr_object_placements_raw_vs_finetuned.py"
    "evaluate_musr_team_allocation_raw_vs_finetuned.py"
)

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUTPUT_DIR="$PROJECT_ROOT/GRPO/Evaluation/14B"

ROOT_DIR="./GRPO/Evaluation/14B"

BASE_RESULTS_DIR="$PROJECT_ROOT/GRPO/results"

# RAW_MODEL_PATH="/home/moein_salimi/PLLMS/unsloth-Qwen2.5-3B-Instruct-unsloth-bnb-4bit"
RAW_MODEL_PATH="/home/moein_salimi/PLLMS/unsloth-Qwen2.5-14B-Instruct-bnb-4bit"

# RUN_NAME="dt11.18.17:40_e20_unsloth_Qwen2.5_3B_Instruct_unsloth_bnb_4bit_bnb_4bit_lr1e-05_t0.7_ε0.2_r64_b16"
RUN_NAME="dt12.03.23:22_e20_unsloth_Qwen2.5_14B_Instruct_bnb_4bit_bnb_4bit_lr1e-05_t0.7_ε0.2_r64_b4"

TRAINING_DIR="$BASE_RESULTS_DIR/Training_${RUN_NAME}"
FINAL_DIR="$BASE_RESULTS_DIR/${RUN_NAME}"

if [ -d "$FINAL_DIR/checkpoint" ]; then
    CHECKPOINT_DIR="$FINAL_DIR/checkpoint"
    TRAINING_BASE="$FINAL_DIR"
elif [ -d "$TRAINING_DIR/checkpoint" ]; then
    CHECKPOINT_DIR="$TRAINING_DIR/checkpoint"
    TRAINING_BASE="$TRAINING_DIR"
else
    echo "ERROR: Could not find checkpoint directory."
    echo "Tried:"
    echo "  $TRAINING_DIR/checkpoint"
    echo "  $FINAL_DIR/checkpoint"
    exit 1
fi

echo "Using checkpoint directory: $CHECKPOINT_DIR"
echo

COMMON_ARGS="--cuda_device 3 --evaluate_checkpoints 1"

declare -A BATCH_SIZES=(
    ["evaluate_strategyqa_raw_vs_finetuned.py"]=16
    ["evaluate_defeasible_nli_raw_vs_finetuned.py"]=32
    ["evaluate_neulr_abductive_raw_vs_finetuned.py"]=8
    ["evaluate_copa_raw_vs_finetuned_guess_effect.py"]=64
    ["evaluate_art_raw_vs_finetuned.py"]=64
    ["evaluate_goEmotion_raw_vs_finetuned.py"]=16
    ["evaluate_medqa_raw_vs_finetuned.py"]=16
    ["evaluate_musr_murder_mystery_raw_vs_finetuned.py"]=8
    ["evaluate_musr_object_placements_raw_vs_finetuned.py"]=4 # Note that each batch has 4 questions!
    ["evaluate_musr_team_allocation_raw_vs_finetuned.py"]=16
)

export TRAINING_BASE

for ckpt_name in $(ls -1 "$CHECKPOINT_DIR" | grep '^checkpoint-' | sort -t- -k2,2n); do
    ckpt="$CHECKPOINT_DIR/$ckpt_name"
    [ -d "$ckpt" ] || continue

    echo "====================================="
    echo "Using checkpoint: $ckpt"
    echo "====================================="

    for script in "${scripts[@]}"; do
        batch_size="${BATCH_SIZES[$script]:-256}"

        echo "Running $script with checkpoint $ckpt (batch_size=$batch_size) ..."
        python3 GRPO/Evaluation/"$script" \
            $COMMON_ARGS \
            --batch_size "$batch_size" \
            --checkpoint_path "$ckpt" \
            --run "$RUN_NAME" \
            --raw_path "$RAW_MODEL_PATH" \
            --output_path "$OUTPUT_DIR"

        echo "Finished $script"
        echo "-------------------------------------"
    done
    python3 GRPO/Evaluation/create_table.py \
        --root "$ROOT_DIR" \
        --out_csv "./GRPO/Evaluation//metrics_summary.xlsx" \
        --run "$RUN_NAME" \
        --base_model_name "qwen2.5-14B" \
        --base_result_dir "$BASE_RESULTS_DIR" \
        --train_data "UniADILR" \
        --raw_model_path "$RAW_MODEL_PATH"
done
