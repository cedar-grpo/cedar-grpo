#!/bin/bash

# ===============================
#      Evaluation Datasets 
# ===============================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

scripts=(
    # "evaluate_all.py"
    # "evaluate_neulr_deductive_raw_vs_finetuned.py"
    # "evaluate_neulr_inductive_raw_vs_finetuned.py"
    "evaluate_strategyqa_raw_vs_finetuned.py"
    "evaluate_defeasible_nli_raw_vs_finetuned.py"
    "evaluate_neulr_abductive_raw_vs_finetuned.py"
    "evaluate_musr_object_placements_raw_vs_finetuned.py"
    "evaluate_musr_murder_mystery_raw_vs_finetuned.py"
    "evaluate_musr_team_allocation_raw_vs_finetuned.py"
    "evaluate_medqa_raw_vs_finetuned.py"
    # "evaluate_gsm8k_raw_vs_finetuned.py"
    # "evaluate_aime_raw_vs_finetuned.py"
    # "evaluate_aimo_raw_vs_finetuned.py"
    "evaluate_art_raw_vs_finetuned.py"
    "evaluate_copa_raw_vs_finetuned_guess_effect.py"
    "evaluate_goEmotion_raw_vs_finetuned.py"
    "evaluate_ml_debugging_raw_vs_finetuned.py"
    # "evaluate_list_function_raw_vs_finetuned.py"
    # "evaluate_miniarc_raw_vs_finetuned.py"
    # "evaluate_pysstubs_raw_vs_finetuned.py"
)

declare -A BATCH_SIZES=(
    # ["evaluate_all.py"]=8
    ["evaluate_neulr_deductive_raw_vs_finetuned.py"]=8
    ["evaluate_neulr_inductive_raw_vs_finetuned.py"]=8
    ["evaluate_defeasible_nli_raw_vs_finetuned.py"]=8
    ["evaluate_neulr_abductive_raw_vs_finetuned.py"]=8
    ["evaluate_strategyqa_raw_vs_finetuned.py"]=8
    ["evaluate_medqa_raw_vs_finetuned.py"]=8
    ["evaluate_musr_murder_mystery_raw_vs_finetuned.py"]=8
    ["evaluate_musr_object_placements_raw_vs_finetuned.py"]=4 # Note that each batch has 4 questions!
    ["evaluate_musr_team_allocation_raw_vs_finetuned.py"]=8
    ["evaluate_gsm8k_raw_vs_finetuned.py"]=32
    ["evaluate_aime_raw_vs_finetuned.py"]=8
    ["evaluate_aimo_raw_vs_finetuned.py"]=8
    ["evaluate_art_raw_vs_finetuned.py"]=32
    ["evaluate_copa_raw_vs_finetuned_guess_effect.py"]=32
    ["evaluate_copa_raw_vs_finetuned_guess_cause.py"]=32
    ["evaluate_goEmotion_raw_vs_finetuned.py"]=8
    ["evaluate_list_function_raw_vs_finetuned.py"]=4
    ["evaluate_miniarc_raw_vs_finetuned.py"]=2
    ["evaluate_ml_debugging_raw_vs_finetuned.py"]=16
    ["evaluate_uniadilr_raw_vs_finetuned"]=4
    ["evaluate_pysstubs_raw_vs_finetuned.py"]=8
)

# CHECKPOINTS=(
#     "checkpoint-6970"
#     "checkpoint-6980"
#     "checkpoint-6990"
#     "checkpoint-7000"
# )

# ============================
#      Input Parameters
# ============================
: "${OUTPUT_DIR:?Error: OUTPUT_DIR must be set.}"
: "${ROOT_DIR:?Error: ROOT_DIR must be set.}"
: "${BASE_RESULTS_DIR:?Error: BASE_RESULTS_DIR must be set.}"
: "${RAW_MODEL_PATH:?Error: RAW_MODEL_PATH must be set.}"
: "${RUN_NAME:?Error: RUN_NAME must be set.}"
: "${CHKPT_NAME:?Error: CHKPT_NAME must be set.}"
: "${BASE_MODEL_NAME:?Error: BASE_MODEL_NAME must be set.}"
: "${TRAIN_DATA:?Error: TRAIN_DATA must be set.}"
: "${CUDA_DEVICE:?Error: CUDA_DEVICE must be set.}"
: "${EVALUATE_CHECKPOINTS:?Error: EVALUATE_CHECKPOINTS must be set.}"

COMMON_ARGS="--cuda_device ${CUDA_DEVICE} --evaluate_checkpoints ${EVALUATE_CHECKPOINTS}"


# ============================
#      Error Handling
# ============================
# ============================
#      Error Handling
# ============================
echo "Base Results Dir: $BASE_RESULTS_DIR"

TRAINING_DIR="$BASE_RESULTS_DIR/${RUN_NAME}"
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

# ====================
#      Main Loop      
# ====================
echo "Using checkpoint directory: $CHECKPOINT_DIR"
echo

export TRAINING_BASE


ckpt="$CHECKPOINT_DIR/$CHKPT_NAME"

echo "====================================="
echo "Using checkpoint: $ckpt"
echo "====================================="

for script in "${scripts[@]}"; do
    batch_size="${BATCH_SIZES[$script]:-256}"
    echo "Running $script with checkpoint $ckpt (batch_size=$batch_size) ..."
    python3 "${SCRIPT_DIR}/${script}" \
        $COMMON_ARGS \
        --batch_size "$batch_size" \
        --checkpoint_path "$ckpt" \
        --run "$RUN_NAME" \
        --split "test" \
        --raw_path "$RAW_MODEL_PATH" \
        --output_path "$OUTPUT_DIR" \
        --max_samples 500

    echo "Finished $script"
    echo "-------------------------------------"
python3 "${SCRIPT_DIR}/create_table.py" \
    --root "$ROOT_DIR" \
    --out_csv "${SCRIPT_DIR}/metrics_summary_${BASE_MODEL_NAME}.xlsx" \
    --run "$RUN_NAME" \
    --base_model_name "$BASE_MODEL_NAME" \
    --base_result_dir "$BASE_RESULTS_DIR" \
    --train_data "$TRAIN_DATA" 
done
