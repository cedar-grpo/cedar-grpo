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

BASE_RESULTS_DIR="/home/msalimi/users/Nima/AbductiveReasoning/GRPO/results"

RAW_MODEL_PATH="/home/msalimi/PLLMS/unsloth-Qwen2.5-14B-Instruct-unsloth-bnb-4bit"
# RAW_MODEL_PATH="/home/moein_salimi/PLLMS/unsloth-Qwen2.5-14B-Instruct-bnb-4bit"

RUN_NAME="dt11.26.15:08_e20_unsloth_Qwen2.5_14B_Instruct_bnb_4bit_bnb_4bit_lr1e-05_t0.7_ε0.2_r64_b4"
# RUN_NAME="dt11.23.10:54_e20_unsloth_Qwen2.5_14B_Instruct_bnb_4bit_bnb_4bit_lr1e-05_t0.7_ε0.2_r64_b8"

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

COMMON_ARGS="--cuda_device 0 --evaluate_checkpoints 1"

declare -A BATCH_SIZES=(
    ["evaluate_strategyqa_raw_vs_finetuned.py"]=16
    ["evaluate_defeasible_nli_raw_vs_finetuned.py"]=32
    ["evaluate_neulr_abductive_raw_vs_finetuned.py"]=8
    ["evaluate_copa_raw_vs_finetuned_guess_effect.py"]=256
    ["evaluate_art_raw_vs_finetuned.py"]=256
    ["evaluate_goEmotion_raw_vs_finetuned.py"]=128
    ["evaluate_medqa_raw_vs_finetuned.py"]=64
    ["evaluate_musr_murder_mystery_raw_vs_finetuned.py"]=8
    ["evaluate_musr_object_placements_raw_vs_finetuned.py"]=4 # Note that each batch has 4 questions!
    ["evaluate_musr_team_allocation_raw_vs_finetuned.py"]=16
)

export TRAINING_BASE

# --- SYNCHRONIZATION CONFIGURATION ---
EXCEL_PATH="./GRPO/Evaluation/metrics_summary.xlsx"
CREDENTIALS_FILE="gen-lang-client-0687240279-843f4d194021.json"
BASE_MODEL_NAME="qwen2.5-14B"

# Set the FIXED name of your Google Sheet document here
GOOGLE_SHEET_TITLE="GRPO Final Metrics Report 14B (Master Document)"
# FOLDER_ID="YOUR_GOOGLE_DRIVE_FOLDER_ID" # Uncomment and set if needed

# --- 1. INITIAL DOWNLOAD STEP: Fetch the existing sheet content ---
echo "--- 1. Initializing Local Excel from Google Sheets ---"
python3 download_sheet.py \
    --excel_path "$EXCEL_PATH" \
    --run_name "$RUN_NAME" \
    --credentials "$CREDENTIALS_FILE" \
    --google_sheet_name "$GOOGLE_SHEET_TITLE"
    # --folder_id "$FOLDER_ID" # Uncomment if used
echo "-------------------------------------"

exit

# --- EVALUATION LOOP ---

for ckpt_name in $(ls -1 "$CHECKPOINT_DIR" | grep '^checkpoint-' | sort -t- -k2,2n); do
    ckpt="$CHECKPOINT_DIR/$ckpt_name"
    [ -d "$ckpt" ] || continue

    echo "====================================="
    echo "Using checkpoint: $ckpt"
    echo "====================================="

    for script in "${scripts[@]}"; do
        batch_size="${BATCH_SIZES[$script]:-256}"

        echo "Running $script with checkpoint $ckpt (batch_size=$batch_size) ..."
        # FIX: Added missing backslash '\' for line continuation
        python3 GRPO/Evaluation/"$script" \
            $COMMON_ARGS \
            --batch_size "$batch_size" \
            --checkpoint_path "$ckpt" \
            --run "$RUN_NAME" \
            --raw_path "$RAW_MODEL_PATH" 

        echo "Finished $script"
        echo "-------------------------------------"
        
        echo "--- Fetching Excel from Google Sheets ---"
        python3 download_sheet.py \
            --excel_path "$EXCEL_PATH" \
            --run_name "$RUN_NAME" \
            --credentials "$CREDENTIALS_FILE" \
            --google_sheet_name "$GOOGLE_SHEET_TITLE"
            # --folder_id "$FOLDER_ID" # Uncomment if used
        echo "-------------------------------------"

        # 2. UPDATE LOCAL EXCEL: Reads evaluation results and merges/updates the local Excel file
        python3 GRPO/Evaluation/create_table.py \
            --root "./GRPO/Evaluation/" \
            --out_csv "$EXCEL_PATH" \
            --run "$RUN_NAME" \
            --base_model_name "$BASE_MODEL_NAME" \
            --raw_model_path "$RAW_MODEL_PATH"

        # 3. UPLOAD/SYNC: Uploads the updated local Excel file back to the specified Google Sheet
        echo "--- Syncing updated Excel to Google Sheets ---"
        python3 upload_sheet.py \
            --excel_path "$EXCEL_PATH" \
            --run_name "$RUN_NAME" \
            --credentials "$CREDENTIALS_FILE" \
            --google_sheet_name "$GOOGLE_SHEET_TITLE"
            # --folder_id "$FOLDER_ID" # Uncomment if used
            
        echo "-------------------------------------"
    done
done