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

raw_paths=(
    "$HOME/PLLMS/unsloth-Qwen2.5-14B-Instruct-bnb-4bit"
    "$HOME/PLLMS/unsloth-Qwen2.5-3B-Instruct-unsloth-bnb-4bit"
)

# RUN_PREFIX="Dec1"

OUTPUT_PATH="$HOME/users/Danial/AbductiveReasoning/GRPO/Evaluation/ablation/raw_model"

COMMON_ARGS="--cuda_device 3 --skip_finetuned"

declare -A BATCH_SIZES=(
    ["evaluate_strategyqa_raw_vs_finetuned.py"]=16
    ["evaluate_defeasible_nli_raw_vs_finetuned.py"]=32
    ["evaluate_neulr_abductive_raw_vs_finetuned.py"]=8
    ["evaluate_copa_raw_vs_finetuned_guess_effect.py"]=128
    ["evaluate_art_raw_vs_finetuned.py"]=64
    ["evaluate_goEmotion_raw_vs_finetuned.py"]=16
    ["evaluate_medqa_raw_vs_finetuned.py"]=16
    ["evaluate_musr_murder_mystery_raw_vs_finetuned.py"]=8
    ["evaluate_musr_object_placements_raw_vs_finetuned.py"]=4 # Note that each batch has 4 questions!
    ["evaluate_musr_team_allocation_raw_vs_finetuned.py"]=16
)

export TRAINING_BASE

script_name_color='\033[0;32m' # Green
model_info_color='\033[0;33m'  # Yellow
batch_info_color='\033[0;34m'  # Blue
no_color='\033[0m'             # Reset color

for script in "${scripts[@]}"; do
    for raw_path in "${raw_paths[@]}"; do
        batch_size="${BATCH_SIZES[$script]:-256}"
        model_name="${raw_path##*/}"   # Extract the model name
        RUN_NAME="${model_name}"
        echo -e "Running ${script_name_color}$script${no_color} with on model ${model_info_color}$model_name ${batch_info_color}(batch_size=$batch_size)${no_color}..."
        
        python3 GRPO/Evaluation/"$script" \
            $COMMON_ARGS \
            --batch_size "$batch_size" \
            --run "$RUN_NAME" \
            --output_path "$OUTPUT_PATH" \
            --raw_path "$raw_path"
    done
done