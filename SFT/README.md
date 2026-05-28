# SFT Training Pipeline (Abductive Reasoning)

This folder shares the GRPO pipeline assets (**model family and evaluation stack**) but replaces RL/GRPO training with **Supervised Fine-Tuning (SFT)**.

## What is included

- `train_abductive_sft.py`
  - Main SFT training pipeline.
  - Keeps the same default model setup used in GRPO (`unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit`).
  - Uses the same prompt-construction logic per dataset.
  - Reads data from `SFT/dataset/` (local copy of the splits).
  - Imports prompt helpers from `GRPO/prompts.py`.
  - Imports `lists_match` from `GRPO/Evaluation/evaluate_list_function_raw_vs_finetuned.py`.

## Folder structure

```text
SFT/
├── train_abductive_sft.py   # SFT training script
├── README.md
├── visualize.ipynb          # Training curve / log visualisation
└── dataset/                 # local dataset splits (mirrors GRPO/dataset/)
    ├── train_split.json
    ├── val_split.json
    └── ...

GRPO/                        # shared assets (read by SFT, not copied)
├── prompts.py               # prompt builders for all datasets
└── Evaluation/
    ├── evaluate_all.py
    ├── evaluate_*_raw_vs_finetuned.py
    └── ...
```

## Training method change

- **Before (GRPO):** policy optimization with rewards (`GRPOTrainer`).
- **Now (SFT):** supervised next-token training (`SFTTrainer`) on the same transformed prompt data.

In SFT mode, each sample is converted into a chat transcript:

1. system prompt
2. user prompt
3. assistant target structured as:

```
<think>
{rationale or placeholder sentence}
</think>
<answer>{ground_truth}</answer>
```

If the dataset provides a `rationale`, `explanation`, or `proof_text` field it is used verbatim inside `<think>`; otherwise a minimal placeholder is generated so the model learns the tag structure while still being trained on the correct answer.

This keeps output formatting fully aligned with your evaluation scripts.

## Default configuration (same core settings as GRPO)

- Model: `unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit`
- Quantization: 4-bit
- LoRA rank/alpha: 64 / 64
- Max sequence length: 4096
- Learning rate: `1e-5`
- LR scheduler: cosine with 2 warmup steps
- Optimizer: `adamw_torch` (β1=0.9, β2=0.99, weight decay=0.1)
- Batch size: `4` (train and eval)
- Gradient accumulation: `1`
- Max grad norm: `0.1`
- Epochs: `1` (test default)
- Eval / save every `25` steps; keep last `20` checkpoints
- `NUM_SAMPLES = 50` (currently capped for quick testing, set to `None` in the script to train on the full dataset)
- Uses `DataCollatorForCompletionOnlyLM` to only calculate loss on the assistant's generation, masking out the system and user prompts.
- Response template is derived dynamically from the tokenizer's chat template (works for Llama-3, Qwen-2.5, etc.).

All key constants are at the top of `train_abductive_sft.py`.

## How to run training

From the repository root:

```bash
cd SFT
python train_abductive_sft.py
```

Outputs are saved under:

- `results/<run_name>/` (checkpoints, metrics, and logs)

## How to run `visualize.ipynb` for SFT results

From the `SFT` directory:

```bash
jupyter notebook visualize.ipynb
```

Then in the notebook:

1. Run all cells in order.
2. Keep the default call `generate_all_runs_in_results()` so it scans `SFT/results/`.
3. Confirm your run folders contain `training_log.json` (and optionally `validation_log.json`).

Generated plots are saved to:

- `SFT/results/<run_name>/plots/`

## How to run evaluation

The script uses `GRPO/Evaluation/` directly. You can run the same evaluation workflow as before, pointing to your SFT checkpoint directory.

## Notes

- `SFT/` contains the training script, notebook, and its own `dataset/` folder.
- Prompt builders are shared with GRPO via `GRPO/prompts.py`; no duplication.
- `lists_match` is still sourced from `GRPO/Evaluation/evaluate_list_function_raw_vs_finetuned.py`.
- Any changes to `GRPO/prompts.py` or `GRPO/Evaluation/` are automatically picked up by SFT training.
- If the dataset splits change, update `SFT/dataset/` accordingly (they are a local copy, not auto-synced).
