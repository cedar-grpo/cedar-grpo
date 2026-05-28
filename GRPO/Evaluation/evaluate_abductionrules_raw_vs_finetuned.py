#!/usr/bin/env python3
"""
AbductionRules Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the AbductionRules abductive reasoning dataset.
The model must output the missing fact inside <answer>...</answer> tags.

Usage (examples):
    python evaluate_abductionrules_raw_vs_finetuned.py --cuda_device 0 --batch_size 2
    python evaluate_abductionrules_raw_vs_finetuned.py --checkpoint_path /path/to/checkpoint-XXXX
"""

import os
import json
import argparse
import re
from datetime import datetime

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import numpy as np
import warnings
import textwrap
import sys

warnings.filterwarnings("ignore")

# Add current directory to sys.path to ensure path_utils can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import path utilities for project-relative paths
from path_utils import get_project_root, get_datasets_dir, get_evaluation_dir, get_results_dir, get_grpo_dir
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prompts import create_abductionrules_prompt, SYSTEM_PROMPT_AbductionRules

# ============================================================================
# Configuration
# ============================================================================

# Get project root for relative paths
PROJECT_ROOT = get_project_root()

# Allow path injection from orchestrator
RAW_MODEL_PATH = os.environ.get(
    "EVAL_RAW_MODEL_PATH",
    "/home/moein_salimi/PLLMS/unsloth-Qwen2.5-3B-Instruct-unsloth-bnb-4bit",
)
TRAINING_DIR = os.environ.get(
    "EVAL_TRAINING_DIR",
    os.path.join(
        get_results_dir(),
        "dt11.10.16:42_e20_unsloth_Qwen2.5_3B_Instruct_unsloth_bnb_4bit_bnb_4bit_lr1e-05_t0.7_ε0.2_r64_b16",
    ),
)
CHECKPOINT_DIR = os.path.join(TRAINING_DIR, "checkpoint")
OUTPUT_DIR = os.environ.get(
    "EVAL_OUTPUT_DIR",
    os.path.join(get_evaluation_dir(), "abductionrules_evaluation_results"),
)
DEFAULT_VAL_SPLIT_PATH = os.environ.get(
    "ABDUCTIONRULES_VAL_SPLIT",
    "./dataset/AbductionRules.json",
)

# ============================================================================
# Helper functions: model loading
# ============================================================================


def find_best_checkpoint(training_dir: str):
    """Find best checkpoint based on validation metrics (avg_reward in val_metrics.json)."""
    print("\n📁 Finding best checkpoint...")

    val_metrics_path = os.path.join(training_dir, "val_metrics.json")
    checkpoint_dir = os.path.join(training_dir, "checkpoint")

    if not os.path.exists(checkpoint_dir):
        print(f"⚠️  No checkpoint directory found under {training_dir}")
        return None, 0.0

    if not os.path.exists(val_metrics_path):
        print("⚠️  No val_metrics.json found, using latest checkpoint instead")
        checkpoints = [
            d
            for d in os.listdir(checkpoint_dir)
            if d.startswith("checkpoint-")
            and os.path.isdir(os.path.join(checkpoint_dir, d))
        ]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            return os.path.join(checkpoint_dir, latest), 0.0
        return None, 0.0

    with open(val_metrics_path, "r") as f:
        val_metrics = json.load(f)

    best_epoch = None
    best_score = 0.0

    for epoch_str, metrics in val_metrics.items():
        if metrics.get("avg_reward", 0.0) > best_score:
            best_score = metrics["avg_reward"]
            best_epoch = float(epoch_str)

    if best_epoch is None:
        print("⚠️  No valid metrics found, using latest checkpoint")
        checkpoints = [
            d
            for d in os.listdir(checkpoint_dir)
            if d.startswith("checkpoint-")
            and os.path.isdir(os.path.join(checkpoint_dir, d))
        ]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            return os.path.join(checkpoint_dir, latest), 0.0
        return None, 0.0

    checkpoints = [
        d
        for d in os.listdir(checkpoint_dir)
        if d.startswith("checkpoint-")
        and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    if not checkpoints:
        return None, 0.0

    checkpoint_steps = [(int(cp.split("-")[1]), cp) for cp in checkpoints]
    checkpoint_steps.sort()
    max_checkpoint_step = max(checkpoint_steps)[0]

    # Training used 20 epochs; approximate mapping from epoch to step
    estimated_steps_per_epoch = max_checkpoint_step / 20.0
    target_step = int(best_epoch * estimated_steps_per_epoch)

    best_checkpoint = min(checkpoint_steps, key=lambda x: abs(x[0] - target_step))
    checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint[1])

    print(f"✅ Best checkpoint: {best_checkpoint[1]}")
    print(f"   Validation score (avg_reward): {best_score:.4f} at epoch {best_epoch:.2f}")
    return checkpoint_path, best_score


def load_raw_model(device):
    """Load the raw/base model."""
    print(f"\n🤖 Loading raw model from: {RAW_MODEL_PATH}")

    if not device.startswith("cuda"):
        device = f"cuda:{device}"

    tokenizer = AutoTokenizer.from_pretrained(RAW_MODEL_PATH, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    if "gemma" in RAW_MODEL_PATH.lower():
        model = AutoModelForCausalLM.from_pretrained(
            RAW_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
            quantization_config=bnb_config,
        )
        print("\nGemma model detected!\n")

    else:
        model = AutoModelForCausalLM.from_pretrained(
            RAW_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map={"": device},
            trust_remote_code=True,
            load_in_4bit=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print("✅ Raw model loaded successfully")

    return model, tokenizer


def load_finetuned_model(checkpoint_path, device):
    print(f"\n🎯 Loading fine-tuned model from: {checkpoint_path}")

    if not device.startswith("cuda"):
        device = f"cuda:{device}"

    base_tokenizer = AutoTokenizer.from_pretrained(RAW_MODEL_PATH, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    if "gemma" in RAW_MODEL_PATH.lower():
        base_model = AutoModelForCausalLM.from_pretrained(
            RAW_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
            quantization_config=bnb_config,
        )
        print("\nGemma model detected!\n")

    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            RAW_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map={"": device},
            trust_remote_code=True,
            load_in_4bit=True,
        )

    model = PeftModel.from_pretrained(base_model, checkpoint_path)

    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    model.eval()
    print("✅ Fine-tuned model loaded successfully")

    return model, base_tokenizer


# ============================================================================
# Helper functions: prompting, parsing, metrics
# ============================================================================




def extract_answer_text(text: str):
    """
    Extract the answer text only if it is inside <answer>...</answer> tags.
    If the model does not follow the requested format, return an empty string.
    """
    answer_match = re.search(
        r"<answer>\s*([^<]+?)\s*</answer>", text, re.IGNORECASE | re.DOTALL
    )

    if not answer_match:
        return ""

    return answer_match.group(1).strip()


def normalize_answer(text: str):
    """Normalize a predicted or ground-truth fact for comparison."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text.strip()).lower()


def parse_ground_truth_answer(answer_str: str):
    """Parse the ground-truth answer as a single normalized fact string."""
    if answer_str is None:
        return ""
    return normalize_answer(answer_str)


def compute_set_metrics(results):
    """
    Compute exact-match accuracy + macro P/R/F1 over singleton answer sets.

    Kept intentionally similar to the original script so downstream consumers
    still receive the same metric keys.
    """
    if not results:
        return {
            "exact_match_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "extraction_rate": 0.0,
        }

    exact_matches = 0
    precisions, recalls, f1s = [], [], []
    extracted = 0

    for r in results:
        gt = set(r["ground_truth"])
        pred = set(r["predicted"])

        if pred:
            extracted += 1

        if pred == gt:
            exact_matches += 1

        if not gt and not pred:
            p = r_ = f = 1.0
        elif not pred:
            p = r_ = f = 0.0
        else:
            inter = len(gt & pred)
            p = inter / len(pred) if pred else 0.0
            r_ = inter / len(gt) if gt else 0.0
            f = (2 * p * r_ / (p + r_)) if (p + r_) > 0 else 0.0

        precisions.append(p)
        recalls.append(r_)
        f1s.append(f)

    n = len(results)
    metrics = {
        "exact_match_accuracy": exact_matches / n,
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
        "extraction_rate": extracted / n,
    }
    return metrics


# ============================================================================
# Core evaluation
# ============================================================================


def load_abductionrules_split(split_path: str, max_samples: int | None = None):
    """Load AbductionRules examples from a JSON file."""
    print(f"\n📂 Loading AbductionRules split from: {split_path}")

    secondary_path = "/home/moein_salimi/users/Parsa/AbductiveReasoning/GRPO/dataset/AbductionRules.json"

    if not os.path.exists(split_path):
        if os.path.exists(secondary_path):
            print(
                f"⚠️ Warning: Primary split path not found. Using hardcoded secondary path: {secondary_path}"
            )
            split_path = secondary_path
        else:
            raise FileNotFoundError(
                f"❌ Error: Neither the primary split path ({split_path}) "
                f"nor the secondary hardcoded path ({secondary_path}) could be found."
            )

    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = [ex for ex in data if ex.get("datasetName") == "AbductionRules"]

    if not examples:
        raise RuntimeError("No examples with datasetName == 'AbductionRules' found in split.")

    if max_samples:
        examples = examples[:max_samples]

    print(f"   Loaded {len(examples)} AbductionRules examples")
    return examples


def evaluate_on_abductionrules(
    model,
    tokenizer,
    split_path: str,
    max_samples: int | None = None,
    model_name: str = "Model",
    batch_size: int = 1,
):
    """Evaluate model on AbductionRules split."""
    print(f"\n🔍 Evaluating {model_name} on AbductionRules...")
    print(f"   Split file: {split_path}")
    print(f"   Batch size: {batch_size}")

    examples = load_abductionrules_split(split_path, max_samples=max_samples)

    results = []
    num_batches = (len(examples) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"Evaluating {model_name}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(examples))
        batch = examples[start_idx:end_idx]

        prompts = []
        gt_answers = []
        batch_meta = []

        for i, ex in enumerate(batch):
            system_prompt, user_prompt = create_abductionrules_prompt(ex)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            try:
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                formatted = system_prompt + "\n\n" + user_prompt

            prompts.append(formatted)

            gt_answer = parse_ground_truth_answer(ex["answer"])
            gt_answers.append(gt_answer)

            batch_meta.append(
                {
                    "record_id": start_idx + i,
                    "context_id": ex.get("context_id"),
                    "query_id": ex.get("query_id"),
                    "context": ex["context"],
                    "query": ex["query"],
                    "ground_truth_raw": ex["answer"],
                }
            )

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            max_length=1024,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            )

        for i in range(len(prompts)):
            input_len = inputs["input_ids"][i].shape[0]
            completion = tokenizer.decode(
                outputs[i][input_len:], skip_special_tokens=True
            )

            extracted_answer = extract_answer_text(completion)
            pred_answer_norm = normalize_answer(extracted_answer)

            gt_answer_norm = gt_answers[i]
            meta = batch_meta[i]

            pred_list = [pred_answer_norm] if pred_answer_norm else []
            gt_list = [gt_answer_norm] if gt_answer_norm else []

            results.append(
                {
                    "record_id": meta["record_id"],
                    "context_id": meta["context_id"],
                    "query_id": meta["query_id"],
                    "context": meta["context"],
                    "query": meta["query"],
                    "ground_truth_text": meta["ground_truth_raw"],
                    "predicted_text": extracted_answer,
                    "ground_truth": gt_list,
                    "predicted": pred_list,
                    "completion": completion,
                    "correct": pred_answer_norm == gt_answer_norm,
                }
            )

    metrics = compute_set_metrics(results)

    print(f"\n📊 {model_name} results (AbductionRules):")
    print(
        f"   Exact-match accuracy: {metrics['exact_match_accuracy']:.4f} "
        f"({metrics['exact_match_accuracy']*100:.2f}%)"
    )
    print(f"   Precision (macro):    {metrics['precision']:.4f}")
    print(f"   Recall (macro):       {metrics['recall']:.4f}")
    print(f"   F1 (macro):           {metrics['f1']:.4f}")
    print(
        f"   Extraction rate:      {metrics['extraction_rate']:.4f} "
        f"({metrics['extraction_rate']*100:.2f}%)"
    )

    return {
        "metrics": metrics,
        "correct_count": sum(1 for r in results if r["correct"]),
        "total": len(results),
        "results": results,
    }


def evaluate_model_with_dynamic_batch(model, tokenizer, args, model_name):
    """Evaluate a model with automatic batch-size backoff to avoid CUDA OOM."""
    results = None
    batch_size = args.batch_size

    while batch_size >= 1 and results is None:
        try:
            print(f"\n🧪 Evaluating {model_name} with batch_size={batch_size}")
            results = evaluate_on_abductionrules(
                model=model,
                tokenizer=tokenizer,
                split_path=args.split_path,
                max_samples=args.max_samples,
                model_name=model_name,
                batch_size=batch_size,
            )
            print(f"✅ {model_name} evaluation succeeded with batch_size={batch_size}")
        except torch.cuda.OutOfMemoryError:
            print(f"⚠️ CUDA OutOfMemoryError at batch_size={batch_size}, halving batch size...")
            results = None
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"⚠️ RuntimeError OOM at batch_size={batch_size}, halving batch size...")
                results = None
            else:
                raise

        if results is None:
            torch.cuda.empty_cache()
            batch_size = batch_size // 2

    if results is None:
        print(f"❌ {model_name}: still out of memory even with batch_size < 1, giving up.")

    return results


# ============================================================================
# Caching helpers (per-sample raw vs finetuned)
# ============================================================================


def ensure_raw_results_cached(args):
    """
    Ensure raw AbductionRules results are cached on disk for the current configuration.
    Returns the loaded or newly computed raw_results dict.
    """
    dataset_name = "abductionrules"
    sample_tag = f"max{args.max_samples}" if args.max_samples else "all"

    raw_results_dir = os.path.join(get_grpo_dir(), args.output_path, "raw_model", dataset_name)
    os.makedirs(raw_results_dir, exist_ok=True)

    raw_results_file = os.path.join(
        raw_results_dir,
        f"raw_results_{sample_tag}.json",
    )

    if os.path.exists(raw_results_file):
        print(f"\n📂 Found cached raw model results: {raw_results_file}")
        with open(raw_results_file, "r") as f:
            raw_results = json.load(f)
        return raw_results

    print("\n🔁 No cached raw model results found for this configuration.")
    print("   Running raw model once and caching per-sample results...")

    raw_model, raw_tokenizer = load_raw_model(args.cuda_device)
    raw_results = evaluate_model_with_dynamic_batch(
        raw_model, raw_tokenizer, args, "Raw Model (cached)"
    )
    del raw_model
    torch.cuda.empty_cache()

    if raw_results is None:
        print("❌ Failed to compute raw model results; cannot cache.")
        return None

    raw_results_with_meta = {
        "model_path": RAW_MODEL_PATH,
        "dataset": dataset_name,
        "split_path": args.split_path,
        "max_samples": args.max_samples,
        **raw_results,
    }

    with open(raw_results_file, "w") as f:
        json.dump(raw_results_with_meta, f, indent=2)
    print(f"💾 Cached raw model results saved to: {raw_results_file}")

    return raw_results_with_meta


def ensure_finetuned_results_cached(args, ckpt_name):
    """
    Check if fine-tuned results already exist for this checkpoint.
    We consider them cached if both all_cases.json and disagreement_cases.json exist.
    """
    dataset_name = "abductionrules"
    ckpt_output_dir = os.path.join(get_grpo_dir(), args.output_path, ckpt_name, dataset_name)
    if (
        os.path.exists(ckpt_output_dir)
        and os.path.exists(os.path.join(ckpt_output_dir, "disagreement_cases.json"))
        and os.path.exists(os.path.join(ckpt_output_dir, "all_cases.json"))
    ):
        print(f"\n📂 Found cached fine-tuned model results: {ckpt_output_dir}")
        return True

    print("\n🔁 No cached fine-tuned model results found for this configuration.")
    return False


def evaluate_checkpoint_cases(args, checkpoint_path: str):
    """
    Given a single checkpoint, evaluate it vs cached raw results and save:
      - all_cases.json
      - disagreement_cases.json
    under: OUTPUT_DIR/<run>/<checkpoint_name>/abductionrules/
    """
    print(f"\n📁 Checkpoint path argument received: {checkpoint_path}")
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(checkpoint_path)
        print(f"   Converted to absolute path: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint path does not exist: {checkpoint_path}")
        return

    ckpt_name = os.path.basename(checkpoint_path.rstrip("/"))
    print(f"✅ Using checkpoint for per-case evaluation: {ckpt_name}")

    raw_results = ensure_raw_results_cached(args)
    if raw_results is None:
        print("❌ Cannot evaluate checkpoint without raw model results.")
        return

    if ensure_finetuned_results_cached(args, ckpt_name):
        print(f"✅ Using cached fine-tuned model results for per-case evaluation: {ckpt_name}")
        ckpt_output_dir = os.path.join(get_grpo_dir(), args.output_path, ckpt_name, "abductionrules")
        with open(os.path.join(ckpt_output_dir, "all_cases.json"), "r") as f:
            finetuned_results = json.load(f)
        return {
            "raw_results": raw_results,
            "finetuned_results": finetuned_results,
            "all_cases_file": os.path.join(ckpt_output_dir, "all_cases.json"),
            "disagreement_file": os.path.join(ckpt_output_dir, "disagreement_cases.json"),
        }

    finetuned_model, finetuned_tokenizer = load_finetuned_model(
        checkpoint_path, args.cuda_device
    )
    finetuned_results = evaluate_model_with_dynamic_batch(
        finetuned_model,
        finetuned_tokenizer,
        args,
        f"Fine-tuned Model ({ckpt_name})",
    )
    del finetuned_model
    torch.cuda.empty_cache()

    if finetuned_results is None:
        print("❌ Fine-tuned model evaluation failed; aborting.")
        return

    dataset_name = "abductionrules"
    ckpt_output_dir = os.path.join(get_grpo_dir(), args.output_path, ckpt_name, dataset_name)
    print(ckpt_output_dir)
    os.makedirs(ckpt_output_dir, exist_ok=True)

    raw_by_id = {r["record_id"]: r for r in raw_results["results"]}
    ft_by_id = {r["record_id"]: r for r in finetuned_results["results"]}

    all_cases = []
    disagreement_cases = []

    for pid, raw_r in raw_by_id.items():
        if pid not in ft_by_id:
            continue
        ft_r = ft_by_id[pid]

        case_entry = {
            "record_id": pid,
            "context_id": raw_r.get("context_id"),
            "query_id": raw_r.get("query_id"),
            "context": raw_r["context"],
            "query": raw_r["query"],
            "ground_truth_text": raw_r["ground_truth_text"],
            "ground_truth": raw_r["ground_truth"],
            "raw": {
                "predicted": raw_r["predicted"],
                "predicted_text": raw_r["predicted_text"],
                "completion": raw_r["completion"],
                "correct": raw_r["correct"],
            },
            "finetuned": {
                "predicted": ft_r["predicted"],
                "predicted_text": ft_r["predicted_text"],
                "completion": ft_r["completion"],
                "correct": ft_r["correct"],
            },
        }

        all_cases.append(case_entry)

        if raw_r["correct"] == ft_r["correct"]:
            continue

        disagreement_type = (
            "raw_correct_finetuned_wrong"
            if raw_r["correct"] and not ft_r["correct"]
            else "finetuned_correct_raw_wrong"
        )

        disagreement_cases.append(
            {
                **case_entry,
                "disagreement_type": disagreement_type,
            }
        )

    disagreement_file = os.path.join(ckpt_output_dir, "disagreement_cases.json")
    with open(disagreement_file, "w") as f:
        json.dump(disagreement_cases, f, indent=2)
    print(f"💾 Disagreement cases saved to: {disagreement_file}")

    all_cases_file = os.path.join(ckpt_output_dir, "all_cases.json")
    with open(all_cases_file, "w") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "split_path": args.split_path,
                "max_samples": args.max_samples,
                "metrics": finetuned_results["metrics"],
                "cases": all_cases,
            },
            f,
            indent=2,
        )
    print(f"💾 All cases saved to: {all_cases_file}")

    return {
        "raw_results": raw_results,
        "finetuned_results": finetuned_results,
        "all_cases_file": all_cases_file,
        "disagreement_file": disagreement_file,
    }


# ============================================================================
# Main CLI
# ============================================================================


def save_results(raw_results, finetuned_results, best_checkpoint_info, output_dir):
    """Save evaluation results to JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if raw_results:
        raw_output = {
            "model": RAW_MODEL_PATH,
            "evaluation_time": timestamp,
            "metrics": raw_results["metrics"],
            "correct_count": raw_results["correct_count"],
            "total": raw_results["total"],
            "detailed_results": raw_results["results"],
        }

        raw_file = os.path.join(output_dir, f"raw_model_results_{timestamp}.json")
        with open(raw_file, "w") as f:
            json.dump(raw_output, f, indent=2)
        print(f"\n💾 Raw model results saved to: {raw_file}")

    if finetuned_results and best_checkpoint_info:
        finetuned_output = {
            "base_model": RAW_MODEL_PATH,
            "checkpoint": best_checkpoint_info["path"],
            "validation_score": best_checkpoint_info["score"],
            "evaluation_time": timestamp,
            "metrics": finetuned_results["metrics"],
            "correct_count": finetuned_results["correct_count"],
            "total": finetuned_results["total"],
            "detailed_results": finetuned_results["results"],
        }

        finetuned_file = os.path.join(
            output_dir, f"finetuned_model_results_{timestamp}.json"
        )
        with open(finetuned_file, "w") as f:
            json.dump(finetuned_output, f, indent=2)
        print(f"💾 Fine-tuned model results saved to: {finetuned_file}")

    summary = {
        "evaluation_time": timestamp,
        "dataset": "AbductionRules",
        "split_path": best_checkpoint_info.get("split_path", "unknown") if best_checkpoint_info else "unknown",
        "num_samples": raw_results["total"] if raw_results else (finetuned_results["total"] if finetuned_results else 0),
        "raw_model": None,
        "finetuned_model": None,
        "comparison": None,
    }

    if raw_results:
        summary["raw_model"] = {
            "path": RAW_MODEL_PATH,
            "metrics": raw_results["metrics"],
            "correct_count": raw_results["correct_count"],
            "total": raw_results["total"],
        }

    if finetuned_results and best_checkpoint_info:
        summary["finetuned_model"] = {
            "base_model": RAW_MODEL_PATH,
            "checkpoint": best_checkpoint_info["path"],
            "validation_score": best_checkpoint_info["score"],
            "metrics": finetuned_results["metrics"],
            "correct_count": finetuned_results["correct_count"],
            "total": finetuned_results["total"],
        }

    if raw_results and finetuned_results:
        acc_raw = raw_results["metrics"]["exact_match_accuracy"]
        acc_ft = finetuned_results["metrics"]["exact_match_accuracy"]
        improvement = acc_ft - acc_raw
        relative_improvement = (improvement / acc_raw * 100) if acc_raw > 0 else 0

        summary["comparison"] = {
            "exact_match_improvement": improvement,
            "exact_match_relative_improvement_percent": relative_improvement,
            "overall_improved": improvement > 0,
        }

    summary_file = os.path.join(output_dir, f"comparison_summary_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Comparison summary saved to: {summary_file}")

    if raw_results and finetuned_results:
        raw_by_id = {r["record_id"]: r for r in raw_results["results"]}
        ft_by_id = {r["record_id"]: r for r in finetuned_results["results"]}

        disagreement_cases = []
        all_cases = []

        for pid, raw_r in raw_by_id.items():
            if pid not in ft_by_id:
                continue
            ft_r = ft_by_id[pid]

            case_entry = {
                "record_id": pid,
                "context_id": raw_r.get("context_id"),
                "query_id": raw_r.get("query_id"),
                "context": raw_r["context"],
                "query": raw_r["query"],
                "ground_truth_text": raw_r["ground_truth_text"],
                "ground_truth": raw_r["ground_truth"],
                "raw": {
                    "predicted": raw_r["predicted"],
                    "predicted_text": raw_r["predicted_text"],
                    "completion": raw_r["completion"],
                    "correct": raw_r["correct"],
                },
                "finetuned": {
                    "predicted": ft_r["predicted"],
                    "predicted_text": ft_r["predicted_text"],
                    "completion": ft_r["completion"],
                    "correct": ft_r["correct"],
                },
            }

            all_cases.append(case_entry)

            if raw_r["correct"] == ft_r["correct"]:
                continue

            disagreement_type = (
                "raw_correct_finetuned_wrong"
                if raw_r["correct"] and not ft_r["correct"]
                else "finetuned_correct_raw_wrong"
            )

            disagreement_cases.append(
                {
                    **case_entry,
                    "disagreement_type": disagreement_type,
                }
            )

        disagreement_file = os.path.join(output_dir, f"disagreement_cases_{timestamp}.json")
        with open(disagreement_file, "w") as f:
            json.dump(disagreement_cases, f, indent=2)
        print(f"💾 Disagreement cases saved to: {disagreement_file}")

        all_cases_file = os.path.join(output_dir, f"all_cases_{timestamp}.json")
        with open(all_cases_file, "w") as f:
            json.dump(all_cases, f, indent=2)
        print(f"💾 All cases saved to: {all_cases_file}")

    return summary


def evaluate_all_checkpoints(args):
    """Evaluate all checkpoints in a directory."""
    checkpoint_dir = args.checkpoint_dir

    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.abspath(checkpoint_dir)

    if not os.path.exists(checkpoint_dir):
        print(f"❌ Error: Checkpoint directory does not exist: {checkpoint_dir}")
        return

    print("=" * 80)
    print("🚀 ABDUCTIONRULES EVALUATION: ALL CHECKPOINTS")
    print("=" * 80)
    print(f"Checkpoint Directory: {checkpoint_dir}")
    print(f"CUDA Device: {args.cuda_device}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Split Path: {args.split_path}")
    if args.max_samples:
        print(f"Max Samples: {args.max_samples}")
    print("=" * 80)

    all_items = os.listdir(checkpoint_dir)
    checkpoint_dirs = [
        d
        for d in all_items
        if d.startswith("checkpoint-")
        and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]

    if not checkpoint_dirs:
        print(f"❌ No checkpoint-* directories found in: {checkpoint_dir}")
        return

    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))

    print(f"\n📁 Found {len(checkpoint_dirs)} checkpoints:")
    for ck in checkpoint_dirs:
        print(f"   - {ck}")

    raw_results = None
    if not args.skip_raw:
        print("\n" + "=" * 80)
        print("🤖 EVALUATING RAW MODEL (once)")
        print("=" * 80)
        raw_model, raw_tokenizer = load_raw_model(args.cuda_device)
        raw_results = evaluate_on_abductionrules(
            raw_model,
            raw_tokenizer,
            split_path=args.split_path,
            max_samples=args.max_samples,
            model_name="Raw Model",
            batch_size=args.batch_size,
        )
        del raw_model
        torch.cuda.empty_cache()
        print("\n✅ Raw model evaluation complete")
        print(
            f"   Exact-match accuracy: {raw_results['metrics']['exact_match_accuracy']:.4f}"
        )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_file = os.path.join(
        OUTPUT_DIR, f"abductionrules_all_checkpoints_summary_{timestamp}.json"
    )

    summary_data = {
        "evaluation_time": timestamp,
        "dataset": "AbductionRules",
        "split_path": args.split_path,
        "checkpoint_directory": checkpoint_dir,
        "num_checkpoints_evaluated": len(checkpoint_dirs),
        "raw_model": {
            "path": RAW_MODEL_PATH,
            "results": raw_results if raw_results else "not_evaluated",
        },
        "checkpoints": [],
    }

    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    all_checkpoint_results = []

    for i, ckpt_name in enumerate(checkpoint_dirs, 1):
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        print("\n" + "=" * 80)
        print(f"🎯 EVALUATING CHECKPOINT {i}/{len(checkpoint_dirs)}: {ckpt_name}")
        print("=" * 80)

        try:
            ft_model, ft_tokenizer = load_finetuned_model(
                ckpt_path, args.cuda_device
            )
            ft_results = evaluate_on_abductionrules(
                ft_model,
                ft_tokenizer,
                split_path=args.split_path,
                max_samples=args.max_samples,
                model_name=ckpt_name,
                batch_size=args.batch_size,
            )
            del ft_model
            torch.cuda.empty_cache()

            checkpoint_info = {
                "checkpoint_name": ckpt_name,
                "checkpoint_path": ckpt_path,
                "results": ft_results,
            }

            summary_data["checkpoints"].append(
                {
                    "name": ckpt_name,
                    "path": ckpt_path,
                    "metrics": ft_results["metrics"],
                    "improvements_vs_raw": {
                        "exact_match_delta": ft_results["metrics"]["exact_match_accuracy"] - raw_results["metrics"]["exact_match_accuracy"] if raw_results else None,
                    } if raw_results else None,
                }
            )

            with open(summary_file, "w") as f:
                json.dump(summary_data, f, indent=2)

            all_checkpoint_results.append(checkpoint_info)

            print("\n✅ Checkpoint evaluation complete")
            print(
                f"   Exact-match accuracy: {ft_results['metrics']['exact_match_accuracy']:.4f}"
            )

            if raw_results:
                delta = (
                    ft_results["metrics"]["exact_match_accuracy"]
                    - raw_results["metrics"]["exact_match_accuracy"]
                )
                print(f"   📈 Δ vs raw (exact-match): {delta:+.4f}")

        except Exception as e:
            print(f"❌ Error evaluating {ckpt_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("📊 SUMMARY: ALL CHECKPOINTS COMPARISON (AbductionRules)")
    print("=" * 80)

    if raw_results:
        print(
            f"\n🤖 RAW MODEL exact-match accuracy: "
            f"{raw_results['metrics']['exact_match_accuracy']:.4f}"
        )

    if all_checkpoint_results:
        print(
            f"\n{'Checkpoint':<20} {'ExactMatch':<12} "
            f"{'Δ vs Raw':<12}"
        )
        print("-" * 60)
        for ck in all_checkpoint_results:
            name = ck["checkpoint_name"]
            em = ck["results"]["metrics"]["exact_match_accuracy"]
            if raw_results:
                delta = em - raw_results["metrics"]["exact_match_accuracy"]
            else:
                delta = float("nan")
            print(f"{name:<20} {em:<12.4f} {delta:<12.4f}")

        best_ckpt = max(
            all_checkpoint_results,
            key=lambda x: x["results"]["metrics"]["exact_match_accuracy"],
        )
        print(f"\n🏆 BEST CHECKPOINT: {best_ckpt['checkpoint_name']}")
        print(
            f"   Exact-match accuracy: "
            f"{best_ckpt['results']['metrics']['exact_match_accuracy']:.4f}"
        )

    print(f"\n💾 All results saved to: {summary_file}")
    print("=" * 80 + "\n")


def print_comparison(summary):
    """Print formatted comparison results."""
    print("\n" + "=" * 80)
    print("📊 ABDUCTIONRULES EVALUATION: RAW vs FINE-TUNED MODEL")
    print("=" * 80)

    if summary.get("raw_model"):
        raw_metrics = summary["raw_model"]["metrics"]
        print("\n🤖 RAW MODEL:")
        print(
            f"   Exact-match accuracy: {raw_metrics['exact_match_accuracy']:.4f} "
            f"({raw_metrics['exact_match_accuracy']*100:.2f}%)"
        )
        print(f"   Precision (macro):    {raw_metrics['precision']:.4f}")
        print(f"   Recall (macro):       {raw_metrics['recall']:.4f}")
        print(f"   F1 (macro):           {raw_metrics['f1']:.4f}")

    if summary.get("finetuned_model"):
        ft_metrics = summary["finetuned_model"]["metrics"]
        print("\n🎯 FINE-TUNED MODEL:")
        print(
            f"   Checkpoint: {os.path.basename(summary['finetuned_model']['checkpoint'])}"
        )
        val_score = summary["finetuned_model"]["validation_score"]
        val_score_str = (
            f"{val_score:.4f}" if isinstance(val_score, (int, float)) else str(val_score)
        )
        print(f"   Validation Score: {val_score_str}")
        print(
            f"   Exact-match accuracy: {ft_metrics['exact_match_accuracy']:.4f} "
            f"({ft_metrics['exact_match_accuracy']*100:.2f}%)"
        )
        print(f"   Precision (macro):    {ft_metrics['precision']:.4f}")
        print(f"   Recall (macro):       {ft_metrics['recall']:.4f}")
        print(f"   F1 (macro):           {ft_metrics['f1']:.4f}")

    if summary.get("comparison"):
        print("\n📈 IMPROVEMENTS:")
        comp = summary["comparison"]
        acc_imp = comp["exact_match_improvement"]
        acc_rel = comp["exact_match_relative_improvement_percent"]

        print(
            f"   Exact-match: {acc_imp:+.4f} ({acc_imp*100:+.2f}%) | Relative: {acc_rel:+.2f}%"
        )

        print("\n" + "-" * 80)

        if comp["overall_improved"]:
            print(
                "✅ RESULT: Fine-tuning on your dataset IMPROVED performance on AbductionRules!"
            )
            print(f"   • Exact-match accuracy improved by {acc_rel:.2f}% (relative)")
        elif acc_imp < 0:
            print(
                "⚠️  RESULT: Fine-tuning on your dataset DECREASED performance on AbductionRules."
            )
            print(f"   • Exact-match accuracy decreased by {acc_rel:.2f}% (relative)")
        else:
            print(
                "➖ RESULT: Fine-tuning had NO SIGNIFICANT IMPACT on AbductionRules performance."
            )

    print("=" * 80 + "\n")


def main():
    global RAW_MODEL_PATH, OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="Evaluate raw vs fine-tuned model on AbductionRules dataset"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--cuda_device",
        type=str,
        default="0",
        help="CUDA device to use (default: 0)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for evaluation (default: 2)",
    )
    parser.add_argument(
        "--skip_raw",
        action="store_true",
        help="Skip raw model evaluation",
    )
    parser.add_argument(
        "--skip_finetuned",
        action="store_true",
        help="Skip fine-tuned model evaluation",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to specific checkpoint to evaluate",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to directory containing multiple checkpoints",
    )
    parser.add_argument(
        "--evaluate_checkpoints",
        type=int,
        default=0,
        help=(
            "If set to 1, run per-checkpoint mode: evaluate the given "
            "--checkpoint_path vs cached raw results and save all_cases/"
            "disagreement_cases under OUTPUT_DIR/run/checkpoint/abductionrules."
        ),
    )
    parser.add_argument(
        "--run",
        type=str,
        default="run",
        help="Which training run to use for the output directory structure.",
    )
    parser.add_argument(
        "--raw_path",
        type=str,
        default=None,
        help="Override the raw model path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=OUTPUT_DIR,
        help="Evaluation output root path (default: env EVAL_OUTPUT_DIR or hardcoded).",
    )
    parser.add_argument(
        "--split_path",
        type=str,
        default=DEFAULT_VAL_SPLIT_PATH,
        help="Path to the AbductionRules split JSON.",
    )

    args = parser.parse_args()

    OUTPUT_DIR = args.output_path

    if args.checkpoint_path and args.checkpoint_dir:
        print("❌ Error: Cannot use both --checkpoint_path and --checkpoint_dir")
        return

    if args.evaluate_checkpoints == 1 and args.checkpoint_dir:
        print(
            "❌ Error: --evaluate_checkpoints 1 is only supported with "
            "--checkpoint_path (single checkpoint)."
        )
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    if args.raw_path:
        RAW_MODEL_PATH = args.raw_path

    if args.evaluate_checkpoints == 1:
        if not args.checkpoint_path:
            print("❌ Error: --evaluate_checkpoints 1 requires --checkpoint_path.")
            return

        print("=" * 80)
        print("🚀 ABDUCTIONRULES PER-CHECKPOINT EVALUATION MODE")
        print("=" * 80)
        print(f"Raw Model:   {RAW_MODEL_PATH}")
        print(f"Output Dir:  {OUTPUT_DIR}")
        print(f"CUDA Device: {args.cuda_device}")
        if args.max_samples:
            print(f"Max Samples: {args.max_samples}")
        print(f"Split Path:  {args.split_path}")
        print(f"Checkpoint:  {args.checkpoint_path}")
        print("=" * 80)

        evaluate_checkpoint_cases(args, args.checkpoint_path)
        print(f"\n✅ Per-checkpoint evaluation finished for: {args.checkpoint_path}")
        print(f"   Results root directory: {OUTPUT_DIR}")
        return

    if args.checkpoint_dir:
        evaluate_all_checkpoints(args)
        return

    print("=" * 70)
    print("🚀 ABDUCTIONRULES EVALUATION: RAW vs FINE-TUNED")
    print("=" * 70)
    print(f"Raw Model:   {RAW_MODEL_PATH}")
    print(f"Training Dir:{TRAINING_DIR}")
    print(f"CUDA Device: {args.cuda_device}")
    print(f"Batch Size:  {args.batch_size}")
    print(f"Split Path:  {args.split_path}")
    if args.max_samples:
        print(f"Max Samples: {args.max_samples}")
    print("=" * 70)

    print("=" * 70)
    print("⚙️  Configuration Summary:")
    if args.max_samples:
        print(f"Max Samples: {args.max_samples}")
    if args.skip_raw:
        print("Mode: Fine-tuned model only")
    elif args.skip_finetuned:
        print("Mode: Raw model only")
    else:
        print("Mode: Both models (comparison)")
    print("=" * 70)

    if not args.skip_finetuned:
        if args.checkpoint_path:
            checkpoint_path = args.checkpoint_path
            if not os.path.isabs(checkpoint_path):
                checkpoint_path = os.path.abspath(checkpoint_path)
            if not os.path.exists(checkpoint_path):
                print(f"❌ Error: Checkpoint path does not exist: {checkpoint_path}")
                return
            print(f"✅ Using user-specified checkpoint: {os.path.basename(checkpoint_path)}")
            best_checkpoint_info = {
                "path": checkpoint_path,
                "score": "N/A (manual)",
            }
        else:
            best_path, best_score = find_best_checkpoint(TRAINING_DIR)
            if best_path is None:
                print("❌ No valid checkpoint found in training dir.")
                return
            best_checkpoint_info = {"path": best_path, "score": best_score}
    else:
        best_checkpoint_info = None

    if not args.skip_raw:
        raw_model, raw_tokenizer = load_raw_model(args.cuda_device)
        raw_results = evaluate_on_abductionrules(
            raw_model,
            raw_tokenizer,
            split_path=args.split_path,
            max_samples=args.max_samples,
            model_name="Raw Model",
            batch_size=args.batch_size,
        )
        del raw_model
        torch.cuda.empty_cache()
    else:
        raw_results = None
        print("\n⏭️  Skipping raw model evaluation")

    if not args.skip_finetuned:
        ft_model, ft_tokenizer = load_finetuned_model(
            best_checkpoint_info["path"], args.cuda_device
        )
        ft_results = evaluate_on_abductionrules(
            ft_model,
            ft_tokenizer,
            split_path=args.split_path,
            max_samples=args.max_samples,
            model_name="Fine-tuned Model",
            batch_size=args.batch_size,
        )
        del ft_model
        torch.cuda.empty_cache()
    else:
        ft_results = None
        print("\n⏭️  Skipping fine-tuned model evaluation")

    if raw_results or ft_results:
        summary = save_results(raw_results, ft_results, best_checkpoint_info, OUTPUT_DIR)
        print_comparison(summary)
    elif raw_results:
        print("\n✅ Raw model evaluation completed")
    elif ft_results:
        print("\n✅ Fine-tuned model evaluation completed")

    print(f"\n✅ All results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
