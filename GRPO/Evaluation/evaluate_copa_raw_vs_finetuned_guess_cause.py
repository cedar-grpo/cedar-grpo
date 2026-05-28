#!/usr/bin/env python3
"""
COPA Dataset Evaluation: Raw vs Fine-tuned Model

Evaluates models on the COPA (Choice of Plausible Alternatives) dataset.
Focuses on tasks where the effect is given and we need to identify the cause.

Usage:
    python evaluate_copa_raw_vs_finetuned.py [--max_samples N] [--batch_size N] [--checkpoint_dir PATH]
"""

import os
import json
import argparse
import re
from datetime import datetime
from tqdm import tqdm
import torch
from datasets import load_dataset, get_dataset_split_names
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import numpy as np
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
import textwrap
import sys
import os
warnings.filterwarnings('ignore')

# Add current directory to sys.path to ensure path_utils can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import path utilities for project-relative paths
from path_utils import get_project_root, get_datasets_dir, get_evaluation_dir, get_results_dir, get_grpo_dir
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prompts import create_copa_cause_prompt as create_copa_prompt, SYSTEM_PROMPT_COPA_CAUSE

# ============================================================================
# Configuration
# ============================================================================

# Get project root for relative paths
PROJECT_ROOT = get_project_root()

# Allow path injection from orchestrator
RAW_MODEL_PATH = os.environ.get('EVAL_RAW_MODEL_PATH', 
    "/home/moein_salimi/PLLMS/unsloth-Qwen2.5-3B-Instruct-unsloth-bnb-4bit")
TRAINING_DIR = os.environ.get('EVAL_TRAINING_DIR',
    os.path.join(get_results_dir(), "dt11.10.16:42_e20_unsloth_Qwen2.5_3B_Instruct_unsloth_bnb_4bit_bnb_4bit_lr1e-05_t0.7_ε0.2_r64_b16"))
CHECKPOINT_DIR = os.path.join(TRAINING_DIR, "checkpoint")
OUTPUT_DIR = os.environ.get('EVAL_OUTPUT_DIR',
    os.path.join(get_evaluation_dir(), "copa_evaluation_results_guess_cause"))  # Change default per script

# ============================================================================
# Helper Functions
# ============================================================================

def find_best_checkpoint(training_dir):
    """Find the best checkpoint based on validation metrics."""
    print("\n📁 Finding best checkpoint...")
    
    val_metrics_path = os.path.join(training_dir, "val_metrics.json")
    checkpoint_dir = os.path.join(training_dir, "checkpoint")
    
    if not os.path.exists(val_metrics_path):
        print(f"⚠️  No val_metrics.json found, using latest checkpoint")
        checkpoints = [d for d in os.listdir(checkpoint_dir) 
                      if d.startswith('checkpoint-') and os.path.isdir(os.path.join(checkpoint_dir, d))]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split('-')[1]))
            return os.path.join(checkpoint_dir, latest), 0.0
        return None, 0.0
    
    with open(val_metrics_path, 'r') as f:
        val_metrics = json.load(f)
    
    # Find epoch with highest avg_reward
    best_epoch = None
    best_score = 0.0
    
    for epoch_str, metrics in val_metrics.items():
        if metrics['avg_reward'] > best_score:
            best_score = metrics['avg_reward']
            best_epoch = float(epoch_str)
    
    if best_epoch is None:
        print("⚠️  No valid metrics found, using latest checkpoint")
        checkpoints = [d for d in os.listdir(checkpoint_dir) 
                      if d.startswith('checkpoint-') and os.path.isdir(os.path.join(checkpoint_dir, d))]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split('-')[1]))
            return os.path.join(checkpoint_dir, latest), 0.0
        return None, 0.0
    
    # Find closest checkpoint
    checkpoints = [d for d in os.listdir(checkpoint_dir) 
                  if d.startswith('checkpoint-') and os.path.isdir(os.path.join(checkpoint_dir, d))]
    
    if not checkpoints:
        return None, 0.0
    
    checkpoint_steps = [(int(cp.split('-')[1]), cp) for cp in checkpoints]
    checkpoint_steps.sort()
    
    max_checkpoint_step = max(checkpoint_steps)[0]
    estimated_steps_per_epoch = max_checkpoint_step / 20.0
    target_step = int(best_epoch * estimated_steps_per_epoch)
    
    best_checkpoint = min(checkpoint_steps, key=lambda x: abs(x[0] - target_step))
    checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint[1])
    
    print(f"✅ Best checkpoint: {best_checkpoint[1]}")
    print(f"   Validation score: {best_score:.4f} at epoch {best_epoch:.2f}")
    
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
        bnb_4bit_quant_type="nf4"
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
        bnb_4bit_quant_type="nf4"
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



def extract_reasoning(response):
    """Extract chain-of-thought reasoning from <think>...</think> tags, if present."""
    match = re.search(r'<think>(.*?)</think>', response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_answer(response):
    """Extract answer from model response.
    
    Returns:
        int: 0 or 1 representing the choice (0-indexed), or None if extraction fails
    """
    # First try to extract <answer>...</answer> tags
    tag_match = re.search(r'<answer>\s*([12])\s*</answer>', response, re.IGNORECASE)
    if tag_match:
        return int(tag_match.group(1)) - 1
    
    # if not successful, follow the old logic
    # Try to find <answer>X</answer> pattern
    answer_pattern = r'<answer>\s*(\d+)\s*</answer>'
    matches = re.findall(answer_pattern, response, re.IGNORECASE)
    
    if matches:
        answer = matches[-1].strip()  # Take the last match
        if answer in ['1', '2']:
            return int(answer) - 1  # Convert to 0-indexed
        
    return None

def evaluate_on_copa(model, tokenizer, max_samples=None, model_name="Model", batch_size=1, split="validation"):
    """Evaluate model on COPA dataset (cause questions only)."""
    print(f"\n🔍 Evaluating {model_name} on COPA dataset (split: {split})...")
    print(f"   Task: Identify the CAUSE given an EFFECT")
    print(f"   Batch size: {batch_size}")

    # Load COPA dataset
    print("Loading COPA dataset...")

    dataset_name = "pkavumba/balanced-copa"
    
    try:
        # Get available splits
        available_splits = get_dataset_split_names(dataset_name)
        fallback_order = ["test", "validation", "train"]

        # Find the first available split
        selected_split = None
        for split_name in fallback_order:
            if split_name in available_splits:
                selected_split = split_name
                break

        if selected_split is None:
            raise ValueError(f"None of the fallback splits {fallback_order} were found in the dataset.")
        
        dataset = load_dataset(dataset_name, split=selected_split)
        print(f"✅ Loaded {len(dataset)} samples from COPA dataset")

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\n💡 Make sure you have internet connection and the dataset is accessible.")
        return None

    print("\nFiltering dataset for samples with input tokens <= 4096...")
    original_len = len(dataset)
    
    def filter_by_token_length(sample):
        system_prompt, user_prompt = create_copa_prompt(sample)
        try:
            messages =[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            formatted_prompt = f"{system_prompt}\n\n{user_prompt}"
            
        # Get the tokenized length
        tokenized = tokenizer(formatted_prompt, truncation=False, add_special_tokens=True)
        return len(tokenized["input_ids"]) <= 4096
        
    dataset = dataset.filter(filter_by_token_length, desc="Filtering lengths")
    print(f"Filtered out {original_len - len(dataset)} samples exceeding 4096 tokens.")
    print(f"{len(dataset)} valid samples remaining.\n")
    
    # Filter for "cause" questions only (where we're given the effect)
    cause_dataset = dataset.filter(lambda x: x['question'] == 'cause')
    print(f"Filtered to {len(cause_dataset)} 'cause' questions (given effect, find cause)")
    
    if len(cause_dataset) == 0:
        print("❌ No 'cause' questions found in dataset!")
        return None
    
    if max_samples:
        cause_dataset = cause_dataset.select(range(min(max_samples, len(cause_dataset))))
        print(f"Evaluating on {len(cause_dataset)} samples (limited)")
    else:
        print(f"Evaluating on {len(cause_dataset)} samples (full filtered {split} set)")
    
    results = []
    all_true_labels = []
    all_pred_labels = []
    failed_extractions = 0
    
    # Process in batches
    num_batches = (len(cause_dataset) + batch_size - 1) // batch_size
    btime = time.time()

    for batch_idx in tqdm(range(num_batches), desc=f"Evaluating {model_name}"):
        # Get batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(cause_dataset))
        batch = cause_dataset[start_idx:end_idx]
        
        # Handle both single sample and batch cases
        if not isinstance(batch['premise'], list):
            batch = {k: [v] for k, v in batch.items()}
        
        batch_size_actual = len(batch['premise'])
        
        # Prepare prompts for batch
        formatted_prompts = []
        true_labels_batch = []
        batch_data = []
        
        for i in range(batch_size_actual):
            premise = batch['premise'][i]
            choice1 = batch['choice1'][i]
            choice2 = batch['choice2'][i]
            true_label = batch['label'][i]  # 0 or 1
            
            # Create prompt
            example = {"premise": premise, "choice1": choice1, "choice2": choice2}
            system_prompt, user_prompt = create_copa_prompt(example)
            
            # Format with chat template if available
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                # Fallback if chat template not available
                formatted_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            formatted_prompts.append(formatted_prompt)
            true_labels_batch.append(true_label)
            batch_data.append({
                'user_prompt': user_prompt,
                'premise': premise,
                'choice1': choice1,
                'choice2': choice2,
                'true_label': true_label,
                'id': start_idx + i
            })
        
        # Tokenize batch with padding
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            max_length=512
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate for batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.0,
                do_sample=False,
                # top_p=0.95,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
            )
        
        # Process each output in batch
        for i in range(len(formatted_prompts)):
            # Decode response (skip input tokens)
            input_length = inputs['input_ids'][i].shape[0]
            response = tokenizer.decode(outputs[i][input_length:], skip_special_tokens=True)
            
            # Extract answer from response
            predicted_label = extract_answer(response)
            
            # Extract reasoning
            # reasoning = extract_reasoning(response)
            reasoning = response
            
            if predicted_label is None:
                failed_extractions += 1
                predicted_label = 0  # Default to 0 if extraction fails
            
            true_label = true_labels_batch[i]
            
            all_true_labels.append(true_label)
            all_pred_labels.append(predicted_label)
            
            # Store result
            is_correct = (predicted_label == true_label)
            results.append({
                'sample_id': batch_data[i]['id'],
                'premise': batch_data[i]['premise'],
                'choice1': batch_data[i]['choice1'],
                'choice2': batch_data[i]['choice2'],
                'true_label': true_label,
                'user_prompt': batch_data[i]['user_prompt'],
                'predicted_label': predicted_label,
                'reasoning': reasoning,
                'correct': is_correct
            })
    
    etime = time.time()
    print(f"Batch processing time: {etime - btime:.2f} seconds")
    # Calculate metrics
    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)
    
    # Accuracy
    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    
    # F1, Precision, Recall (binary classification)
    f1 = f1_score(all_true_labels, all_pred_labels, average='binary', zero_division=0)
    precision = precision_score(all_true_labels, all_pred_labels, average='binary', zero_division=0)
    recall = recall_score(all_true_labels, all_pred_labels, average='binary', zero_division=0)
    
    # Count correct predictions
    correct_count = sum(1 for r in results if r['correct'])
    
    # Extraction rate
    extraction_rate = (len(results) - failed_extractions) / len(results) if results else 0.0
    
    print(f"\n📊 {model_name} Results:")
    print(f"   Accuracy:         {accuracy:.4f} ({accuracy*100:.2f}%) - {correct_count}/{len(results)} correct")
    print(f"   F1 Score:         {f1:.4f}")
    print(f"   Precision:        {precision:.4f}")
    print(f"   Recall:           {recall:.4f}")
    print(f"   Extraction Rate:  {extraction_rate:.4f} ({extraction_rate*100:.2f}%)")
    print(f"   Failed extractions: {failed_extractions}/{len(results)} ({failed_extractions/len(results)*100:.1f}%)")
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'extraction_rate': extraction_rate,
        'correct_count': correct_count,
        'total': len(results),
        'failed_extractions': failed_extractions,
        'time': etime - btime,
        'results': results
    }


def evaluate_model_with_dynamic_batch(model, tokenizer, args, model_name):
    """Evaluate a model with automatic batch-size backoff to avoid CUDA OOM."""
    results = None
    batch_size = args.batch_size
    
    while batch_size >= 1 and results is None:
        try:
            print(f"\n🧪 Evaluating {model_name} with batch_size={batch_size}")
            results = evaluate_on_copa(
                model,
                tokenizer,
                args.max_samples,
                model_name,
                batch_size,
                args.split
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

def ensure_raw_results_cached(args):
    """
    Ensure raw copa results are cached on disk for the current configuration.
    Returns the loaded or newly computed raw_results dict.
    """
    dataset_name = "copa_guess_cause"
    split = args.split
    sample_tag = f"max{args.max_samples}" if args.max_samples else "all"

    raw_results_dir = os.path.join(get_grpo_dir(), args.output_path, "raw_model", dataset_name)
    os.makedirs(raw_results_dir, exist_ok=True)
    
    raw_results_file = os.path.join(
        raw_results_dir,
        f"raw_results_train_all.json"
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
        "split": split,
        "max_samples": args.max_samples,
        **raw_results
    }
    
    with open(raw_results_file, "w") as f:
        json.dump(raw_results_with_meta, f, indent=2)
    print(f"💾 Cached raw model results saved to: {raw_results_file}")
    
    return raw_results_with_meta

def ensure_finetuned_results_cached(args, ckpt_name):
    """
    Ensure fine-tuned model results are cached on disk for the current configuration.
    Returns the loaded or newly computed fine-tuned results dict.
    """
    dataset_name = "copa_guess_cause"
    # Save in Evaluation directory instead of possibly duplicating root
    ckpt_output_dir = os.path.join(get_grpo_dir(), args.output_path, ckpt_name, dataset_name)
    if os.path.exists(ckpt_output_dir) and os.path.exists(os.path.join(ckpt_output_dir, "disagreement_cases.json")) and os.path.exists(os.path.join(ckpt_output_dir, "all_cases.json")):
        print(f"\n📂 Found cached fine-tuned model results: {ckpt_output_dir}")
        return True
    
    print("\n🔁 No cached fine-tuned model results found for this configuration.")
    return False


def evaluate_checkpoint_cases(args, checkpoint_path):
    """
    Given a single checkpoint, evaluate it vs cached raw results and save:
      - all_cases.json
      - disagreement_cases.json
    under: OUTPUT_DIR/<checkpoint_name>/copa/
    """
    print(f"\n📁 Checkpoint path argument received: {checkpoint_path}")
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(checkpoint_path)
        print(f"   Converted to absolute path: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint path does not exist: {checkpoint_path}")
        print(f"   Please check the path and try again.")
        return
    
    ckpt_name = os.path.basename(checkpoint_path.rstrip("/"))
    print(f"✅ Using checkpoint for per-case evaluation: {ckpt_name}")
    
    # Get cached (or newly computed) raw results
    raw_results = ensure_raw_results_cached(args)
    if raw_results is None:
        print("❌ Cannot evaluate checkpoint without raw model results.")
        return

    # Get cached (or newly computed) fine-tuned results
    if ensure_finetuned_results_cached(args, ckpt_name):
        print(f"✅ Using cached fine-tuned model results for per-case evaluation: {ckpt_name}")
        ckpt_output_dir = os.path.join(get_grpo_dir(), args.output_path, ckpt_name, "copa_guess_cause")
        with open(os.path.join(ckpt_output_dir, "all_cases.json"), "r") as f:
            finetuned_results = json.load(f)
        return {
            "raw_results": raw_results,
            "finetuned_results": finetuned_results,
            "all_cases_file": os.path.join(ckpt_output_dir, "all_cases.json"),
            "disagreement_file": os.path.join(ckpt_output_dir, "disagreement_cases.json")
        }
        return
    
    # Evaluate fine-tuned checkpoint
    finetuned_model, finetuned_tokenizer = load_finetuned_model(checkpoint_path, args.cuda_device)
    finetuned_results = evaluate_model_with_dynamic_batch(
        finetuned_model,
        finetuned_tokenizer,
        args,
        f"Fine-tuned Model ({ckpt_name})"
    )
    del finetuned_model
    torch.cuda.empty_cache()
    
    if finetuned_results is None:
        print("❌ Fine-tuned model evaluation failed; aborting.")
        return
    
    # Build per-case comparison
    dataset_name = "copa_guess_cause"
    # Save in Evaluation directory instead of possibly duplicating root
    ckpt_output_dir = os.path.join(get_grpo_dir(), args.output_path, ckpt_name, dataset_name)

    # print(output_dir)
    print(ckpt_output_dir)
    os.makedirs(ckpt_output_dir, exist_ok=True)
    
    raw_by_id = {idx + 1: r for idx, r in enumerate(raw_results["results"])}
    ft_by_id = {idx + 1: r for idx, r in enumerate(finetuned_results["results"])}
    
    disagreement_cases = []
    
    for pid, raw_r in raw_by_id.items():
        if pid not in ft_by_id:
            continue
        ft_r = ft_by_id[pid]
        
        case_entry = {
            "problem_id": pid,
            "user_prompt": raw_r['user_prompt'],
            "premise": raw_r["premise"],          
            "choice1": raw_r["choice1"],
            "choice2": raw_r["choice2"],  
            "true_label": raw_r["true_label"],  
            "raw": {
                "predicted_label": raw_r["predicted_label"],
                "reasoning": raw_r["reasoning"],
                "correct": raw_r["correct"]
            },
            "finetuned": {
                "predicted_label": ft_r["predicted_label"],
                "reasoning": ft_r["reasoning"],
                "correct": ft_r["correct"]
            }
        }
        
        if raw_r["correct"] == ft_r["correct"]:
            continue
        
        if raw_r["correct"] and not ft_r["correct"]:
            disagreement_type = "raw_correct_finetuned_wrong"
        else:
            disagreement_type = "finetuned_correct_raw_wrong"
        
        disagreement_cases.append({
            **case_entry,
            "disagreement_type": disagreement_type
        })
    
    disagreement_file = os.path.join(ckpt_output_dir, "disagreement_cases.json")
    with open(disagreement_file, "w") as f:
        json.dump(disagreement_cases, f, indent=2)
    print(f"💾 Disagreement cases saved to: {disagreement_file}")
    
    finetune_results_with_meta = {
        "dataset": dataset_name,
        "max_samples": args.max_samples,
        **finetuned_results
    }
    
    finetune_results_file = os.path.join(ckpt_output_dir, "all_cases.json")
    with open(finetune_results_file, "w") as f:
        json.dump(finetune_results_with_meta, f, indent=2)
    print(f"💾 finetune model results saved to: {finetune_results_file}")
    
    return {
        "raw_results": raw_results,
        "finetuned_results": finetuned_results,
        "all_cases_file": finetune_results_file,
        "disagreement_file": disagreement_file
    }



def save_results(raw_results, finetuned_results, best_checkpoint_info, output_dir):
    """Save evaluation results to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Save raw model results
    raw_output = {
        'model': RAW_MODEL_PATH,
        'evaluation_time': timestamp,
        'dataset': 'COPA (cause questions only)',
        'metrics': {
            'accuracy': raw_results['accuracy'],
            'f1': raw_results['f1'],
            'precision': raw_results['precision'],
            'recall': raw_results['recall'],
            'extraction_rate': raw_results['extraction_rate']
        },
        'correct_count': raw_results['correct_count'],
        'total': raw_results['total'],
        'failed_extractions': raw_results['failed_extractions'],
        'detailed_results': raw_results['results'][:100]  # Save first 100 for space
    }
    raw_file = os.path.join(output_dir, f"raw_model_copa_results_{timestamp}.json")
    with open(raw_file, 'w') as f:
        json.dump(raw_output, f, indent=2)
    print(f"\n💾 Raw model results saved to: {raw_file}")
    
    # Save fine-tuned model results
    finetuned_output = {
        'base_model': RAW_MODEL_PATH,
        'checkpoint': best_checkpoint_info['path'],
        'validation_score': best_checkpoint_info['score'],
        'evaluation_time': timestamp,
        'dataset': 'COPA (cause questions only)',
        'metrics': {
            'accuracy': finetuned_results['accuracy'],
            'f1': finetuned_results['f1'],
            'precision': finetuned_results['precision'],
            'recall': finetuned_results['recall'],
            'extraction_rate': finetuned_results['extraction_rate']
        },
        'correct_count': finetuned_results['correct_count'],
        'total': finetuned_results['total'],
        'failed_extractions': finetuned_results['failed_extractions'],
        'detailed_results': finetuned_results['results'][:100]  # Save first 100 for space
    }
    
    finetuned_file = os.path.join(output_dir, f"finetuned_model_copa_results_{timestamp}.json")
    with open(finetuned_file, 'w') as f:
        json.dump(finetuned_output, f, indent=2)
    print(f"💾 Fine-tuned model results saved to: {finetuned_file}")
    
    # Save comparison summary
    improvement_acc = finetuned_results['accuracy'] - raw_results['accuracy']
    improvement_f1 = finetuned_results['f1'] - raw_results['f1']
    
    summary = {
        'evaluation_time': timestamp,
        'dataset': 'COPA (cause questions only)',
        'split': 'validation',
        'num_samples': raw_results['total'],
        'raw_model': {
            'path': RAW_MODEL_PATH,
            'metrics': {
                'accuracy': raw_results['accuracy'],
                'f1': raw_results['f1'],
                'precision': raw_results['precision'],
                'recall': raw_results['recall'],
                'extraction_rate': raw_results['extraction_rate']
            }
        },
        'finetuned_model': {
            'base_model': RAW_MODEL_PATH,
            'checkpoint': best_checkpoint_info['path'],
            'validation_score': best_checkpoint_info['score'],
            'metrics': {
                'accuracy': finetuned_results['accuracy'],
                'f1': finetuned_results['f1'],
                'precision': finetuned_results['precision'],
                'recall': finetuned_results['recall'],
                'extraction_rate': finetuned_results['extraction_rate']
            }
        },
        'comparison': {
            'accuracy_improvement': improvement_acc,
            'f1_improvement': improvement_f1,
            'overall_improved': improvement_acc > 0
        }
    }
    
    summary_file = os.path.join(output_dir, f"copa_comparison_summary_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Comparison summary saved to: {summary_file}")
    
    # Save disagreement and all cases summary
    raw_by_id = {idx+1: r for idx, r in enumerate(raw_results['results'])}
    ft_by_id = {idx+1: r for idx, r in enumerate(finetuned_results['results'])}
    
    disagreement_cases, all_cases = [], []
    
    for pid, raw_r in raw_by_id.items():
        if pid not in ft_by_id:
            continue
        ft_r = ft_by_id[pid]
        
        all_cases.append({
            "problem_id": pid,
            "user_prompt": raw_r['user_prompt'],
            "premise": raw_r["premise"],          
            "choice1": raw_r["choice1"],
            "choice2": raw_r["choice2"],  
            "true_label": raw_r["true_label"],  
            "raw": {
                "predicted_label": raw_r["predicted_label"],
                "reasoning": raw_r["reasoning"],
                "correct": raw_r["correct"]
            },
            "finetuned": {
                "predicted_label": ft_r["predicted_label"],
                "reasoning": ft_r["reasoning"],
                "correct": ft_r["correct"]
            }
        })
        
        if raw_r['correct'] == ft_r['correct']:
            continue
        
        if raw_r['correct'] and not ft_r['correct']:
            disagreement_type = "raw_correct_finetuned_wrong"
        else:
            disagreement_type = "finetuned_correct_raw_wrong"
        
        disagreement_cases.append({
            "problem_id": pid,
            "user_prompt": raw_r['user_prompt'],
            "premise": raw_r["premise"],          
            "choice1": raw_r["choice1"],
            "choice2": raw_r["choice2"],
            "true_label": raw_r["true_label"],
            "raw": {
                "predicted_label": raw_r["predicted_label"],
                "reasoning": raw_r["reasoning"],
                "correct": raw_r["correct"]
            },
            "finetuned": {
                "predicted_label": ft_r["predicted_label"],
                "reasoning": ft_r["reasoning"],
                "correct": ft_r["correct"]
            },
            "disagreement_type": disagreement_type
        })
    
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
    
    # Handle relative vs absolute paths
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.abspath(checkpoint_dir)
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Error: Checkpoint directory does not exist: {checkpoint_dir}")
        return
    
    print("="*80)
    print("🚀 COPA EVALUATION: ALL CHECKPOINTS")
    print("="*80)
    print(f"Checkpoint Directory: {checkpoint_dir}")
    print(f"CUDA Device: {args.cuda_device}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Split: {args.split}")
    if args.max_samples:
        print(f"Max Samples: {args.max_samples}")
    print("="*80)
    
    # Find all checkpoint directories
    all_items = os.listdir(checkpoint_dir)
    checkpoint_dirs = [
        d for d in all_items 
        if d.startswith('checkpoint-') and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    
    if not checkpoint_dirs:
        print(f"❌ No checkpoint directories found in: {checkpoint_dir}")
        print(f"   Looking for directories named 'checkpoint-*'")
        return
    
    # Sort checkpoints by number
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[1]))
    
    print(f"\n📁 Found {len(checkpoint_dirs)} checkpoints:")
    for ckpt in checkpoint_dirs:
        print(f"   - {ckpt}")
    print()
    
    # Optionally evaluate raw model once
    raw_results = None
    if not args.skip_raw:
        print("\n" + "="*80)
        print("🤖 EVALUATING RAW MODEL (once)")
        print("="*80)
        raw_model, raw_tokenizer = load_raw_model(args.cuda_device)
        raw_results = evaluate_on_copa(raw_model, raw_tokenizer, args.max_samples, "Raw Model", args.batch_size, args.split)
        if raw_results is None:
            print("❌ Failed to evaluate raw model")
            return
        del raw_model
        torch.cuda.empty_cache()
        print(f"\n✅ Raw model evaluation complete")
        print(f"   Accuracy: {raw_results['accuracy']:.4f}")
        print(f"   F1 Score: {raw_results['f1']:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    summary_data = {
        'evaluation_time': timestamp,
        'dataset': 'COPA (cause questions only)',
        'split': args.split,
        'checkpoint_directory': checkpoint_dir,
        'num_checkpoints_evaluated': len(checkpoint_dirs),
        'raw_model': {
            'path': RAW_MODEL_PATH,
            'results': {k: v for k, v in raw_results.items() if k != 'results'} if raw_results else 'not_evaluated'
        },
        'checkpoints': []
    }
    
    summary_file = os.path.join(OUTPUT_DIR, f"copa_all_checkpoints_summary_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Evaluate each checkpoint
    all_checkpoint_results = []
    
    for i, ckpt_name in enumerate(checkpoint_dirs, 1):
        checkpoint_path = os.path.join(checkpoint_dir, ckpt_name)
        
        print("\n" + "="*80)
        print(f"🎯 EVALUATING CHECKPOINT {i}/{len(checkpoint_dirs)}: {ckpt_name}")
        print("="*80)
        
        try:
            # Load and evaluate checkpoint
            finetuned_model, finetuned_tokenizer = load_finetuned_model(checkpoint_path, args.cuda_device)
            finetuned_results = evaluate_on_copa(
                finetuned_model, finetuned_tokenizer, args.max_samples, 
                f"{ckpt_name}", args.batch_size, args.split
            )
            
            if finetuned_results is None:
                print(f"❌ Failed to evaluate {ckpt_name}")
                continue
                
            del finetuned_model
            torch.cuda.empty_cache()
            
            # Store results
            checkpoint_info = {
                'checkpoint_name': ckpt_name,
                'checkpoint_path': checkpoint_path,
                'results': finetuned_results
            }
            
            summary_data["checkpoints"].append({
                'name': checkpoint_info['checkpoint_name'],
                'path': checkpoint_info['checkpoint_path'],
                'metrics': {
                    'accuracy': checkpoint_info['results']['accuracy'],
                    'f1': checkpoint_info['results']['f1'],
                    'precision': checkpoint_info['results']['precision'],
                    'recall': checkpoint_info['results']['recall']
                }
            })
            
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
                
            all_checkpoint_results.append(checkpoint_info)
            
            print(f"\n✅ {ckpt_name} evaluation complete")
            print(f"   Accuracy: {finetuned_results['accuracy']:.4f} ({finetuned_results['accuracy']*100:.2f}%)")
            print(f"   F1 Score: {finetuned_results['f1']:.4f}")
            
            # Show improvement vs raw model if available
            if raw_results:
                acc_improvement = finetuned_results['accuracy'] - raw_results['accuracy']
                f1_improvement = finetuned_results['f1'] - raw_results['f1']
                print(f"   📈 Improvement vs Raw: Acc {acc_improvement:+.4f}, F1 {f1_improvement:+.4f}")
            
        except Exception as e:
            print(f"❌ Error evaluating {ckpt_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("📊 SUMMARY: ALL CHECKPOINTS COMPARISON (COPA)")
    print("="*80)
    
    if raw_results:
        print(f"\n🤖 RAW MODEL:")
        print(f"   Accuracy: {raw_results['accuracy']:.4f}")
        print(f"   F1 Score: {raw_results['f1']:.4f}")
    
    print(f"\n🎯 FINE-TUNED CHECKPOINTS:")
    if raw_results:
        print(f"   {'Checkpoint':<20} {'Accuracy':<15} {'F1 Score':<15} {'Acc Δ':<12} {'F1 Δ':<12}")
        print(f"   {'-'*80}")
        
        for checkpoint_info in all_checkpoint_results:
            res = checkpoint_info['results']
            acc_delta = res['accuracy'] - raw_results['accuracy']
            f1_delta = res['f1'] - raw_results['f1']
            
            print(f"   {checkpoint_info['checkpoint_name']:<20} "
                  f"{res['accuracy']:.4f}         "
                  f"{res['f1']:.4f}         "
                  f"{acc_delta:+.4f}      "
                  f"{f1_delta:+.4f}")
    
    # Find best checkpoint
    if all_checkpoint_results:
        best_ckpt = max(all_checkpoint_results, key=lambda x: x['results']['accuracy'])
        print(f"\n🏆 BEST CHECKPOINT: {best_ckpt['checkpoint_name']}")
        print(f"   Accuracy: {best_ckpt['results']['accuracy']:.4f}")
        print(f"   F1 Score: {best_ckpt['results']['f1']:.4f}")
    
    print(f"\n💾 All results saved to: {summary_file}")
    print("="*80 + "\n")

def print_comparison(summary):
    """Print formatted comparison results."""
    print("\n" + "="*80)
    print("📊 COPA EVALUATION: RAW vs FINE-TUNED MODEL")
    print("="*80)
    
    raw_metrics = summary['raw_model']['metrics']
    ft_metrics = summary['finetuned_model']['metrics']
    
    print("\n🤖 RAW MODEL:")
    print(f"   Accuracy:  {raw_metrics['accuracy']:.4f} ({raw_metrics['accuracy']*100:.2f}%)")
    print(f"   F1 Score:  {raw_metrics['f1']:.4f}")
    print(f"   Precision: {raw_metrics['precision']:.4f}")
    print(f"   Recall:    {raw_metrics['recall']:.4f}")
    
    print("\n🎯 FINE-TUNED MODEL:")
    print(f"   Checkpoint: {os.path.basename(summary['finetuned_model']['checkpoint'])}")
    print(f"   Accuracy:  {ft_metrics['accuracy']:.4f} ({ft_metrics['accuracy']*100:.2f}%)")
    print(f"   F1 Score:  {ft_metrics['f1']:.4f}")
    print(f"   Precision: {ft_metrics['precision']:.4f}")
    print(f"   Recall:    {ft_metrics['recall']:.4f}")
    
    print("\n📈 IMPROVEMENTS:")
    comp = summary['comparison']
    acc_imp = comp['accuracy_improvement']
    f1_imp = comp['f1_improvement']
    
    print(f"   Accuracy:  {acc_imp:+.4f} ({acc_imp*100:+.2f}%)")
    print(f"   F1 Score:  {f1_imp:+.4f} ({f1_imp*100:+.2f}%)")
    
    print("\n" + "-"*80)
    
    if comp['overall_improved']:
        print("✅ RESULT: Fine-tuning IMPROVED causal reasoning performance!")
        print(f"   • Accuracy improved by {acc_imp:.4f}")
    else:
        print("⚠️  RESULT: Fine-tuning did not improve causal reasoning.")
    
    print("="*80 + "\n")

def main():
    global RAW_MODEL_PATH, OUTPUT_DIR
    parser = argparse.ArgumentParser(description='Evaluate raw vs fine-tuned model on COPA dataset')
    parser.add_argument('--max_samples', type=int, default=None, 
                       help='Maximum number of samples to evaluate (default: all samples)')
    parser.add_argument('--cuda_device', type=str, default='0',
                       help='CUDA device to use (default: 0)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation (default: 4)')
    parser.add_argument('--split', type=str, default='validation', choices=['train', 'test', 'validation'],
                       help='Dataset split to use (default: validation)')
    parser.add_argument('--skip_raw', action='store_true',
                       help='Skip raw model evaluation')
    parser.add_argument('--skip_finetuned', action='store_true',
                       help='Skip fine-tuned model evaluation')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to specific checkpoint to evaluate')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Path to directory containing multiple checkpoints')
    parser.add_argument('--evaluate_checkpoints', type=int, default=0,
                       help='If set to 1, run per-checkpoint mode: '
                            'evaluate the given --checkpoint_path vs cached raw results and '
                            'save all_cases/disagreement_cases under OUTPUT_DIR/checkpoint/dataset_name.')
    parser.add_argument('--run', type=str, default="run",
                       help='Which training run to use for the output directory.')
    parser.add_argument('--raw_path', type=str, default=None,
                       help='The raw model path')
    parser.add_argument('--output_path', type=str, default=OUTPUT_DIR,
                       help='Model output path, defaults to env variable.')
    
    args = parser.parse_args()

    OUTPUT_DIR = args.output_path
    
    # Validate arguments
    if args.checkpoint_path and args.checkpoint_dir:
        print("❌ Error: Cannot use both --checkpoint_path and --checkpoint_dir")
        print("   Use --checkpoint_path for a single checkpoint")
        print("   Use --checkpoint_dir to evaluate all checkpoints in a directory")
        return
    
    if args.evaluate_checkpoints == 1 and args.checkpoint_dir:
        print("❌ Error: --evaluate_checkpoints 1 is only supported with --checkpoint_path (single checkpoint).")
        print("   Please pass a single --checkpoint_path, or omit --evaluate_checkpoints to use --checkpoint_dir.")
        return
    
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

    if args.raw_path:
        RAW_MODEL_PATH = args.raw_path
    
    # Special mode: per-checkpoint evaluation with cached raw results
    if args.evaluate_checkpoints == 1:
        if not args.checkpoint_path:
            print("❌ Error: --evaluate_checkpoints 1 requires --checkpoint_path to be set.")
            return
        
        print("="*80)
        print("🚀 Copa guess cause PER-CHECKPOINT EVALUATION MODE")
        print("="*80)
        print(f"Raw Model:     {RAW_MODEL_PATH}")
        print(f"Output Dir:    {OUTPUT_DIR}")
        print(f"CUDA Device:   {args.cuda_device}")
        print(f"Split:         {args.split}")
        if args.max_samples:
            print(f"Max Samples:   {args.max_samples}")
        print(f"Checkpoint:    {args.checkpoint_path}")
        print("="*80)
        
        evaluate_checkpoint_cases(args, args.checkpoint_path)
        print(f"\n✅ Per-checkpoint evaluation finished for: {args.checkpoint_path}")
        print(f"   Results root directory: {OUTPUT_DIR}")
        return
    
    # If checkpoint_dir is provided, evaluate all checkpoints
    if args.checkpoint_dir:
        evaluate_all_checkpoints(args)
        return
    
    print("="*70)
    print("🚀 COPA EVALUATION: RAW vs FINE-TUNED")
    print("="*70)
    print(f"Raw Model: {RAW_MODEL_PATH}")
    print(f"Training Dir: {TRAINING_DIR}")
    print(f"CUDA Device: {args.cuda_device}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Split: {args.split}")
    print(f"Task: Identify CAUSE given EFFECT")
    if args.max_samples:
        print(f"Max Samples: {args.max_samples}")
    print("="*70)
    
    # Determine which checkpoint to use
    if not args.skip_finetuned:
        if args.checkpoint_path:
            # Use user-provided checkpoint
            checkpoint_path = args.checkpoint_path
            # Debug: show what we received
            print(f"\n📁 Checkpoint path argument received: {checkpoint_path}")
            
            # Handle relative vs absolute paths
            if not os.path.isabs(checkpoint_path):
                checkpoint_path = os.path.abspath(checkpoint_path)
            
            if not os.path.exists(checkpoint_path):
                print(f"❌ Error: Checkpoint path does not exist: {checkpoint_path}")
                print(f"   Please check the path and try again.")
                return
            
            print(f"✅ Using user-specified checkpoint: {os.path.basename(checkpoint_path)}")
            best_checkpoint_info = {
                'path': checkpoint_path,
                'score': 'N/A (manually specified)'
            }
        else:
            # Auto-select best checkpoint
            best_checkpoint_path, best_score = find_best_checkpoint(TRAINING_DIR)
            if best_checkpoint_path is None:
                print("❌ No valid checkpoint found!")
                return
            best_checkpoint_info = {
                'path': best_checkpoint_path,
                'score': best_score
            }
    else:
        best_checkpoint_info = None
    
    # Evaluate raw model
    if not args.skip_raw:
        raw_model, raw_tokenizer = load_raw_model(args.cuda_device)
        raw_results = evaluate_on_copa(raw_model, raw_tokenizer, args.max_samples, "Raw Model", args.batch_size, args.split)
        if raw_results is None:
            print("❌ Failed to evaluate raw model")
            return
        del raw_model # Free memory
        torch.cuda.empty_cache()
    else:
        raw_results = None
        print("\n⏭️  Skipping raw model evaluation")
    
    # Evaluate fine-tuned model
    if not args.skip_finetuned:
        finetuned_model, finetuned_tokenizer = load_finetuned_model(best_checkpoint_info['path'], args.cuda_device)
        finetuned_results = evaluate_on_copa(finetuned_model, finetuned_tokenizer, args.max_samples, "Fine-tuned Model", args.batch_size, args.split)
        if finetuned_results is None:
            print("❌ Failed to evaluate fine-tuned model")
            return
        del finetuned_model
        torch.cuda.empty_cache()
    else:
        finetuned_results = None
        print("\n⏭️  Skipping fine-tuned model evaluation")
    
    # Save and display results
    if raw_results and finetuned_results:
        summary = save_results(raw_results, finetuned_results, best_checkpoint_info, OUTPUT_DIR)
        print_comparison(summary)
    elif raw_results:
        print("\n✅ Raw model evaluation completed")
    elif finetuned_results:
        print("\n✅ Fine-tuned model evaluation completed")
    
    print(f"\n✅ All results saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
