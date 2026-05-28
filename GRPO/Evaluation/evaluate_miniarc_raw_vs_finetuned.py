#!/usr/bin/env python
# coding: utf-8

# # MiniARC Dataset Evaluation: Raw vs Fine-tuned Model
# 
# This notebook evaluates models on the MiniARC task, where the model must infer a grid transformation rule from a few training examples and apply it to a test case. ACR answers are 2D grids (lists of lists of integers).
# 
# ## 1. Setup and Environment

# In[ ]:


import os
import sys
import json
import argparse
import re
import logging
import multiprocessing
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
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
import sys, os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from prompts import (create_acr_prompt, SYSTEM_PROMPT_ACR,
    SYSTEM_PROMPT_ACR_V1, SYSTEM_PROMPT_ACR_V2, SYSTEM_PROMPT_ACR_V3,
    grid_to_string, grid_to_string_simple)


# Verify installation
# --- LOCAL SETUP ---
# print(f"\n🔥 PyTorch version: {torch.__version__}")
# print(f"🎮 CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"🎮 CUDA version: {torch.version.cuda}")
    


# In[ ]:


# def set_reproducibility(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     # For deterministic behavior in cuDNN
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     transformers_set_seed(seed)
#     print(f"✅ Random seed set to: {seed}")

# set_reproducibility(42)


# ## 2. Configuration
# Defines global constants, model paths, and directory structures for training data, checkpoints, and evaluation outputs.

# In[ ]:


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
    os.path.join(get_evaluation_dir(), "acr_evaluation_results"))  # Change default per script

ACR_DATASET_PATH = "/home/moein_salimi/users/Parsa/AbductiveReasoning/GRPO/dataset/miniarc.jsonl" 

# ## 3. Model Loading and Checkpoint Utilities
# Contains functions to discover the best checkpoint based on validation metrics and load models (raw or fine-tuned) using 4-bit quantization for efficient inference.

# In[ ]:


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

    # Assuming 20 epochs based on original script's logic, estimate steps/epoch
    max_checkpoint_step = max(checkpoint_steps)[0] if checkpoint_steps else 0
    estimated_steps_per_epoch = max_checkpoint_step / 20.0 if max_checkpoint_step > 0 else 1 
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

# ## 4. Data Processing and Prompt Engineering
# Handles grid serialization and defines various system prompt templates to guide the model in inferring transformation rules and generating Python code.

# In[ ]:


# ! diffrence



def extract_reasoning(response):
    """Extract chain-of-thought reasoning from <think>...</think> tags."""
    match = re.search(r'<think>(.*?)</think>', response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_code(response):
    """
    Extracts the Python code string from the <answer> tags.
    Handles potential markdown formatting and conversational filler.
    """
    if not response:
        return None

    # Look for content between <answer> tags
    tag_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.IGNORECASE | re.DOTALL)
    if not tag_match:
        return None

    code = tag_match.group(1).strip()

    # Clean up markdown code blocks if the model included them (e.g., ```python ... ```)
    code = re.sub(r'^```python\s*', '', code)
    code = re.sub(r'^```\s*', '', code)
    code = re.sub(r'\s*```$', '', code)

    return code

def _execute_worker(code_str, input_grid, queue):
    """Worker function for multiprocessing execution to handle timeouts."""
    local_namespace = {}
    try:
        # Execute the code to define the 'transform' function
        exec(code_str, {"__builtins__": __builtins__}, local_namespace)

        if 'transform' not in local_namespace:
            queue.put((None, "Error: Function 'transform' not found in generated code."))
            return

        result = local_namespace['transform'](input_grid)

        # Validate the output type (must be a list of lists of ints for MiniARC)
        if not isinstance(result, list):
            queue.put((None, f"Error: Function returned {type(result).__name__}, expected a list."))
            return

        if not all(isinstance(row, list) and all(isinstance(val, int) for val in row) for row in result):
            queue.put((None, "Error: Function returned a grid containing non-integer values or invalid structure."))
            return

        queue.put((result, None))

    except Exception as e:
        queue.put((None, f"Execution Error: {str(e)}"))

def execute_transform(code_str, input_grid, timeout=5):
    """
    Executes the model-generated code and runs the 'transform' function with a timeout.

    Args:
        code_str: The Python code string to execute.
        input_grid: The 2D list (grid) to pass to the transform function.
        timeout: Maximum time in seconds to allow for execution.

    Returns:
        tuple: (predicted_grid, error_message)
    """
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_execute_worker, args=(code_str, input_grid, queue))
    p.start()

    # Wait for the process to finish or timeout
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return None, f"Timeout Error: Execution took longer than {timeout} seconds."

    if not queue.empty():
        return queue.get()

    return None, "Execution Error: Unknown error during subprocess execution."

def grids_match(grid1, grid2):
    """Checks for exact match between two 2D lists of integers."""
    if not (isinstance(grid1, list) and isinstance(grid2, list)):
        return False
    if len(grid1) != len(grid2):
        return False

    for row1, row2 in zip(grid1, grid2):
        if not (isinstance(row1, list) and isinstance(row2, list)):
            return False
        if len(row1) != len(row2):
            return False
        if row1 != row2:
            return False

    return True


# ## 6. Batch Evaluation with OOM Handling
# The core evaluation loop that processes the dataset in batches, with built-in logic to dynamically reduce batch sizes if CUDA Out-of-Memory errors occur.

# In[ ]:


def evaluate_on_acr(model, tokenizer, max_samples=None, model_name="Model", batch_size=1, split='train'):
    """
    Evaluate model on ACR dataset using DYNAMIC CODE EXECUTION.
    Includes automatic batch-size reduction on CUDA Out-of-Memory errors.
    """
    start_time = time.time()
    print(f"\n🔍 Evaluating {model_name} on ACR dataset...")
    print(f"   Initial Batch size: {batch_size} | Split: {split}")

    # Load ACR dataset
    try:
        dataset = load_dataset("json", data_files=ACR_DATASET_PATH)["train"]
    except Exception as e:
        print(f"❌ Error loading dataset from {ACR_DATASET_PATH}: {e}")
        return {
            'accuracy': 0.0, 'correct': 0, 'total': 0, 
            'failed_extractions': 0, 'extraction_rate': 0.0, 'results': []
        }

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"📊 Evaluating on {len(dataset)} samples (limited)")
    else:
        print(f"📊 Evaluating on {len(dataset)} samples (full dataset)")

    results = []
    correct = 0
    total = 0 # This will track VALID samples
    failed_extractions = 0
    execution_errors = 0
    timeout_errors = 0
    invalid_inputs = 0

    current_idx = 0
    current_batch_size = batch_size
    pbar = tqdm(total=len(dataset), desc=f"Evaluating {model_name}")

    while current_idx < len(dataset):
        try:
            actual_batch_size = min(current_batch_size, len(dataset) - current_idx)
            batch = dataset[current_idx : current_idx + actual_batch_size]

            # Normalize batch format
            if not isinstance(batch['train'], list) and not isinstance(batch['train'], np.ndarray):
                 batch = {k: [v] for k, v in batch.items()}

            batch_size_actual = len(batch['train'])
            formatted_prompts = []
            test_inputs = []
            true_answers = []
            batch_ids = []
            train_examples_list = []

            for i in range(batch_size_actual):
                # ACR task has multiple test cases per entry
                sample_test_cases = batch['test'][i]

                sample_inputs = []
                sample_outputs = []
                valid_sample = True

                for test_case in sample_test_cases:
                    test_input = test_case['input']
                    true_output = test_case['output']

                    # Basic validation for MiniARC grids
                    if not (isinstance(test_input, list) and isinstance(true_output, list)):
                        valid_sample = False
                        break

                    sample_inputs.append(test_input)
                    sample_outputs.append(true_output)

                # Validate parsed inputs
                if not valid_sample:
                    invalid_inputs += 1
                    print(f"⚠️ Warning: Failed to parse one or more test cases at index {current_idx + i}. Skipping.")
                    results.append({
                        'problem_id': batch['idx'][i] if 'idx' in batch else current_idx + i,
                        'test_input': None,
                        'true_answer': None,
                        'predicted_answer': None,
                        'reasoning': None,
                        'code': None,
                        'error': "Invalid input/output format in dataset",
                        'correct': False,
                        'invalid': True
                    })
                    continue

                test_inputs.append(sample_inputs)
                true_answers.append(sample_outputs)
                batch_ids.append(batch['idx'][i] if 'idx' in batch else current_idx + i)
                train_examples_list.append(batch['train'][i])

                # Create prompt
                example = {'train': batch['train'][i], 'test': batch['test'][i]}
                system_prompt, user_prompt = create_acr_prompt(example)

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
                    formatted_prompt = f"{system_prompt}\n\n{user_prompt}"

                formatted_prompts.append(formatted_prompt)

            if not formatted_prompts:
                current_idx += batch_size_actual
                pbar.update(batch_size_actual)
                continue

            # Tokenize batch
            inputs = tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                max_length=4096
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
                )

            # Process results
            for i in range(len(formatted_prompts)):
                input_length = inputs['input_ids'][i].shape[0]
                response = tokenizer.decode(outputs[i][input_length:], skip_special_tokens=True)

                reasoning = extract_reasoning(response)
                generated_code = extract_code(response)

                is_correct = False
                error_msg = None
                predicted_grids = []

                if generated_code:
                    sample_passed = True
                    sample_errors = []

                    # Run against ALL test cases for this problem
                    for t_idx in range(len(test_inputs[i])):
                        pred, err = execute_transform(generated_code, test_inputs[i][t_idx])
                        predicted_grids.append(pred)
                        if err:
                            sample_errors.append(err)
                            sample_passed = False
                        elif not grids_match(pred, true_answers[i][t_idx]):
                            sample_passed = False

                    is_correct = sample_passed

                    if sample_errors:
                        # Check if any error was a timeout
                        if any("Timeout Error" in err for err in sample_errors):
                            timeout_errors += 1
                            logging.warning(f"⚠️ Timeout occurred for Sample ID: {batch_ids[i]}")
                            error_msg = f"Timeout Error in {len([e for e in sample_errors if 'Timeout' in e])}/{len(test_inputs[i])} tests."
                        else:
                            execution_errors += 1
                            error_msg = f"Execution Error in {len(sample_errors)}/{len(test_inputs[i])} tests: " + "; ".join(list(set(sample_errors))[:2])
                    elif not is_correct:
                        error_msg = "Logic Error: One or more test cases failed."
                else:
                    failed_extractions += 1
                    error_msg = "Failed to extract code from <answer> tags."

                if is_correct:
                    correct += 1
                total += 1

                # Detailed Logging for every sample
                status = "✅ PASSED" if is_correct else "❌ FAILED"
                logging.info(f"\n{'='*80}\n{status} | Sample ID: {batch_ids[i]} ({len(test_inputs[i])} test cases)\n{'='*80}")

                # Log ALL test case details
                for t_idx in range(len(test_inputs[i])):
                    t_status = "PASS" if (predicted_grids and t_idx < len(predicted_grids) and grids_match(predicted_grids[t_idx], true_answers[i][t_idx])) else "FAIL"
                    logging.info(f"Test {t_idx+1}: [{t_status}]")
                    logging.info(f"  📥 Input:     \n{grid_to_string(test_inputs[i][t_idx])}")
                    logging.info(f"  🎯 Expected:  \n{grid_to_string(true_answers[i][t_idx])}")
                    logging.info(f"  🔮 Predicted: \n{grid_to_string(predicted_grids[t_idx]) if predicted_grids and t_idx < len(predicted_grids) else 'N/A'}")

                if error_msg:
                    logging.info(f"⚠️ Error:    {error_msg}")

                logging.info(f"\n📝 Full Response:\n{'-'*40}\n{response}\n{'-'*40}")
                logging.info(f"\n💻 Extracted Code:\n{'-'*40}\n{generated_code}\n{'-'*40}\n")

                # Also print to console for failures to keep user informed
                if not is_correct:
                    print(f"\n❌ Sample {batch_ids[i]} Failed (see log for details)")

                # Store result in requested style
                results.append({
                    'problem_id': batch_ids[i],
                    'observation_1': str(train_examples_list[i]), # Training examples
                    'observation_2': str(test_inputs[i]),        # Test inputs
                    'hypothesis_1': generated_code,              # Generated code
                    'hypothesis_2': str(predicted_grids),        # Predicted outputs
                    'true_label': 1,                             # Always 1 (correct)
                    'predicted_label': 1 if is_correct else 0,   # 1 if correct, 0 if wrong
                    'test_input': test_inputs[i],
                    'true_answer': true_answers[i],
                    'predicted_answer': predicted_grids,
                    'reasoning': reasoning,
                    'code': generated_code,
                    'response': response,
                    'error': error_msg,
                    'correct': is_correct,
                    'invalid': False
                })

            # Success! Move to next batch
            current_idx += batch_size_actual
            pbar.update(batch_size_actual)

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                raise

            torch.cuda.empty_cache()
            if current_batch_size > 1:
                new_batch_size = current_batch_size // 2
                print(f"\n⚠️ CUDA OutOfMemoryError at batch_size={current_batch_size}, halving to {new_batch_size} and retrying current batch...")
                current_batch_size = new_batch_size
            else:
                print(f"\n❌ CUDA OutOfMemoryError even with batch_size=1 at index {current_idx}. Skipping this sample.")
                try:
                    problem_id = dataset[current_idx]['idx'] if 'idx' in dataset[current_idx] else current_idx
                except:
                    problem_id = current_idx

                results.append({
                    'problem_id': problem_id,
                    'test_input': None,
                    'true_answer': None,
                    'predicted_answer': None,
                    'reasoning': None,
                    'code': None,
                    'error': "CUDA Out of Memory",
                    'correct': False,
                    'invalid': False
                })
                total += 1
                current_idx += 1
                pbar.update(1)

    pbar.close()

    end_time = time.time()
    total_time = end_time - start_time

    accuracy = correct / total if total > 0 else 0.0
    extraction_rate = (total - failed_extractions) / total if total > 0 else 0.0

    # Calculate metrics using sklearn
    y_true = [r['true_label'] for r in results if not r.get('invalid', False)]
    y_pred = [r['predicted_label'] for r in results if not r.get('invalid', False)]

    if y_true:
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred).tolist()
    else:
        precision_macro = recall_macro = f1_macro = 0.0
        precision_weighted = recall_weighted = f1_weighted = 0.0
        precision_per_class = recall_per_class = f1_per_class = support_per_class = []
        conf_matrix = []

    print(f"\n📊 {model_name} Final Results:")
    print(f"   Accuracy (on valid): {accuracy:.4f} ({correct}/{total})")
    print(f"   Extraction Rate: {extraction_rate:.4f}")
    print(f"   Invalid Inputs: {invalid_inputs}")
    print(f"   Failed Extractions: {failed_extractions}")
    print(f"   Execution Errors: {execution_errors}")
    print(f"   Timeouts: {timeout_errors}")

    return {
        'accuracy': accuracy,
        'extraction_rate': extraction_rate,
        'correct': correct,
        'total': total,
        'invalid_inputs': invalid_inputs,
        'failed_extractions': failed_extractions,
        'execution_errors': execution_errors,
        'timeout_errors': timeout_errors,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision_per_class.tolist() if isinstance(precision_per_class, np.ndarray) else precision_per_class,
        'recall_per_class': recall_per_class.tolist() if isinstance(recall_per_class, np.ndarray) else recall_per_class,
        'f1_per_class': f1_per_class.tolist() if isinstance(f1_per_class, np.ndarray) else f1_per_class,
        'support_per_class': support_per_class.tolist() if isinstance(support_per_class, np.ndarray) else support_per_class,
        'confusion_matrix': conf_matrix,
        'time': total_time,
        'results': results
    }

def evaluate_model_with_dynamic_batch(model, tokenizer, args, model_name):
    """
    Wrapper for evaluate_on_acr. 
    The OOM handling is now integrated directly into the evaluation loop 
    to avoid restarting from the beginning.
    """
    return evaluate_on_acr(
        model,
        tokenizer,
        args.max_samples,
        model_name,
        args.batch_size,
        args.split
    )


# ## 7. Result Persistence and Comparison
# Utilities for caching evaluation results, saving categorized outcomes (successes, failures, etc.), and generating comparison summaries between different model versions.

# In[ ]:


def ensure_raw_results_cached(args):
    """
    Ensure raw ACR results are cached on disk for the current configuration.
    Returns the loaded or newly computed raw_results dict.
    """
    # 1. Manual Skipping
    if getattr(args, 'skip_raw', False):
        logging.info("⏭️ Manual Skipping: --skip_raw is True. Bypassing raw model evaluation.")
        return None

    dataset_name = "miniarc"
    split = args.split

    # 2. Automatic Caching
    # The script checks the Output Directory at args.output_path/raw_model/miniarc/.
    # Save in Evaluation directory instead of project root
    raw_results_dir = os.path.join(get_grpo_dir(), args.output_path, "raw_model", "miniarc")
    os.makedirs(raw_results_dir, exist_ok=True)

    raw_results_file = os.path.join(
        raw_results_dir,
        "raw_results_train_all.json"
    )

    if os.path.exists(raw_results_file):
        logging.info(f"📂 Automatic Caching: Found cached raw model results at {raw_results_file}")
        logging.info("   Skipping GPU-heavy inference.")
        with open(raw_results_file, "r") as f:
            raw_results = json.load(f)
        return raw_results

    logging.info(f"🔁 Automatic Caching: No cached raw results found at {raw_results_file}")
    logging.info("   Running raw model evaluation and saving as output for future runs...")

    raw_model, raw_tokenizer = load_raw_model(args.cuda_device)
    raw_results = evaluate_model_with_dynamic_batch(
        raw_model, raw_tokenizer, args, "Raw Model (cached)"
    )
    del raw_model
    torch.cuda.empty_cache()

    if raw_results is None:
        logging.error("❌ Failed to compute raw model results; cannot cache.")
        return None

    raw_results_with_meta = {
        "model_path": RAW_MODEL_PATH,
        "dataset": dataset_name,
        "split": split,
        "max_samples": args.max_samples,
        "accuracy": raw_results['accuracy'],
        "extraction_rate": raw_results.get('extraction_rate', 0.0),
        "correct": raw_results['correct'],
        "total": raw_results['total'],
        "failed_extractions": raw_results.get('failed_extractions'),
        "precision_macro": raw_results.get('precision_macro'),
        "precision_weighted": raw_results.get('precision_weighted'),
        "recall_macro": raw_results.get('recall_macro'),
        "recall_weighted": raw_results.get('recall_weighted'),
        "f1_macro": raw_results.get('f1_macro'),
        "f1_weighted": raw_results.get('f1_weighted'),
        "precision_per_class": raw_results.get('precision_per_class'),
        "recall_per_class": raw_results.get('recall_per_class'),
        "f1_per_class": raw_results.get('f1_per_class'),
        "support_per_class": raw_results.get('support_per_class'),
        "confusion_matrix": raw_results.get('confusion_matrix'),
        "time": raw_results.get('time'),
        "results": raw_results['results']
    }

    with open(raw_results_file, "w") as f:
        json.dump(raw_results_with_meta, f, indent=2)
    logging.info(f"💾 Automatic Caching: Raw model results saved to: {raw_results_file}")

    return raw_results_with_meta

def ensure_finetuned_results_cached(args, ckpt_name):
    """
    Ensure fine-tuned model results are cached on disk for the current configuration.
    Returns True if cached results are found, False otherwise.
    """
    # Save in Evaluation directory instead of project root
    ckpt_output_dir = os.path.join(get_grpo_dir(), args.output_path, ckpt_name, "miniarc")

    if os.path.exists(ckpt_output_dir) and os.path.exists(os.path.join(ckpt_output_dir, "all_cases.json")):
        print(f"\n📂 Found cached fine-tuned model results: {ckpt_output_dir}")
        return True

    print("\n🔁 No cached fine-tuned model results found for this configuration.")
    return False


def evaluate_checkpoint_cases(args, checkpoint_path):
    """
    Given a single checkpoint, evaluate it vs cached raw results and save:
      - all_cases.json
      - disagreement_cases.json
    under: args.output_path/<checkpoint_name>/
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

    # Get cached (or newly computed) fine-tuned results
    if ensure_finetuned_results_cached(args, ckpt_name):
        print(f"✅ Using cached fine-tuned model results for per-case evaluation: {ckpt_name}")
        # If we have cached results, we still need to return them for the caller
        ckpt_output_dir = os.path.join(get_grpo_dir(), args.output_path, ckpt_name, "miniarc")
        with open(os.path.join(ckpt_output_dir, "all_cases.json"), "r") as f:
            finetuned_results = json.load(f)
        return {
            "raw_results": raw_results,
            "finetuned_results": finetuned_results,
            "all_cases_file": os.path.join(ckpt_output_dir, "all_cases.json"),
            "disagreement_file": os.path.join(ckpt_output_dir, "disagreement_cases.json")
        }

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
        return None

    # Build per-case comparison
    ckpt_output_dir = os.path.join(get_grpo_dir(), args.output_path, ckpt_name, "miniarc")
    
    # print(output_dir)
    print(ckpt_output_dir)
    os.makedirs(ckpt_output_dir, exist_ok=True)

    disagreement_cases = []
    if raw_results:
        # Index results by problem_id
        raw_by_id = {r['problem_id']: r for r in raw_results["results"]}
        ft_by_id = {r['problem_id']: r for r in finetuned_results["results"]}

        for pid, raw_r in raw_by_id.items():
            if pid not in ft_by_id:
                continue
            ft_r = ft_by_id[pid]

            case_entry = {
                "problem_id": pid,
                "test_input": raw_r["test_input"],          
                "true_answer": raw_r["true_answer"],  
                "raw": {
                    "predicted_answer": raw_r["predicted_answer"],
                    "reasoning": raw_r["reasoning"],
                    "correct": raw_r["correct"]
                },
                "finetuned": {
                    "predicted_answer": ft_r["predicted_answer"],
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
    else:
        disagreement_file = None

    finetune_results_with_meta = {
        "model_path": checkpoint_path,
        "dataset": "miniarc",
        "split": args.split,
        "max_samples": args.max_samples,
        "accuracy": finetuned_results['accuracy'],
        "extraction_rate": finetuned_results.get('extraction_rate', 0.0),
        "correct": finetuned_results['correct'],
        "total": finetuned_results['total'],
        "failed_extractions": finetuned_results['failed_extractions'],
        "precision_macro": finetuned_results.get('precision_macro'),
        "precision_weighted": finetuned_results.get('precision_weighted'),
        "recall_macro": finetuned_results.get('recall_macro'),
        "recall_weighted": finetuned_results.get('recall_weighted'),
        "f1_macro": finetuned_results.get('f1_macro'),
        "f1_weighted": finetuned_results.get('f1_weighted'),
        "precision_per_class": finetuned_results.get('precision_per_class'),
        "recall_per_class": finetuned_results.get('recall_per_class'),
        "f1_per_class": finetuned_results.get('f1_per_class'),
        "support_per_class": finetuned_results.get('support_per_class'),
        "confusion_matrix": finetuned_results.get('confusion_matrix'),
        "time": finetuned_results.get('time'),
        "results": finetuned_results['results']
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

def save_categorized_results(results, output_dir):
    """Save categorized results (successes, failures, timeouts) to separate JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    successes = [r for r in results['results'] if r['correct']]
    extraction_failures = [r for r in results['results'] if r['error'] == "Failed to extract code from <answer> tags."]
    logic_errors = [r for r in results['results'] if r['error'] == "Logic Error: One or more test cases failed."]
    execution_errors = [r for r in results['results'] if r['error'] and "Execution Error" in r['error']]
    timeout_errors = [r for r in results['results'] if r['error'] and "Timeout Error" in r['error']]

    with open(os.path.join(output_dir, 'successes.json'), 'w') as f:
        json.dump(successes, f, indent=2)

    with open(os.path.join(output_dir, 'extraction_failures.json'), 'w') as f:
        json.dump(extraction_failures, f, indent=2)

    with open(os.path.join(output_dir, 'logic_errors.json'), 'w') as f:
        json.dump(logic_errors, f, indent=2)

    if execution_errors:
        with open(os.path.join(output_dir, 'execution_errors.json'), 'w') as f:
            json.dump(execution_errors, f, indent=2)

    if timeout_errors:
        with open(os.path.join(output_dir, 'timeout_errors.json'), 'w') as f:
            json.dump(timeout_errors, f, indent=2)

    logging.info(f"Categorized results saved to {output_dir}:")
    logging.info(f"  - successes.json: {len(successes)}")
    logging.info(f"  - extraction_failures.json: {len(extraction_failures)}")
    logging.info(f"  - logic_errors.json: {len(logic_errors)}")
    if execution_errors:
        logging.info(f"  - execution_errors.json: {len(execution_errors)}")
    if timeout_errors:
        logging.info(f"  - timeout_errors.json: {len(timeout_errors)}")

def save_results(raw_results, finetuned_results, best_checkpoint_info, output_dir):
    """Save evaluation results to JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save raw model results
    if raw_results:
        raw_output = {
            'model': RAW_MODEL_PATH,
            'evaluation_time': timestamp,
            'metrics': {
                'accuracy': raw_results['accuracy'],
                'extraction_rate': raw_results.get('extraction_rate', 0.0)
            },
            'correct': raw_results['correct'],
            'total': raw_results['total'],
            'invalid_inputs': raw_results.get('invalid_inputs', 0),
            'failed_extractions': raw_results['failed_extractions'],
            'detailed_results': raw_results['results']
        }

        raw_file = os.path.join(output_dir, f"raw_model_results_{timestamp}.json")
        with open(raw_file, 'w') as f:
            json.dump(raw_output, f, indent=2)
        print(f"\n💾 Raw model results saved to: {raw_file}")

    if not finetuned_results:
        print("❌ Fine-tuned model evaluation not available")
        return None

    # Save fine-tuned model results
    finetuned_output = {
        'base_model': RAW_MODEL_PATH,
        'checkpoint': best_checkpoint_info['path'],
        'validation_score': best_checkpoint_info['score'],
        'evaluation_time': timestamp,
        'metrics': {
            'accuracy': finetuned_results['accuracy'],
            'extraction_rate': finetuned_results.get('extraction_rate', 0.0)
        },
        'correct': finetuned_results['correct'],
        'total': finetuned_results['total'],
        'invalid_inputs': finetuned_results.get('invalid_inputs', 0),
        'failed_extractions': finetuned_results['failed_extractions'],
        'detailed_results': finetuned_results['results']
    }

    finetuned_file = os.path.join(output_dir, f"finetuned_model_results_{timestamp}.json")
    with open(finetuned_file, 'w') as f:
        json.dump(finetuned_output, f, indent=2)
    print(f"💾 Fine-tuned model results saved to: {finetuned_file}")

    # Save comparison summary
    if raw_results:
        improvement = finetuned_results['accuracy'] - raw_results['accuracy']
        relative_improvement = (improvement / raw_results['accuracy'] * 100) if raw_results['accuracy'] > 0 else 0
        extraction_improvement = finetuned_results.get('extraction_rate', 0.0) - raw_results.get('extraction_rate', 0.0)
    else:
        improvement = 0
        relative_improvement = 0
        extraction_improvement = 0

    summary = {
        'evaluation_time': timestamp,
        'dataset': 'MiniARC Grid Transformation',
        'split': 'eval',
        'num_samples': finetuned_results['total'],
        'raw_model': {
            'path': RAW_MODEL_PATH,
            'metrics': {
                'accuracy': raw_results['accuracy'] if raw_results else 0,
                'extraction_rate': raw_results.get('extraction_rate', 0.0) if raw_results else 0
            },
            'correct': raw_results['correct'] if raw_results else 0,
            'total': raw_results['total'] if raw_results else 0,
            'invalid_inputs': raw_results.get('invalid_inputs', 0) if raw_results else 0,
            'failed_extractions': raw_results['failed_extractions'] if raw_results else 0
        },
        'finetuned_model': {
            'base_model': RAW_MODEL_PATH,
            'checkpoint': best_checkpoint_info['path'],
            'validation_score': best_checkpoint_info['score'],
            'metrics': {
                'accuracy': finetuned_results['accuracy'],
                'extraction_rate': finetuned_results.get('extraction_rate', 0.0)
            },
            'correct': finetuned_results['correct'],
            'total': finetuned_results['total'],
            'invalid_inputs': finetuned_results.get('invalid_inputs', 0),
            'failed_extractions': finetuned_results['failed_extractions']
        },
        'comparison': {
            'accuracy_improvement': improvement,
            'accuracy_relative_improvement_percent': relative_improvement,
            'extraction_improvement': extraction_improvement,
            'overall_improved': improvement > 0
        }
    }

    summary_file = os.path.join(output_dir, f"comparison_summary_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Comparison summary saved to: {summary_file}")
    
    
    # Save disagreement and all cases summary
    if raw_results:
        raw_by_id = {r['problem_id']: r for r in raw_results['results']}
        ft_by_id = {r['problem_id']: r for r in finetuned_results['results']}

        disagreement_cases, all_cases = [], []

        for pid, raw_r in raw_by_id.items():
            if pid not in ft_by_id:
                continue
            ft_r = ft_by_id[pid]

            all_cases.append({
                "problem_id": pid,
                "test_input": raw_r["test_input"],          
                "true_answer": raw_r["true_answer"],  
                "raw": {
                    "predicted_answer": raw_r["predicted_answer"],
                    "reasoning": raw_r["reasoning"],
                    "correct": raw_r["correct"]
                },
                "finetuned": {
                    "predicted_answer": ft_r["predicted_answer"],
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
                "test_input": raw_r["test_input"],          
                "true_answer": raw_r["true_answer"],  
                "raw": {
                    "predicted_answer": raw_r["predicted_answer"],
                    "reasoning": raw_r["reasoning"],
                    "correct": raw_r["correct"]
                },
                "finetuned": {
                    "predicted_answer": ft_r["predicted_answer"],
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

def print_comparison(summary):
    """Print formatted comparison results."""
    print("\n" + "="*80)
    print("📊 MINIARC EVALUATION: RAW vs FINE-TUNED MODEL")
    print("="*80)

    raw_metrics = summary['raw_model']['metrics']
    ft_metrics = summary['finetuned_model']['metrics']

    print("\n🤖 RAW MODEL:")
    if summary['raw_model']['total'] > 0:
        print(f"   Accuracy:  {raw_metrics['accuracy']:.4f} ({raw_metrics['accuracy']*100:.2f}%) - {summary['raw_model']['correct']}/{summary['raw_model']['total']} correct")
        if summary['raw_model'].get('invalid_inputs', 0) > 0:
            print(f"   Invalid Inputs: {summary['raw_model']['invalid_inputs']} (excluded from accuracy)")
        print(f"   Extraction Rate: {raw_metrics['extraction_rate']:.4f} ({raw_metrics['extraction_rate']*100:.2f}%)")
    else:
        print("   (Skipped or not available)")

    print("\n🎯 FINE-TUNED MODEL:")
    print(f"   Checkpoint: {os.path.basename(summary['finetuned_model']['checkpoint'])}")
    val_score = summary['finetuned_model']['validation_score']
    val_score_str = f"{val_score:.4f}" if isinstance(val_score, (int, float)) else str(val_score)
    print(f"   Validation Score: {val_score_str}")
    print(f"   Accuracy:  {ft_metrics['accuracy']:.4f} ({ft_metrics['accuracy']*100:.2f}%) - {summary['finetuned_model']['correct']}/{summary['finetuned_model']['total']} correct")
    if summary['finetuned_model'].get('invalid_inputs', 0) > 0:
        print(f"   Invalid Inputs: {summary['finetuned_model']['invalid_inputs']} (excluded from accuracy)")
    print(f"   Extraction Rate: {ft_metrics['extraction_rate']:.4f} ({ft_metrics['extraction_rate']*100:.2f}%)")

    if summary['raw_model']['total'] > 0:
        print("\n📈 IMPROVEMENTS:")
        comp = summary['comparison']
        acc_imp = comp['accuracy_improvement']
        acc_rel = comp['accuracy_relative_improvement_percent']
        ext_imp = comp['extraction_improvement']

        print(f"   Accuracy:  {acc_imp:+.4f} ({acc_imp*100:+.2f}%) | Relative: {acc_rel:+.2f}%")
        print(f"   Extraction: {ext_imp:+.4f} ({ext_imp*100:+.2f}%)")

    print("\n" + "-"*80)

    if summary['raw_model']['total'] > 0:
        comp = summary['comparison']
        if comp['overall_improved']:
            print("✅ RESULT: Fine-tuning on your dataset IMPROVED performance on MiniARC!")
            print(f"   • Accuracy improved by {comp['accuracy_relative_improvement_percent']:.2f}% (relative)")
            print(f"   The model shows better grid transformation ability.")
        elif comp['accuracy_improvement'] < 0:
            print("⚠️  RESULT: Fine-tuning on your dataset DECREASED performance on MiniARC.")
            print(f"   • Accuracy decreased by {comp['accuracy_relative_improvement_percent']:.2f}% (relative)")
            print(f"   • This suggests potential overfitting to your training data.")
        else:
            print("➖ RESULT: Fine-tuning had NO SIGNIFICANT IMPACT on MiniARC performance.")
            print(f"   The model maintained baseline rule inference ability.")
    else:
        print("ℹ️  RESULT: Fine-tuned model evaluated. No raw baseline comparison available.")

    print("="*80 + "\n")


# ## 8. Visualization and Reporting Utilities
# Provides functions for generating styled summary tables, plotting performance metrics, analyzing error distributions, and visualizing accuracy vs. task complexity.

# In[ ]:


def display_results_summary(results_dict):
    """Displays a styled summary table of the evaluation results using pandas."""
    if not results_dict or 'results' not in results_dict or not results_dict['results']:
        print("No results to display.")
        return pd.DataFrame()

    all_rows = []
    for r in results_dict['results']:
        if r.get('invalid', False):
            all_rows.append({
                'ID': r['problem_id'],
                'Test #': 'N/A',
                'Input': 'N/A',
                'Expected': 'N/A',
                'Predicted': 'N/A',
                'Correct': False,
                'All Passed': False,
                'Error/Status': r['error']
            })
            continue

        num_tests = len(r['test_input'])
        for t_idx in range(num_tests):
            pred = r['predicted_answer'][t_idx] if r['predicted_answer'] and t_idx < len(r['predicted_answer']) else None
            test_correct = grids_match(pred, r['true_answer'][t_idx])

            all_rows.append({
                'ID': r['problem_id'],
                'Test #': f"Test {t_idx+1}/{num_tests}",
                'Input': grid_to_string(r['test_input'][t_idx]),
                'Expected': grid_to_string(r['true_answer'][t_idx]),
                'Predicted': grid_to_string(pred) if pred is not None else 'N/A',
                'Correct': test_correct,
                'All Passed': r['correct'],
                'Error/Status': r['error'] if not test_correct else 'Success'
            })

    summary_df = pd.DataFrame(all_rows)
    if summary_df.empty:
        print("No detailed results to display.")
        return summary_df

    summary_df['Error/Status'] = summary_df['Error/Status'].fillna('Success')

    print(f"\n📋 Detailed Results Summary (All test instances):")

    # Apply styling
    def style_rows(row):
        if row['Correct'] is True:
            return ['background-color: #d4edda'] * len(row)
        else:
            return ['background-color: #f8d7da'] * len(row)

    # Display first 50 rows to avoid overwhelming the notebook
    styled_df = summary_df.head(50).style.apply(style_rows, axis=1)
    display(styled_df)

    return summary_df

def plot_evaluation_metrics(results_dict, model_name="Model", output_dir=None):
    """Plots key metrics: Accuracy, Extraction Rate, and Error Distribution."""
    if not results_dict or 'total' not in results_dict or results_dict['total'] == 0:
        print("No metrics to plot.")
        return

    # 1. Overall Metrics Bar Chart
    metrics = {
        'Accuracy': results_dict['accuracy'],
        'Extraction Rate': results_dict['extraction_rate']
    }

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette='viridis')
    plt.ylim(0, 1.1)
    plt.title(f'Overall Performance: {model_name}')
    plt.ylabel('Score')
    for i, v in enumerate(metrics.values()):
        plt.text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')

    # 2. Error Distribution Pie Chart
    plt.subplot(1, 2, 2)
    # Calculate logic errors (wrong answer but code executed successfully)
    logic_errors = max(0, results_dict['total'] - results_dict['correct'] - results_dict['failed_extractions'] - results_dict['execution_errors'] - results_dict.get('timeout_errors', 0))

    error_counts = {
        'Correct': results_dict['correct'],
        'Logic Error': logic_errors,
        'Execution Error': results_dict['execution_errors'],
        'Timeout Error': results_dict.get('timeout_errors', 0),
        'Extraction Failed': results_dict['failed_extractions'],
        'Invalid Input': results_dict.get('invalid_inputs', 0)
    }
    # Filter out zero counts
    error_counts = {k: v for k, v in error_counts.items() if v > 0}

    if not error_counts:
        plt.text(0.5, 0.5, 'No outcomes to display', ha='center', va='center')
    else:
        plt.pie(error_counts.values(), labels=error_counts.keys(), autopct='%1.1f%%', 
                colors=sns.color_palette('pastel'), startangle=140)
    plt.title('Outcome Distribution')

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'overall_metrics.png'))
    plt.show()

def plot_accuracy_by_complexity(results_dict, output_dir=None):
    """Analyzes and plots accuracy based on the number of unique colors in the input grid."""
    if not results_dict or 'results' not in results_dict or not results_dict['results']:
        return

    data = []
    for r in results_dict['results']:
        if r.get('invalid', False):
            continue

        # Use the number of unique colors in the first test input as a complexity proxy
        test_input = r['test_input']
        if isinstance(test_input, list) and len(test_input) > 0:
            grid = test_input[0]
            # Flatten grid and get unique elements
            unique_colors = len(set([cell for row in grid for cell in row]))
        else:
            unique_colors = 0

        data.append({
            'unique_colors': unique_colors,
            'correct': 1 if r['correct'] else 0
        })

    df = pd.DataFrame(data)
    if df.empty: return

    # Group by unique colors and calculate accuracy
    complexity_acc = df.groupby('unique_colors')['correct'].agg(['mean', 'count']).reset_index()
    complexity_acc.columns = ['Unique Colors', 'Accuracy', 'Sample Count']

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=complexity_acc, x='Unique Colors', y='Accuracy', palette='magma')
    plt.title('Accuracy vs. Input Complexity (Unique Colors)')
    plt.ylim(0, 1.1)

    # Add sample counts on top of bars
    for i, p in enumerate(ax.patches):
        count = complexity_acc.iloc[i]['Sample Count']
        ax.annotate(f'n={int(count)}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=9)

    if output_dir:
        plt.savefig(os.path.join(output_dir, 'accuracy_by_complexity.png'))
    plt.show()

def plot_model_comparison(comparison_df, output_path=None):
    """Plots a bar chart comparing accuracy across different models."""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=comparison_df, x='Model', y='Accuracy', palette='magma')
    plt.title('Accuracy Comparison Across All Models')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    for i, v in enumerate(comparison_df['Accuracy']):
        plt.text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
    plt.tight_layout()
    if output_path:
        plt.savefig(os.path.join(output_path, 'model_comparison.png'))
    plt.show()

def analyze_failures(results_dict):
    """Prints a detailed breakdown of why the model failed."""
    if not results_dict or 'results' not in results_dict or not results_dict['results']:
        print("No results to analyze.")
        return

    print("\n" + "="*50)
    print("🔍 DETAILED FAILURE ANALYSIS")
    print("="*50)

    df = pd.DataFrame(results_dict['results'])

    # 0. Invalid Inputs
    invalid_count = results_dict.get('invalid_inputs', 0)
    if invalid_count > 0:
        print(f"⚠️ Invalid Inputs (Skipped): {invalid_count}")

    if df.empty:
        print("No valid results to analyze.")
        return

    failures = df[(df['correct'] == False) & (df.get('invalid', False) == False)]

    if failures.empty:
        if invalid_count == 0:
            print("✨ No failures found! Perfect score.")
        else:
            print("✨ No model failures found (excluding invalid inputs).")
        return

    # 1. Extraction Failures
    ext_fails = failures[failures['code'].isna()]
    print(f"❌ Extraction Failures: {len(ext_fails)}")

    # 2. Execution Failures
    exec_fails = failures[failures['error'].str.contains('Execution Error', na=False)]
    print(f"❌ Execution Errors: {len(exec_fails)}")
    if not exec_fails.empty:
        print("   Top Error Messages:")
        print(exec_fails['error'].value_counts().head(3))

    # 3. Timeout Failures
    timeout_fails = failures[failures['error'].str.contains('Timeout Error', na=False)]
    print(f"❌ Timeout Errors: {len(timeout_fails)}")

    # 4. Logic Failures (Code ran but got wrong answer)
    logic_fails = failures[failures['code'].notna() & ~failures['error'].str.contains('Execution Error|Timeout Error', na=False, regex=True)]
    print(f"❌ Logic Errors (Wrong Answer): {len(logic_fails)}")

    print("="*50)



def evaluate_all_checkpoints(args):
    """
    Evaluates all checkpoints in args.checkpoint_dir and compares them.
    """
    print("="*80)
    print("🚀 MINIARC MULTI-CHECKPOINT EVALUATION")
    print("="*80)
    
    checkpoint_dir = args.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Error: Checkpoint directory not found: {checkpoint_dir}")
        return

    # Discover checkpoints
    checkpoints = [d for d in os.listdir(checkpoint_dir) 
                   if d.startswith('checkpoint-') and os.path.isdir(os.path.join(checkpoint_dir, d))]
    checkpoints.sort(key=lambda x: int(x.split('-')[1]))
    
    if not checkpoints:
        print(f"⚠️ No checkpoints found in {checkpoint_dir}")
        return
        
    print(f"Found {len(checkpoints)} checkpoints: {checkpoints}")
    
    all_model_results = {}
    
    # Evaluate Raw Model first if not skipped
    if not args.skip_raw:
        print("\n" + "="*60)
        print("Evaluating Raw Model (Baseline)")
        print("="*60)
        raw_results_meta = ensure_raw_results_cached(args)
        if raw_results_meta:
            all_model_results['Raw Model'] = raw_results_meta

    # Evaluate Checkpoints
    for ckpt_name in checkpoints:
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        print(f"\n" + "="*60)
        print(f"Evaluating Checkpoint: {ckpt_name}")
        print("="*60)
        
        try:
            eval_results = evaluate_checkpoint_cases(args, ckpt_path)
            if eval_results and eval_results.get('finetuned_results'):
                all_model_results[ckpt_name] = eval_results['finetuned_results']
        except Exception as e:
            print(f"❌ Failed to evaluate {ckpt_name}: {e}")
            import traceback
            traceback.print_exc()

    # --- Comparison Summary & Best Model Selection ---
    if all_model_results:
        comparison_data = []
        for model_name, results in all_model_results.items():
            if results:
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': results['accuracy'],
                    'Extraction Rate': results.get('extraction_rate', 0.0),
                    'Correct': results['correct'],
                    'Total': results['total']
                })

        comparison_df = pd.DataFrame(comparison_data)

        # Identify the best model with priority for non-raw model if accuracies are equal
        comparison_df['IsRaw'] = comparison_df['Model'] == 'Raw Model'
        comparison_df = comparison_df.sort_values(by=['Accuracy', 'IsRaw'], ascending=[False, True])

        print("\n📊 Model Comparison Summary:")
        display(comparison_df.drop(columns=['IsRaw']).style.highlight_max(subset=['Accuracy'], color='lightgreen'))

        # Identify the best model
        best_model_name = comparison_df.iloc[0]['Model']
        best_accuracy = comparison_df.iloc[0]['Accuracy']

        print(f"\n🏆 The best model is **{best_model_name}** with an accuracy of **{best_accuracy:.2%}**.")

        # Plotting Comparison
        plot_model_comparison(comparison_df, args.output_path)

        # Save comparison results to Excel
        excel_path = os.path.join(args.output_path, 'model_comparison.xlsx')
        final_df = comparison_df.drop(columns=['IsRaw'])
        styled_df = final_df.style.highlight_max(subset=['Accuracy'], color='lightgreen')

        try:
            styled_df.to_excel(excel_path, index=False, engine='openpyxl')
            print(f"✅ Comparison results saved to: {excel_path}")
        except Exception as e:
            logging.error(f"❌ Failed to save Excel file: {e}")
            final_df.to_csv(os.path.join(args.output_path, 'model_comparison.csv'), index=False)
            print(f"⚠️ Saved to CSV instead due to Excel error.")

        # Detailed analysis for the best model
        if best_model_name in all_model_results:
            print(f"\n🔍 Detailed Analysis for Best Model: {best_model_name}")
            best_results = all_model_results[best_model_name]
            display_results_summary(best_results)
            analyze_failures(best_results)
            
            # Plot accuracy by complexity for the best model (Specific to MiniARC)
            print(f"\n📈 Plotting Accuracy by Complexity for {best_model_name}...")
            plot_accuracy_by_complexity(best_results, args.output_path)

def main():
    global RAW_MODEL_PATH, OUTPUT_DIR
    parser = argparse.ArgumentParser(description='Evaluate raw vs fine-tuned model on miniarc dataset')
    parser.add_argument('--max_samples', type=int, default=None, 
                       help='Maximum number of samples to evaluate')
    parser.add_argument('--cuda_device', type=str, default='0',
                       help='CUDA device to use (default: 0)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation. Higher values (4-8) are faster but use more GPU memory (default: 1)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test', 'validation'],
                       help='Dataset split to use (default: train).')
    parser.add_argument('--skip_raw', action='store_true',
                       help='Skip raw model evaluation (evaluate only fine-tuned model)')
    parser.add_argument('--skip_finetuned', action='store_true',
                       help='Skip fine-tuned model evaluation (evaluate only raw model)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to specific checkpoint to evaluate (e.g., /path/to/checkpoint-640). '
                            'If not provided, automatically selects the best checkpoint based on validation metrics.')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Path to directory containing multiple checkpoints (e.g., /path/to/checkpoint/). '
                            'Will evaluate ALL checkpoint-* directories found. Cannot be used with --checkpoint_path.')    
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
        print("🚀 miniarc PER-CHECKPOINT EVALUATION MODE")
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
    print("🚀 miniarc EVALUATION: RAW vs FINE-TUNED")
    print("="*70)
    print(f"Raw Model: {RAW_MODEL_PATH}")
    print(f"Training Dir: {TRAINING_DIR}")
    print(f"CUDA Device: {args.cuda_device}")
    print(f"Batch Size: {args.batch_size}")
    if args.max_samples:
        print(f"Max Samples: {args.max_samples}")
    if args.skip_raw:
        print(f"Mode: Fine-tuned model only")
    elif args.skip_finetuned:
        print(f"Mode: Raw model only")
    else:
        print(f"Mode: Both models (comparison)")
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
                print(f"   Converted to absolute path: {checkpoint_path}")
            
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
            print("\n📁 No checkpoint path provided, auto-selecting best checkpoint...")
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
        raw_results = evaluate_on_acr(raw_model, raw_tokenizer, args.max_samples, "Raw Model", args.batch_size)
        del raw_model  # Free memory
        torch.cuda.empty_cache()

    else:
        raw_results = None
        print("\n⏭️  Skipping raw model evaluation")
    
    # Evaluate fine-tuned model
    if not args.skip_finetuned:
        finetuned_model, finetuned_tokenizer = load_finetuned_model(best_checkpoint_info['path'], args.cuda_device)
        finetuned_results = evaluate_on_acr(finetuned_model, finetuned_tokenizer, args.max_samples, "Fine-tuned Model", args.batch_size)
        del finetuned_model  # Free memory
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
        
if __name__ == "__main__":
    main()

