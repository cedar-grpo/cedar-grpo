import sys
import os

_SFT_DIR = os.path.dirname(os.path.abspath(__file__))
_GRPO_DIR = os.path.join(os.path.dirname(_SFT_DIR), "GRPO")

# Add GRPO/ to sys.path so `from prompts import` resolves to GRPO/prompts.py
# and `from Evaluation.xxx import` resolves to GRPO/Evaluation/
sys.path.insert(0, _GRPO_DIR)
sys.path.insert(0, _SFT_DIR)

from datetime import datetime
import json
import logging
import multiprocessing
import queue
import random
import re
import signal
import subprocess
import threading
import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset
from huggingface_hub import HfApi
from transformers import TrainerCallback
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

try:
    from vllm import SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

    class SamplingParams:  # type: ignore[no-redef]
        """Minimal stand-in when vLLM is not installed."""

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

from prompts import (
    create_uniadilr_prompt,
    create_copa_cause_prompt as create_copa_prompt,
    create_climate_fever_prompt,
    create_causelogics_prompt,
    create_list_functions_prompt,
    create_abductionrules_prompt,
    create_crypto_functions_prompt,
)
from Evaluation.evaluate_list_function_raw_vs_finetuned import lists_match

warnings.filterwarnings("ignore")

# =========================
# Model Configuration
# =========================
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit"
LOAD_IN_4BIT = True
LOAD_IN_8BIT = False
USE_VLLM = False
TEMPERATURE = 0.7
LORA_RANK = 64
LORA_ALPHA = 64
GPU_MEMORY_UTILIZATION = 1.0
MAX_SEQ_LENGTH = 4096
MAX_PROMPT_LENGTH = 2048
MAX_COMPLETION_LENGTH = MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH

RESUME_FROM_CHECKPOINT = False
PREVIOUS_RUN_DIR = ""  # MUST be set to the folder name under results/ when RESUME_FROM_CHECKPOINT = True
RUN_DESC = "test_run_sft"
CUDA_VISIBLE_DEVICES = "0"

# =========================
# Training Configuration
# =========================
LEARNING_RATE = 1e-5
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.99
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 2
LR_SCHEDULER_TYPE = "cosine"
OPTIM = "adamw_torch"

EVAL_STEPS = 25          # Evaluate on validation set every N steps
SAVE_STEPS = 25          # Save checkpoint every N steps
LOG_VALIDATION = True    # Whether to log validation metrics
LOG_TRAIN_EVERY = 1      # Save training log every N steps
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 0.1
NUM_TRAIN_EPOCHS = 1
LOGGING_STEPS = 1
SAVE_TOTAL_LIMIT = 20

# =========================
# Data Configuration
# =========================
NUM_SAMPLES = 50  # None for full dataset
EXCLUDED_DATASETS = []
TRAIN_DATA_VAL = "v1"

ERROR_LOG_PATH = "error_log.log"
TRAINING_LOG_PATH = "training_log.json"
VALIDATION_LOG_PATH = "validation_log.json"
VALIDATION_METRICS_PATH = "val_metrics.json"
WANDB_DISABLED = "true"

# Random State Configuration
RANDOM_STATE = 3407
TORCH_SEED = 42
NUMPY_SEED = 42

# Module-level sampling params (mirrors GRPO sampling_params = SamplingParams(...))
sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    top_p=0.95,
    max_tokens=MAX_COMPLETION_LENGTH,
)

# Response template that marks the start of the assistant turn in the chat format.
# DataCollatorForCompletionOnlyLM uses this to mask out the prompt tokens from the loss.
# Derived dynamically at runtime via get_response_template(tokenizer) so it works for
# any model family (Llama-3 uses  "<|start_header_id|>assistant<|end_header_id|>\n\n",
# Qwen-2.5 uses "<|im_start|>assistant\n", etc.).
RESPONSE_TEMPLATE: str = ""  # populated in main() after the tokenizer is loaded


def get_response_template(tokenizer) -> str:
    """Derive the assistant response-start marker from the tokenizer's chat template.

    Works by comparing the formatted output with and without add_generation_prompt:
    the extra tokens appended by add_generation_prompt=True are exactly the string
    that DataCollatorForCompletionOnlyLM must search for to locate the answer span.
    """
    dummy = [{"role": "user", "content": "x"}]
    try:
        without_gen = tokenizer.apply_chat_template(dummy, tokenize=False, add_generation_prompt=False)
        with_gen = tokenizer.apply_chat_template(dummy, tokenize=False, add_generation_prompt=True)
        template = with_gen[len(without_gen):]
    except Exception:
        template = ""
        
    if not template:
        model_name_lower = getattr(tokenizer, "name_or_path", MODEL_NAME).lower()
        if "llama-3" in model_name_lower:
            template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "qwen" in model_name_lower:
            template = "<|im_start|>assistant\n"
        else:
            raise ValueError(
                "Could not derive response template from tokenizer chat template. "
                "Set RESPONSE_TEMPLATE manually for this model."
            )
    return template


def get_run_name() -> str:
    model_name = MODEL_NAME.split("/")[-1].replace("-", "_")
    if LOAD_IN_8BIT:
        model_name += "_8bit"
    elif LOAD_IN_4BIT:
        model_name += "_bnb_4bit"

    now = datetime.now()
    name = (
        f"dt{now.strftime('%m.%d.%H.%M')}_e{NUM_TRAIN_EPOCHS}_{model_name}"
        f"_lr{LEARNING_RATE}_r{LORA_RANK}_b{PER_DEVICE_TRAIN_BATCH_SIZE}_sft"
    )
    if RUN_DESC:
        name += f"_{RUN_DESC}"
    return name


def get_results_dir(run_name: str | None = None) -> str:
    if run_name is None:
        run_name = get_run_name()
    if RESUME_FROM_CHECKPOINT:
        if not PREVIOUS_RUN_DIR:
            raise ValueError(
                "RESUME_FROM_CHECKPOINT is True but PREVIOUS_RUN_DIR is empty. "
                "Set PREVIOUS_RUN_DIR to the folder name under results/ to resume from."
            )
        run_name = PREVIOUS_RUN_DIR
    return f"results/{run_name}"


def set_environment() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    os.environ["WANDB_DISABLED"] = WANDB_DISABLED
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "240"

    random.seed(RANDOM_STATE)
    np.random.seed(NUMPY_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)


def get_model_size(model_name: str) -> int | None:
    try:
        api = HfApi()
        model_info = api.model_info(model_name)
        return sum(file.size for file in model_info.siblings if file.size)
    except Exception:
        return None


def _normalize_list_function_code(code: str) -> str:
    """Normalize a ListFunction code snippet to training requirements.

    1. Remove standalone comment lines.
    2. Rename the function signature to 'def transform(lst):'.
    3. Replace all remaining standalone 'x' identifiers with 'lst'.
    """
    # Remove standalone comment lines (including EOF comments without trailing newline)
    code = re.sub(r'^[ \t]*#.*(?:\n|$)', '', code, flags=re.MULTILINE)
    # Rename function: def c[num](x): -> def transform(lst):
    code = re.sub(r'def\s+[a-zA-Z0-9_]+\(x\):', 'def transform(lst):', code)
    # Replace remaining standalone 'x' with 'lst' (word-boundary safe)
    code = re.sub(r'\bx\b', 'lst', code)
    return code.strip()


def transform_to_prompt_format(example: Dict[str, Any], record_id: int) -> Dict[str, Any]:
    dataset_name = example.get("datasetName", "")
    rule_test_input = None
    rule_test_output = None
    rule_train_examples = None
    rationale = example.get("rationale")

    if dataset_name == "UniADILR":
        system_prompt, user_prompt = create_uniadilr_prompt(example)
        ground_truth = json.dumps(example["proof"])

    elif dataset_name == "BalancedCOPA":
        system_prompt, user_prompt = create_copa_prompt(example)
        ground_truth = str(example["label"] + 1)

    elif dataset_name == "CauseLogics":
        system_prompt, user_prompt = create_causelogics_prompt(example)
        ground_truth = example["Label"].upper()

    elif dataset_name == "ListFunction":
        system_prompt, user_prompt = create_list_functions_prompt(example)

        # Ground truth is the normalized function code (goes into <answer> tags)
        ground_truth = _normalize_list_function_code(example["function"])

        # Test inputs/outputs kept separately for validation evaluation.
        # Keep all tests, not only index 0, to avoid inflated validation reward.
        test_inputs = []
        test_outputs = []
        for t in example["test"]:
            raw_input = t["input"]
            input_obj = json.loads(raw_input) if isinstance(raw_input, str) else raw_input
            test_inputs.append(input_obj)

            raw_output = t["output"]
            output_obj = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
            test_outputs.append(output_obj)
        rule_test_input = json.dumps(test_inputs, ensure_ascii=False)
        rule_test_output = json.dumps(test_outputs, ensure_ascii=False)

        rule_train_examples = []
        for ex in example["train"]:
            ex_in = json.loads(ex["input"]) if isinstance(ex["input"], str) else ex["input"]
            ex_out = json.loads(ex["output"]) if isinstance(ex["output"], str) else ex["output"]
            rule_train_examples.append({"input": ex_in, "output": ex_out})
        rule_train_examples = json.dumps(rule_train_examples, ensure_ascii=False)

    elif dataset_name == "ClimateFever":
        system_prompt, user_prompt = create_climate_fever_prompt(example)
        label_map = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO", 3: "DISPUTED"}
        ground_truth = label_map[example["claim_label"]]

    elif dataset_name == "AbductionRules":
        system_prompt, user_prompt = create_abductionrules_prompt(example)
        ground_truth = example["answer"]

    elif dataset_name == "Crypto":
        system_prompt, user_prompt = create_crypto_functions_prompt(example)

        # Ground truth is the function code (goes into <answer> tags)
        ground_truth = example["function"].strip()

        # Test inputs/outputs kept separately for validation evaluation
        test_inputs = [t["input"] for t in example["test"]]
        test_outputs = [t["output"] for t in example["test"]]
        rule_test_input = json.dumps(test_inputs, ensure_ascii=False)
        rule_test_output = json.dumps(test_outputs, ensure_ascii=False)

        train_examples = example.get("train", {})
        if isinstance(train_examples, dict):
            train_examples = train_examples.get("normal", [])
        rule_train_examples = json.dumps(train_examples, ensure_ascii=False)

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    

    
    return {
        "prompt": prompt,
        "record_id": record_id,
        "ground_truth": ground_truth,
        "reasoning_type": example.get("reasoning_type", "abduction"),
        "dataset_name": dataset_name,
        "rule_test_input": rule_test_input,
        "rule_test_output": rule_test_output,
        "rule_train_examples": rule_train_examples,
        # Forward rationale fields so build_text_row can use them as <think> content.
        "rationale": rationale
    }


def load_and_prepare_data(tokenizer) -> tuple[Dataset, Dataset]:
    logging.info("[DATA] Starting data loading and preparation...")

    train_path = os.path.join(_SFT_DIR, "dataset", "train_split.json")
    val_path = os.path.join(_SFT_DIR, "dataset", "val_split.json")

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    logging.info(f"[DATA] Loaded {len(train_data)} raw training examples from {train_path}")

    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    logging.info(f"[DATA] Loaded {len(val_data)} raw validation examples from {val_path}")

    train_skipped = 0
    train_transformed = []
    for idx, example in enumerate(train_data):
        if example.get("datasetName") in EXCLUDED_DATASETS:
            continue
        try:
            train_transformed.append(transform_to_prompt_format(example, record_id=idx))
        except Exception as e:
            train_skipped += 1
            logging.error(
                f"[DATA] Skipping train example idx={idx} dataset={example.get('datasetName')}: "
                f"{type(e).__name__}: {e}"
            )
    if train_skipped:
        logging.warning(f"[DATA] Skipped {train_skipped} train examples due to transform errors")
    logging.info(f"[DATA] Built {len(train_transformed)} train examples after filtering")

    val_skipped = 0
    val_transformed = []
    for idx, example in enumerate(val_data):
        if example.get("datasetName") in EXCLUDED_DATASETS:
            continue
        try:
            val_transformed.append(transform_to_prompt_format(example, record_id=idx))
        except Exception as e:
            val_skipped += 1
            logging.error(
                f"[DATA] Skipping val example idx={idx} dataset={example.get('datasetName')}: "
                f"{type(e).__name__}: {e}"
            )
    if val_skipped:
        logging.warning(f"[DATA] Skipped {val_skipped} val examples due to transform errors")
    logging.info(f"[DATA] Built {len(val_transformed)} val examples after filtering")

    if NUM_SAMPLES is not None and NUM_SAMPLES > 0:
        train_transformed = train_transformed[:NUM_SAMPLES]
        val_transformed = val_transformed[: max(1, int(NUM_SAMPLES * 0.2))]
        logging.info(f"[DATA] Applied NUM_SAMPLES={NUM_SAMPLES} cap")
    else:
        total = len(train_transformed) + len(val_transformed)
        val_cap = max(1, int(total * 0.2))
        if len(val_transformed) > val_cap:
            logging.info(f"[DATA] Capping val set from {len(val_transformed)} → {val_cap} (20% of total={total})")
            val_transformed = val_transformed[:val_cap]

    train_dist: Dict[str, int] = {}
    for ex in train_transformed:
        train_dist[ex["dataset_name"]] = train_dist.get(ex["dataset_name"], 0) + 1
    val_dist: Dict[str, int] = {}
    for ex in val_transformed:
        val_dist[ex["dataset_name"]] = val_dist.get(ex["dataset_name"], 0) + 1
    logging.info(f"[DATA] Final train samples={len(train_transformed)} | Distribution: {train_dist}")
    logging.info(f"[DATA] Final val   samples={len(val_transformed)} | Distribution: {val_dist}")

    def build_text_row(example: Dict[str, Any]) -> Dict[str, Any]:
        # Build the assistant response with <think> + <answer> tags.
        # Goal 1: teach the model to always emit the <think>...</think><answer>...</answer> structure.
        # Goal 2: anchor the loss on the correct answer inside <answer>.
        # Since our datasets contain no written rationales, we use a minimal placeholder
        # inside <think> so the model learns the tag format while still being trained on
        # the correct answer. Datasets that DO carry a 'rationale' field will use it directly.
        # Parse the raw ground_truth into the canonical form that extract_prediction also returns,
        # so the training target is consistent with how the model will be evaluated.
        parsed_gt = parse_ground_truth(example["ground_truth"], example["dataset_name"])
        # Convert to a plain string for the <answer> tag.
        # Sets (UniADILR) become sorted comma-separated numbers to match the system prompt format.
        if isinstance(parsed_gt, set):
            gt_for_answer = ", ".join(str(n) for n in sorted(parsed_gt))
        else:
            gt_for_answer = str(parsed_gt) if not isinstance(parsed_gt, str) else parsed_gt

        rationale = example.get("rationale")
        if rationale:
            think_content = str(rationale).strip()
        else:
            think_content = f"Based on careful analysis of the question, the answer is {gt_for_answer}."

        assistant_content = f"<think>\n{think_content}\n</think>\n<answer>{gt_for_answer}</answer>"

        messages = [
            {"role": "system", "content": example["prompt"][0]["content"]},
            {"role": "user", "content": example["prompt"][1]["content"]},
            {"role": "assistant", "content": assistant_content},
        ]

        # Use the tokenizer's chat template. If it fails, we MUST fail loudly,
        # as falling back to a custom text format will prevent DataCollatorForCompletionOnlyLM
        # from finding the response template mask and corrupt the training loss.
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception as e:
            logging.error(
                f"[DATA] apply_chat_template FAILED | record_id={example.get('record_id')} | "
                f"dataset={example.get('dataset_name')} | error={type(e).__name__}: {e} | "
                f"system_len={len(str(messages[0]['content']))} "
                f"user_len={len(str(messages[1]['content']))} "
                f"assistant_len={len(str(messages[2]['content']))}"
            )
            raise

        return {**example, "text": text, "target_content": assistant_content}

    logging.info(f"[DATA] Building text rows for {len(train_transformed)} train examples (applying chat template)...")
    train_ds = Dataset.from_list(train_transformed).map(build_text_row)
    logging.info(f"[DATA] Train dataset ready: {len(train_ds)} rows")
    if len(train_ds) > 0:
        logging.info("📝 --- SAMPLE FORMATTED TRAINING TEXT (First 1500 chars) ---")
        logging.info(train_ds[0]["text"][:1500] + "\n---------------------------------------------------------")

    logging.info(f"[DATA] Building text rows for {len(val_transformed)} val examples (applying chat template)...")
    val_ds = Dataset.from_list(val_transformed).map(build_text_row)
    logging.info(f"[DATA] Val dataset ready: {len(val_ds)} rows")

    return train_ds, val_ds


# =========================
# Prediction Helpers (mirrored from GRPO)
# =========================

def normalize_abductionrules_answer(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text.strip()).lower()


def extract_prediction(text: str, datasetName: str):
    """Extract the model's answer from <answer>...</answer> tags."""
    if isinstance(text, list):
        if len(text) > 0 and isinstance(text[-1], dict) and "content" in text[-1]:
            text = text[-1]["content"]
        elif len(text) > 0 and isinstance(text[0], str):
            text = text[0]
        else:
            text = str(text)

    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
    answer = (m.group(1).strip() if m else "").strip()

    if datasetName == "UniADILR":
        nums = re.findall(r"\b(\d+)\b", answer)
        return set(int(n) for n in nums)
    if datasetName == "BalancedCOPA":
        return int(answer) if answer else -123
    if datasetName == "CauseLogics":
        return answer.upper()
    if datasetName == "ClimateFever":
        return answer.upper()
    if datasetName == "AbductionRules":
        return normalize_abductionrules_answer(answer)
    if datasetName in ("ListFunction", "Crypto"):
        return answer
    return answer


def parse_ground_truth(gt, datasetName: str):
    """Convert stored ground_truth into the same type as extract_prediction returns."""
    if datasetName == "UniADILR":
        proof_str = json.loads(gt) if isinstance(gt, str) else gt
        if isinstance(proof_str, str):
            left = proof_str.split("->")[0] if "->" in proof_str else proof_str
            nums = re.findall(r"sent(\d+)", left)
            return set(int(n) for n in nums)
        raise ValueError(f"Unexpected UniADILR proof type: {type(proof_str)}")
    if datasetName == "BalancedCOPA":
        return int(gt)
    if datasetName == "CauseLogics":
        return str(gt).upper()
    if datasetName == "ClimateFever":
        return str(gt).upper()
    if datasetName == "AbductionRules":
        return normalize_abductionrules_answer(gt)
    if datasetName == "ListFunction":
        return str(gt).strip()
    if datasetName == "Crypto":
        return gt
    return gt


def extract_code(response) -> Optional[str]:
    """Extract Python code from <answer> tags, stripping markdown fences."""
    if isinstance(response, list):
        if len(response) > 0 and isinstance(response[-1], dict) and "content" in response[-1]:
            response = response[-1]["content"]
        elif len(response) > 0 and isinstance(response[0], str):
            response = response[0]
        else:
            response = str(response)

    tag_match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.IGNORECASE | re.DOTALL)
    if not tag_match:
        return None

    code = tag_match.group(1).strip()
    code = re.sub(r"^```python\s*", "", code)
    code = re.sub(r"^```\s*", "", code)
    code = re.sub(r"\s*```$", "", code)
    return code


def string_to_list(s) -> Optional[List[int]]:
    """Safely convert a string like '[1, 2, 3]' to a Python list of ints."""
    if not s:
        return None
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list) and all(isinstance(v, int) for v in parsed):
            return parsed
        return None
    except Exception:
        return None


def _execute_worker(code_str, input_data, result_queue, dataset_name):
    """Worker function for multiprocessing execution to handle timeouts."""
    local_ns: Dict[str, Any] = {}
    try:
        import numpy as _np
        local_ns["np"] = _np
    except ImportError:
        pass

    try:
        exec(code_str, {"__builtins__": __builtins__, "np": local_ns.get("np")}, local_ns)
        if "transform" not in local_ns:
            result_queue.put((None, "Error: Function 'transform' not found in generated code."))
            return
        result = local_ns["transform"](input_data)
        if dataset_name == "ListFunction":
            if not isinstance(result, list):
                result_queue.put((None, f"Error: ListFunction expected list, got {type(result).__name__}"))
                return
            if not all(isinstance(x, int) for x in result):
                result_queue.put((None, "Error: ListFunction expected list of integers"))
                return
        elif dataset_name == "Crypto":
            if not isinstance(result, str):
                result_queue.put((None, f"Error: Crypto expected string, got {type(result).__name__}"))
                return
        result_queue.put((result, None))
    except Exception as e:
        result_queue.put((None, f"Execution Error: {str(e)}"))


def execute_transform(code_str, input_data, dataset_name, timeout=5):
    """Execute model-generated code with a timeout via multiprocessing."""
    q: multiprocessing.Queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_execute_worker, args=(code_str, input_data, q, dataset_name))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(timeout=2)
        if p.is_alive():
            p.kill()
            p.join()
        return None, f"Timeout after {timeout}s"
    if not q.empty():
        return q.get()
    return None, "No result returned"


def _calculate_code_reward(code, dataset_name, test_input, ground_truth_raw, train_examples_raw=None):
    """Calculate reward for code-based tasks (ListFunction, Crypto)."""
    if not code:
        return 0.0, "No code found", "No code found"

    all_examples = []
    try:
        test_inputs = json.loads(test_input) if isinstance(test_input, str) else test_input
    except Exception:
        test_inputs = test_input
    try:
        gt_outputs = json.loads(ground_truth_raw) if isinstance(ground_truth_raw, str) else ground_truth_raw
    except Exception:
        gt_outputs = ground_truth_raw

    # Backward-compatible normalization for both old (single test) and new (all tests) storage.
    if not isinstance(test_inputs, list):
        test_inputs = [test_inputs]
    if not isinstance(gt_outputs, list):
        gt_outputs = [gt_outputs]

    if dataset_name == "ListFunction":
        # If old format accidentally yields a plain list of ints, wrap it as one test case.
        if test_inputs and all(isinstance(v, int) for v in test_inputs):
            test_inputs = [test_inputs]
        if gt_outputs and all(isinstance(v, int) for v in gt_outputs):
            gt_outputs = [gt_outputs]

    for inp, out in zip(test_inputs, gt_outputs):
        all_examples.append({"input": inp, "output": out})

    passed_count = 0
    test_predictions = []
    test_errors = []

    for ex in all_examples:
        ex_out, ex_err = execute_transform(code, ex["input"], dataset_name)
        if dataset_name == "Crypto":
            is_correct = not ex_err and ex_out == ex["output"]
        else:
            is_correct = not ex_err and lists_match(ex_out, ex["output"])
        if is_correct:
            passed_count += 1
            test_predictions.append(ex_out)
        else:
            test_predictions.append(f"Error: {ex_err}" if ex_err else ex_out)
            test_errors.append(ex_err if ex_err else "Logic Error")

    reward = 1.0 if passed_count == len(all_examples) else 0.0

    if dataset_name == "Crypto":
        prediction = test_predictions
        error = None if not test_errors else "; ".join(dict.fromkeys(test_errors))
    else:
        prediction = test_predictions[0] if test_predictions else None
        error = test_errors[0] if test_errors else None

    return reward, prediction, error


# =========================
# Evaluation Worker Thread System (mirrored from GRPO Cell 9)
# Runs run_eval_checkpoints_midtrain.sh concurrently during training,
# serialised via a background queue so checkpoints are evaluated one at a time
# without blocking the training loop.
# =========================

_evaluation_job_queue: queue.Queue = queue.Queue()
_evaluation_worker_thread: Optional[threading.Thread] = None
_evaluation_worker_lock = threading.Lock()


def _evaluation_worker() -> None:
    """Background worker that processes evaluation jobs one at a time (serialized)."""
    while True:
        job_args = _evaluation_job_queue.get()
        if job_args is None:  # Sentinel value to stop the worker
            _evaluation_job_queue.task_done()
            break
        try:
            _execute_evaluation_job(*job_args)
        except Exception as e:
            print(f"[QUEUE ERROR] Evaluation job failed with exception: {e}")
        finally:
            _evaluation_job_queue.task_done()


def _ensure_evaluation_worker_running() -> None:
    """Ensure the background evaluation worker thread is running."""
    global _evaluation_worker_thread
    with _evaluation_worker_lock:
        if _evaluation_worker_thread is None or not _evaluation_worker_thread.is_alive():
            _evaluation_worker_thread = threading.Thread(
                target=_evaluation_worker,
                daemon=True,
                name="EvaluationJobWorker",
            )
            _evaluation_worker_thread.start()


def _execute_evaluation_job(
    output_dir: str,
    root_dir: str,
    base_results_dir: str,
    raw_model_path: str,
    run_name: str,
    chkpt_name: str,
    base_model_name: str,
    train_data: str,
    cuda_device: int,
    evaluate_checkpoints: int,
) -> None:
    """The actual synchronous execution of an evaluation job. Runs inside the worker thread."""
    bash_env: Dict[str, str] = os.environ.copy()
    bash_env.update({
        "OUTPUT_DIR": str(output_dir),
        "ROOT_DIR": str(root_dir),
        "BASE_RESULTS_DIR": str(base_results_dir),
        "RAW_MODEL_PATH": str(raw_model_path),
        "RUN_NAME": str(run_name),
        "CHKPT_NAME": str(chkpt_name),
        "BASE_MODEL_NAME": str(base_model_name),
        "TRAIN_DATA": str(train_data),
        "CUDA_DEVICE": str(cuda_device),
        "EVALUATE_CHECKPOINTS": str(evaluate_checkpoints),
    })

    bash_script_path = os.path.join(_GRPO_DIR, "Evaluation", "run_eval_checkpoints_midtrain.sh")

    print("--- Executing Bash Script ---")
    print(f"Target Script: {bash_script_path}")
    print(f"RUN_NAME: {run_name}")
    print(f"CUDA_DEVICE: {cuda_device}")
    print("-----------------------------")

    try:
        subprocess.run(
            ["bash", bash_script_path],
            check=True,
            text=True,
            env=bash_env,
        )
        print("\nBash script executed successfully.")
        time.sleep(60)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Bash script failed with exit code {e.returncode}")
    except FileNotFoundError:
        print(f"\nERROR: The Bash script '{bash_script_path}' was not found.")


def run_evaluation_job(
    output_dir: str,
    root_dir: str,
    base_results_dir: str,
    raw_model_path: str,
    run_name: str,
    chkpt_name: str,
    base_model_name: str,
    train_data: str,
    cuda_device: int,
    evaluate_checkpoints: int,
) -> None:
    """
    Queues an evaluation job for async execution.

    - Returns immediately (non-blocking to main thread)
    - Jobs are processed one at a time in order (serialized in the background)
    """
    _ensure_evaluation_worker_running()
    print(f"[QUEUE] Adding job to queue: {run_name}")
    _evaluation_job_queue.put((
        output_dir,
        root_dir,
        base_results_dir,
        raw_model_path,
        run_name,
        chkpt_name,
        base_model_name,
        train_data,
        cuda_device,
        evaluate_checkpoints,
    ))


def wait_for_all_evaluation_jobs() -> None:
    """Block until all queued evaluation jobs are complete."""
    _evaluation_job_queue.join()
    print("[QUEUE] All evaluation jobs completed.")


def shutdown_evaluation_worker() -> None:
    """Gracefully shutdown the worker thread after finishing current jobs."""
    _evaluation_job_queue.put(None)  # Sentinel to stop
    if _evaluation_worker_thread is not None:
        _evaluation_worker_thread.join()
    print("[QUEUE] Worker thread shut down.")


# =========================
# Callbacks
# =========================

class SFTTrainingLogCallback(TrainerCallback):
    """
    Records per-step training losses and samples of trainer data to training_log.json.
    """

    def __init__(self, train_ds: Dataset, output_path: str, log_every: int = LOG_TRAIN_EVERY) -> None:
        self.train_ds = train_ds
        self.output_path = output_path
        self.log_every = log_every
        self.training_log: List[Dict[str, Any]] = []

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if logs is None:
            return
        loss = logs.get("loss")
        if loss is None:
            return
        entry: Dict[str, Any] = {
            "step": state.global_step,
            "loss": loss,
            "epoch": state.epoch,
        }

        # Pull EXACTLY the pre-built assistant target block from our original train_ds
        # (SFTTrainer internally tokenizes and drops string columns, so we keep a ref to
        #  the original dataset which still has 'target_content'.)
        try:
            idx = (state.global_step - 1) % len(self.train_ds)
            entry["trained_on_sample"] = self.train_ds[idx]["target_content"]
        except Exception as e:
            logging.warning(f"[TRAIN LOG] Could not fetch target_content at step {state.global_step}: {e}")

        self.training_log.append(entry)

        if len(self.training_log) % self.log_every == 0:
            try:
                with open(self.output_path, "w", encoding="utf-8") as f:
                    json.dump(self.training_log, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logging.warning(f"Failed to save training log: {e}")


class EnhancedEpochCallback(TrainerCallback):
    """
    Custom callback to log epoch progress and handle validation.
    Adapted from the GRPO EnhancedEpochCallback for SFT training.
    - Logs start and end of each epoch.
    - Triggers validation every EVAL_STEPS steps (and at epoch end).
    - Writes validation_log.json and val_metrics.json identical to GRPO.
    """

    def __init__(
        self,
        val_dataset: Dataset,
        results_dir: str,
        tokenizer,
        run_name: str,
        eval_interval: int = EVAL_STEPS,
        use_vllm: bool = False,
    ) -> None:
        self.val_dataset = val_dataset
        self.results_dir = results_dir
        self.tokenizer = tokenizer
        self.run_name = run_name
        self.step_count = 0
        self.last_eval_step = -1
        self.start_time: Optional[float] = None
        self.validation_metrics: Dict[str, Any] = {}
        self.trainer = None
        self.formatted_inputs: Optional[List[str]] = None
        self.eval_interval = eval_interval
        self.use_vllm = use_vllm and VLLM_AVAILABLE

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        self.start_time = time.time()
        print(f"🚀 Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        # Format each prompt individually to avoid truncation/concatenation issues
        self.formatted_inputs = [
            self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in self.val_dataset["prompt"]
        ]

        # Trigger validation for step 0 (raw model)
        if LOG_VALIDATION and self.trainer:
            print("\n📊 Evaluating raw model (step 0) before training...")
            self.evaluate_validation(
                self.trainer.model,
                self.trainer.processing_class,
                state.global_step,
            )

    def on_epoch_begin(self, args, state, control, **kwargs) -> None:
        epoch_idx = int(state.epoch) + 1  # Convert to 1-indexed
        print(f"\n📍 Starting epoch {epoch_idx}")

    def on_step_end(self, args, state, control, **kwargs) -> None:
        self.step_count += 1
        if self.step_count % 50 == 0 and self.start_time is not None:
            elapsed = time.time() - self.start_time
            steps_per_sec = self.step_count / elapsed
            print(f"   Step {self.step_count} | Speed: {steps_per_sec:.2f} steps/s")

        if (
            LOG_VALIDATION
            and self.eval_interval
            and (self.step_count % self.eval_interval == 0)
            and self.trainer
        ):
            self.evaluate_validation(
                self.trainer.model,
                self.trainer.processing_class,
                state.global_step,
            )

    def evaluate_validation(self, model, tokenizer, step: int) -> None:
        if step == self.last_eval_step:
            return
        self.last_eval_step = step

        print(f"\n🔍 Validation at step {step}:")

        try:
            val_rewards: List[float] = []
            validation_log: List[Dict[str, Any]] = []
            batch_size = PER_DEVICE_EVAL_BATCH_SIZE

            with torch.no_grad():
                FastLanguageModel.for_inference(model)
                for batch_num in range(0, len(self.val_dataset), batch_size):
                    batch = self.formatted_inputs[batch_num : batch_num + batch_size]

                    if self.use_vllm:
                        outputs = model.fast_generate(
                            batch,
                            lora_request=None,
                            sampling_params=sampling_params,
                        )
                        completions = [o.outputs[0].text.strip() for o in outputs]
                    else:
                        old_padding_side = tokenizer.padding_side
                        tokenizer.padding_side = "left"
                        batch_encodings = tokenizer(
                            batch,
                            return_tensors="pt",
                            padding=True,
                            truncation=False,
                        ).to(model.device)
                        outputs = model.generate(
                            **batch_encodings,
                            do_sample=True,
                            temperature=sampling_params.temperature,
                            top_p=sampling_params.top_p,
                            max_new_tokens=sampling_params.max_tokens,
                        )
                        tokenizer.padding_side = old_padding_side

                        prompt_lengths = batch_encodings["input_ids"].shape[1]
                        generated_tokens = outputs[:, prompt_lengths:]
                        completions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

                    batch_records = [
                        self.val_dataset[batch_num + j] for j in range(len(completions))
                    ]
                    results = self._evaluate_batch(completions, batch_records)

                    for batch_idx, result in enumerate(results):
                        val_rewards.append(result["reward"])
                        log_entry: Dict[str, Any] = {
                            "record_id": self.val_dataset["record_id"][batch_num + batch_idx],
                            "dataset_name": result.get("dataset_name", ""),
                            "input": batch[batch_idx],
                            "ground_truth": result["ground_truth"],
                            "predicted": result["predicted"],
                            "reward": result["reward"],
                            "completion": result["completion"],
                        }
                        if result.get("extracted_code") is not None:
                            log_entry["extracted_code"] = result["extracted_code"]
                        if result.get("execution_error") is not None:
                            log_entry["execution_error"] = result["execution_error"]
                        validation_log.append(log_entry)

            FastLanguageModel.for_training(model)

            if val_rewards:
                avg_val_reward = sum(val_rewards) / len(val_rewards)
                
                # 🔍 --- SHOW ONE SAMPLE PREDICTION LOG ---
                if validation_log:
                    sample = validation_log[0]
                    print(f"\n[EVAL SAMPLE] Dataset: {sample.get('dataset_name')} | Reward: {sample.get('reward')}")
                    print(f"[EVAL TRUTH]:  {sample.get('ground_truth')}")
                    print(f"[EVAL PREDICT]:{sample.get('predicted')}")
                    print(f"[EVAL EXTRACTION / COMPLETION]:\n{sample.get('completion')[:800]}...\n------------------")

                print(f"   📊 Validation reward: {avg_val_reward:.4f} (n={len(val_rewards)})")

                # Use float epoch key to avoid overwriting mid-epoch evaluations
                epoch_key = f"{self.trainer.state.epoch:.4f}" if self.trainer else f"step_{step}"
                self.validation_metrics[epoch_key] = {
                    "avg_reward": avg_val_reward,
                    "num_samples": len(val_rewards),
                }

                # Save validation log
                val_log_path = os.path.join(self.results_dir, VALIDATION_LOG_PATH)
                existing_data: Dict[str, Any] = {}
                if os.path.exists(val_log_path):
                    with open(val_log_path, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                existing_data[epoch_key] = validation_log
                with open(val_log_path, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)

                # Save validation metrics
                val_metrics_path = os.path.join(self.results_dir, VALIDATION_METRICS_PATH)
                all_metrics: Dict[str, Any] = {}
                if os.path.exists(val_metrics_path):
                    with open(val_metrics_path, "r", encoding="utf-8") as f:
                        all_metrics = json.load(f)
                all_metrics[epoch_key] = {
                    "avg_reward": avg_val_reward,
                    "num_samples": len(val_rewards),
                }
                with open(val_metrics_path, "w", encoding="utf-8") as f:
                    json.dump(all_metrics, f, ensure_ascii=False, indent=2)
            else:
                logging.warning(f"⚠️  No validation rewards computed. Step: {step}")

        except Exception as e:
            logging.exception(f"❌ Validation error: {e}")

    def _evaluate_batch(
        self, completions: List[str], records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Score a batch of completions against ground truth."""
        results = []
        for completion, record in zip(completions, records):
            dataset_name = record.get("dataset_name", "")
            ground_truth_raw = record.get("ground_truth", "")
            extracted_code = None
            execution_error = None

            try:
                if dataset_name in ("ListFunction", "Crypto"):
                    # ground_truth_raw is now function code; use rule_test_output for expected outputs
                    rule_test_output = record.get("rule_test_output")
                    try:
                        eval_gt = json.loads(rule_test_output) if isinstance(rule_test_output, str) else rule_test_output
                    except Exception:
                        eval_gt = string_to_list(rule_test_output) if dataset_name == "ListFunction" else rule_test_output

                    extracted_code = extract_code(completion)
                    # Enforce function-only validation for code tasks.
                    if extracted_code and "def transform" in extracted_code:
                        reward, predicted, execution_error = _calculate_code_reward(
                            extracted_code,
                            dataset_name,
                            record.get("rule_test_input"),
                            eval_gt,
                            record.get("rule_train_examples"),
                        )
                    else:
                        reward = 0.0
                        predicted = None
                        execution_error = None
                        if extracted_code is None:
                            execution_error = "Missing <answer>...</answer> code block"
                        else:
                            execution_error = "Code answer must define 'def transform(...)'"
                else:
                    ground_truth = parse_ground_truth(ground_truth_raw, dataset_name)
                    predicted = extract_prediction(completion, dataset_name)
                    reward = 1.0 if predicted == ground_truth else 0.0
                    ground_truth_raw = (
                        sorted(list(ground_truth)) if dataset_name == "UniADILR" else ground_truth
                    )
                    predicted = (
                        sorted(list(predicted)) if dataset_name == "UniADILR" else predicted
                    )
                result: Dict[str, Any] = {
                    "reward": reward,
                    "predicted": predicted,
                    "ground_truth": ground_truth_raw,
                    "completion": completion,
                    "dataset_name": dataset_name,
                    "extracted_code": extracted_code,
                    "execution_error": execution_error,
                }
            except Exception as e:
                logging.exception(f"Error evaluating completion: {e}")
                result = {
                    "reward": 0.0,
                    "predicted": [],
                    "ground_truth": [],
                    "completion": completion,
                    "dataset_name": dataset_name,
                }
            results.append(result)
        return results

    def on_epoch_end(self, args, state, control, **kwargs) -> None:
        completed_epoch_idx = int(state.epoch)
        print(f"✅ Completed epoch {completed_epoch_idx}")
        if LOG_VALIDATION and self.trainer:
            self.evaluate_validation(
                self.trainer.model,
                self.trainer.processing_class,
                state.global_step,
            )
        elif LOG_VALIDATION:
            logging.warning("⚠️  No trainer assigned; cannot evaluate validation.")

    def on_save(self, args, state, control, **kwargs) -> None:
        print(f"💾 Checkpoint saved at step {state.global_step}")
        output_dir = os.path.join("Evaluation", self.run_name)
        cuda_device = str(CUDA_VISIBLE_DEVICES.split(",")[0])  # Use the same GPU, removed +1 to avoid crashing on single-GPU setups
        run_evaluation_job(
            output_dir=output_dir,
            root_dir=os.path.dirname(output_dir),
            base_results_dir="results",
            raw_model_path=MODEL_NAME,
            run_name=self.run_name,
            chkpt_name=f"checkpoint-{state.global_step}",
            base_model_name=MODEL_NAME.split("/")[-1],
            train_data=TRAIN_DATA_VAL,
            cuda_device=cuda_device,
            evaluate_checkpoints=1,
        )


class DetailedProgressCallback(TrainerCallback):

    def __init__(self) -> None:
        self.start_time = time.time()
        self.last_log_time = time.time()

    def on_step_begin(self, args, state, control, **kwargs) -> None:
        current_time = time.time()
        if state.global_step % 10 == 0 or (current_time - self.last_log_time) > 30:
            elapsed = current_time - self.start_time
            steps_per_sec = state.global_step / elapsed if elapsed > 0 else 0
            remaining_steps = state.max_steps - state.global_step
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
            progress_pct = (state.global_step / state.max_steps) * 100 if state.max_steps else 0
            print(
                f"\r⏳ Step {state.global_step}/{state.max_steps} ({progress_pct:.1f}%) | "
                f"Speed: {steps_per_sec:.2f} steps/s | ETA: {eta_str} | "
                f"Epoch: {state.epoch:.1f}",
                end="",
                flush=True,
            )
            self.last_log_time = current_time

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if logs:
            print()
            log_str = " | ".join(
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in logs.items()
                if k != "epoch"
            )
            print(f"📊 {log_str}")
            logging.info(log_str)

    def on_epoch_end(self, args, state, control, **kwargs) -> None:
        print()
        elapsed = time.time() - self.start_time
        print(
            f"\n✅ Epoch {int(state.epoch)} completed | "
            f"Total time: {elapsed / 60:.1f}m | "
            f"Steps: {state.global_step}/{state.max_steps}"
        )
        logging.info(f"Epoch {int(state.epoch)} completed")

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        print(f"\n🎯 Training will run for {state.max_steps} steps")
        print(f"📝 Logging every {args.logging_steps} steps")
        print(f"💾 Saving checkpoints every {args.save_steps} steps")
        print("-" * 70)
        logging.info("Training started")


def main() -> None:
    set_environment()

    print("🔧 SFT Abductive Reasoning Training Pipeline")
    print(f"Model: {MODEL_NAME}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    model_size = get_model_size(MODEL_NAME)
    if model_size:
        print(f"Model size: {model_size / (1024 ** 3):.2f} GB")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        max_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
        fast_inference=USE_VLLM,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        use_gradient_checkpointing="unsloth",
        random_state=RANDOM_STATE,
    )

    run_name = get_run_name()
    results_dir = get_results_dir(run_name)
    checkpoint_dir = os.path.join(results_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Wire up logging to both a file and stdout (mirrors GRPO Cell 12 logging setup)
    log_fmt = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_fmt,
        handlers=[
            logging.FileHandler(os.path.join(results_dir, ERROR_LOG_PATH)),
            logging.StreamHandler(sys.stdout),
        ],
    )

    train_ds, val_ds = load_and_prepare_data(tokenizer)

    training_args = SFTConfig(
        learning_rate=LEARNING_RATE,
        adam_beta1=ADAM_BETA1,
        adam_beta2=ADAM_BETA2,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        optim=OPTIM,
        logging_steps=LOGGING_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_seq_length=MAX_SEQ_LENGTH,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        save_steps=SAVE_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        report_to=["tensorboard"],
        output_dir=checkpoint_dir,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        dataset_text_field="text",
        dataset_num_proc=1,
        packing=False,
        max_steps=-1,
        # --- FIX for Unsloth + transformers v4.57+ compatibility ---
        # Prevents the new default behavior that scales loss in-place (loss *= factor)
        # which breaks with Unsloth's fused loss tensor and can turn it into a Python int.
        average_tokens_across_devices=False,
    )

    training_log_callback = SFTTrainingLogCallback(
        train_ds=train_ds,
        output_path=os.path.join(results_dir, TRAINING_LOG_PATH),
        log_every=LOG_TRAIN_EVERY,
    )
    enhanced_callback = EnhancedEpochCallback(
        val_dataset=val_ds,
        results_dir=results_dir,
        tokenizer=tokenizer,
        run_name=run_name,
        eval_interval=EVAL_STEPS,
        use_vllm=USE_VLLM,
    )
    progress_callback = DetailedProgressCallback()

    response_template = get_response_template(tokenizer)
    print(f"📌 Response template: {repr(response_template)}")

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    # Reset vision-related token IDs that GRPOTrainer sets but SFTTrainer may leave
    # unset (mirrors GRPO Cell 11: trainer.image_token_id = None, etc.)
    trainer.image_token_id = None
    trainer.vision_start_token_id = None
    trainer.vision_end_token_id = None

    # Give the enhanced callback a reference to the trainer (same pattern as GRPO)
    enhanced_callback.trainer = trainer
    trainer.add_callback(training_log_callback)
    trainer.add_callback(enhanced_callback)
    trainer.add_callback(progress_callback)

    def signal_handler(sig, frame):
        print("\n⚠️  Interrupt signal received. Saving progress...")
        logging.warning("Training interrupted by user")
        trainer.save_model(os.path.join(results_dir, "checkpoint", "interrupted"))
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("\n🚀 Starting Training")
    print("=" * 70)
    print(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🏷️  Run name: {run_name}")
    print(f"📁 Output directory: {results_dir}")
    print(f"🔍 Logs will be saved to: {ERROR_LOG_PATH}")
    print("-" * 70)

    logging.info(f"Starting training run: {run_name}")
    logging.info(f"Output directory: {results_dir}")
    logging.info(f"Training config: epochs={NUM_TRAIN_EPOCHS}, batch_size={PER_DEVICE_TRAIN_BATCH_SIZE}")

    training_start_time = time.time()
    final_model_path = os.path.join(checkpoint_dir, "final_model")  # defined early to avoid NameError

    try:
        print("🎬 Initiating training loop...\n")
        trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)

        training_end_time = time.time()
        training_duration = training_end_time - training_start_time

        print("\n" + "=" * 70)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"⏱️  Duration: {training_duration / 3600:.2f} hours ({training_duration / 60:.1f} minutes)")
        print(f"🏁 Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Training completed successfully in {training_duration / 3600:.2f} hours")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        logging.warning("Training interrupted by user (KeyboardInterrupt)")

    except Exception as e:
        print(f"\n\n❌ Training failed with error: {type(e).__name__}: {e}")
        logging.exception(f"Training failed with error: {e}")
        raise

    finally:
        training_end_time = time.time()
        actual_duration = training_end_time - training_start_time

        print("\n" + "=" * 70)
        print("🔄 Cleanup and saving...")
        print("=" * 70)

        try:
            # Save final training log
            if training_log_callback.training_log:
                try:
                    log_path = os.path.join(results_dir, TRAINING_LOG_PATH)
                    with open(log_path, "w", encoding="utf-8") as f:
                        json.dump(training_log_callback.training_log, f, ensure_ascii=False, indent=2)
                    print(f"✅ Training log saved: {len(training_log_callback.training_log)} entries")
                    logging.info(f"Saved training log with {len(training_log_callback.training_log)} entries")
                except Exception as e:
                    print(f"⚠️  Failed to save training log: {e}")
                    logging.warning(f"Failed to save training log: {e}")

            final_model_path = os.path.join(checkpoint_dir, "final_model")
            os.makedirs(final_model_path, exist_ok=True)
            trainer.save_model(final_model_path)
            tokenizer.save_pretrained(final_model_path)
            print(f"✅ Model saved to: {final_model_path}")
            logging.info(f"Final model saved to: {final_model_path}")

            # Trigger evaluation for the final model (mirrors GRPO Cell 12 finally block)
            output_dir = os.path.join("Evaluation", run_name)
            cuda_device = str(CUDA_VISIBLE_DEVICES.split(",")[0])  # same GPU as training
            run_evaluation_job(
                output_dir=output_dir,
                root_dir=os.path.dirname(output_dir),
                base_results_dir="results",
                raw_model_path=MODEL_NAME,
                run_name=run_name,
                chkpt_name="final_model",
                base_model_name=MODEL_NAME.split("/")[-1],
                train_data=TRAIN_DATA_VAL,
                cuda_device=cuda_device,
                evaluate_checkpoints=1,
            )

            print(f"\n⏱️  Total elapsed time: {actual_duration / 60:.1f} minutes")
            print("=" * 70)

        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")
            logging.exception("Error during cleanup")

    metadata = {
        "run_name": run_name,
        "results_dir": results_dir,
        "model_name": MODEL_NAME,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "training_type": "SFT",
    }
    with open(os.path.join(results_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("✅ SFT training completed")
    print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
    # Wait for all async evaluation jobs to finish, then shut down the worker
    # (mirrors GRPO Cell 14: wait_for_all_evaluation_jobs() / shutdown_evaluation_worker())
    wait_for_all_evaluation_jobs()
    shutdown_evaluation_worker()
