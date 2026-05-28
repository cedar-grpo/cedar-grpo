#!/usr/bin/env python3
"""
Create BIG-Bench train/val JSON files.

Behavior:
- Downloads selected BIG-Bench tasks from Hugging Face
- Uses standard multiple-choice examples only
- Uses exactly 100 samples from each task, split as:
    - 80 train examples from the BIG-Bench train split
    - 20 val examples from the BIG-Bench validation split
- Produces:
    - 320 training examples total
    - 80 validation examples total
- Reproducible via fixed seed

Outputs written next to this script:
- bigbench_train.json
- bigbench_val.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    from datasets import load_dataset
except ImportError as e:
    raise ImportError(
        "This script requires the Hugging Face datasets package. "
        "Install it with: pip install datasets"
    ) from e


# =========================
# User-configurable values
# =========================
TRAIN_PER_TASK = 80
VAL_PER_TASK = 20
SEED = 42

DATASET_NAME = "tasksource/bigbench"
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"

TASKS = [
    "logical_deduction",
    "tracking_shuffled_objects",
    "date_understanding",
    "penguins_in_a_table",
]

OUTPUT_DIR = Path(__file__).resolve().parent
TRAIN_OUTPUT = OUTPUT_DIR / "bigbench_train.json"
VAL_OUTPUT = OUTPUT_DIR / "bigbench_val.json"

REQUIRED_OUTPUT_FIELDS = [
    "dataset_name",
    "bigbench_task",
    "bigbench_idx",
    "split",
    "input",
    "answer_choices",
    "correct_answer_index",
    "correct_answer_text",
    "answer",
    "reward_type",
]


LETTER_CHOICES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def log(msg: str) -> None:
    """Log messages to the console."""
    print(msg, flush=True)


def validate_config() -> None:
    """Validate user-configurable values."""
    if TRAIN_PER_TASK <= 0:
        raise ValueError("TRAIN_PER_TASK must be positive.")
    if VAL_PER_TASK <= 0:
        raise ValueError("VAL_PER_TASK must be positive.")
    if not TASKS:
        raise ValueError("TASKS must be non-empty.")
    if len(TASKS) != len(set(TASKS)):
        raise ValueError("TASKS contains duplicate task names.")


def load_task_split(task: str, split: str) -> List[Dict]:
    """Load one BIG-Bench task split from Hugging Face."""
    log(f"Loading BIG-Bench task={task!r}, split={split!r} ...")

    dataset = load_dataset(
        DATASET_NAME,
        task,
        split=split,
        trust_remote_code=True,
    )

    return [dict(row) for row in dataset]


def normalize_text(value) -> str:
    """Normalize text fields for duplicate checks and exact-answer storage."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def get_single_correct_answer_index(scores: List[int]) -> Optional[int]:
    """Return the single correct answer index, or None if invalid."""
    if not isinstance(scores, list):
        return None
    if not scores:
        return None

    correct_indices = [i for i, score in enumerate(scores) if score == 1]
    if len(correct_indices) != 1:
        return None

    return correct_indices[0]


def convert_example(
    example: Dict,
    task: str,
    split: str,
    seen_inputs: Set[str],
) -> Optional[Dict]:
    """Convert a raw BIG-Bench example to the target schema, or return None if invalid."""
    raw_input = example.get("inputs")
    answer_choices = example.get("multiple_choice_targets")
    scores = example.get("multiple_choice_scores")

    input_text = normalize_text(raw_input)
    if not input_text:
        return None

    # Skip duplicate inputs across both output files.
    duplicate_key = input_text.casefold()
    if duplicate_key in seen_inputs:
        return None

    if not isinstance(answer_choices, list) or not answer_choices:
        return None
    if not all(isinstance(choice, str) and choice.strip() for choice in answer_choices):
        return None

    if not isinstance(scores, list) or len(scores) != len(answer_choices):
        return None

    correct_answer_index = get_single_correct_answer_index(scores)
    if correct_answer_index is None:
        return None

    if correct_answer_index < 0 or correct_answer_index >= len(answer_choices):
        return None

    cleaned_choices = [normalize_text(choice) for choice in answer_choices]
    if any(not choice for choice in cleaned_choices):
        return None

    # BIG-Bench MC tasks are expected to have one correct score of 1 and all others 0.
    # Reject anything with non-binary scores so the reward can remain deterministic.
    if any(score not in {0, 1} for score in scores):
        return None

    correct_answer_text = cleaned_choices[correct_answer_index]
    answer_letter = LETTER_CHOICES[correct_answer_index] if correct_answer_index < len(LETTER_CHOICES) else str(correct_answer_index)

    output = {
        "dataset_name": "BIG-Bench",
        "bigbench_task": task,
        "bigbench_idx": example.get("idx"),
        "split": split,
        "input": input_text,
        "answer_choices": cleaned_choices,
        "correct_answer_index": correct_answer_index,
        "correct_answer_text": correct_answer_text,
        # Use exact text as the canonical answer; answer_letter is also retained for MC grading.
        "answer": correct_answer_text,
        "answer_letter": answer_letter,
        "reward_type": "exact_match_correct_answer_text_or_letter",
    }

    for field in REQUIRED_OUTPUT_FIELDS:
        if field not in output:
            raise ValueError(f"Converted example is missing required field: {field}")

    seen_inputs.add(duplicate_key)
    return output


def filter_valid_examples(
    rows: List[Dict],
    task: str,
    split: str,
    seen_inputs: Set[str],
) -> List[Dict]:
    """Filter and convert valid standard multiple-choice examples."""
    valid_examples: List[Dict] = []

    for row in rows:
        converted = convert_example(row, task, split, seen_inputs)
        if converted is not None:
            valid_examples.append(converted)

    return valid_examples


def sample_task_split(
    rows: List[Dict],
    task: str,
    split: str,
    sample_size: int,
    rng: random.Random,
    seen_inputs: Set[str],
) -> List[Dict]:
    """Filter, sample, and shuffle one BIG-Bench task split."""
    
    local_seen = set(seen_inputs)
    valid_examples = filter_valid_examples(rows, task, split, local_seen)

    log(
        f"Task {task!r} {split} valid standard MC examples after filtering: "
        f"{len(valid_examples)}"
    )

    if len(valid_examples) < sample_size:
        raise ValueError(
            f"Requested {sample_size} examples for task {task!r} split {split!r}, "
            f"but only found {len(valid_examples)} valid examples."
        )

    log(f"Sampling {sample_size} examples from task {task!r} split {split!r} ...")
    selected_indices = rng.sample(range(len(valid_examples)), sample_size)
    sampled_rows = [valid_examples[i] for i in selected_indices]

    log(f"Shuffling task {task!r} split {split!r} sampled examples ...")
    rng.shuffle(sampled_rows)

    for row in sampled_rows:
        seen_inputs.add(row["input"].casefold())

    return sampled_rows


def sample_and_split_task(task: str, rng: random.Random, seen_inputs: Set[str]) -> Dict[str, List[Dict]]:
    """Sample train and val examples for a single BIG-Bench task."""
    train_rows = load_task_split(task, TRAIN_SPLIT)
    train_samples = sample_task_split(
        train_rows,
        task,
        TRAIN_SPLIT,
        TRAIN_PER_TASK,
        rng,
        seen_inputs,
    )

    val_rows = load_task_split(task, VAL_SPLIT)
    val_samples = sample_task_split(
        val_rows,
        task,
        VAL_SPLIT,
        VAL_PER_TASK,
        rng,
        seen_inputs,
    )

    if len(train_samples) != TRAIN_PER_TASK:
        raise RuntimeError(f"Unexpected train split size for task {task!r}.")
    if len(val_samples) != VAL_PER_TASK:
        raise RuntimeError(f"Unexpected validation split size for task {task!r}.")

    return {
        "train": train_samples,
        "val": val_samples,
    }


def write_json(path: Path, data: List[Dict]) -> None:
    """Write data to a JSON file."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    """Main function to create train and validation splits."""
    validate_config()

    log("Configuration:")
    log(f"  DATASET_NAME = {DATASET_NAME}")
    log(f"  TASKS = {TASKS}")
    log(f"  TRAIN_PER_TASK = {TRAIN_PER_TASK}")
    log(f"  VAL_PER_TASK = {VAL_PER_TASK}")
    log(f"  TOTAL_TRAIN = {TRAIN_PER_TASK * len(TASKS)}")
    log(f"  TOTAL_VAL = {VAL_PER_TASK * len(TASKS)}")
    log(f"  SEED = {SEED}")
    log("")

    rng = random.Random(SEED)
    seen_inputs: Set[str] = set()

    all_train_samples: List[Dict] = []
    all_val_samples: List[Dict] = []

    for task in TASKS:
        log(f"=== Processing BIG-Bench task {task} ===")
        task_splits = sample_and_split_task(task, rng, seen_inputs)

        all_train_samples.extend(task_splits["train"])
        all_val_samples.extend(task_splits["val"])

        log(f"Task {task} train split complete: {len(task_splits['train'])} samples")
        log(f"Task {task} val split complete: {len(task_splits['val'])} samples")
        log("")

    log("Shuffling final train and val sets ...")
    rng.shuffle(all_train_samples)
    rng.shuffle(all_val_samples)

    expected_train_size = TRAIN_PER_TASK * len(TASKS)
    expected_val_size = VAL_PER_TASK * len(TASKS)

    assert len(all_train_samples) == expected_train_size, "Unexpected final train split size."
    assert len(all_val_samples) == expected_val_size, "Unexpected final validation split size."

    train_task_counts = {
        task: sum(1 for row in all_train_samples if row["bigbench_task"] == task)
        for task in TASKS
    }
    val_task_counts = {
        task: sum(1 for row in all_val_samples if row["bigbench_task"] == task)
        for task in TASKS
    }

    log("Per-task counts:")
    for task in TASKS:
        log(f"  {task} train: {train_task_counts[task]}")
        log(f"  {task} val: {val_task_counts[task]}")

        assert train_task_counts[task] == TRAIN_PER_TASK, (
            f"Unexpected train count for task {task}: {train_task_counts[task]}"
        )
        assert val_task_counts[task] == VAL_PER_TASK, (
            f"Unexpected val count for task {task}: {val_task_counts[task]}"
        )

    log("")
    log(f"Final train split complete: {len(all_train_samples)} samples")
    log(f"Final val split complete: {len(all_val_samples)} samples")
    log("")

    log("Writing output files...")
    write_json(TRAIN_OUTPUT, all_train_samples)
    write_json(VAL_OUTPUT, all_val_samples)

    log("Done.")
    log(
        f"Created:\n"
        f"  {TRAIN_OUTPUT}\n"
        f"  {VAL_OUTPUT}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr, flush=True)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
