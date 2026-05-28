#!/usr/bin/env python3
"""
Create MMLU train/val JSON files.

Behavior:
- Downloads MMLU subject test splits from Hugging Face
- Uses exactly 50 random samples from each selected subject
- Splits each subject into:
    - 40 train
    - 10 val
- Produces:
    - 320 training examples total
    - 80 validation examples total
- Reproducible via fixed seed

Outputs written next to this script:
- mmlu_train.json
- mmlu_val.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List

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
SAMPLES_PER_SUBJECT = 50
TRAIN_PER_SUBJECT = 40
VAL_PER_SUBJECT = 10
SEED = 42

DATASET_NAME = "cais/mmlu"
DATASET_SPLIT = "test"

SUBJECTS = [
    "abstract_algebra",
    "formal_logic",
    "logical_fallacies",
    "college_computer_science",
    "high_school_statistics",
    "high_school_physics",
    "econometrics",
    "high_school_world_history",
]

OUTPUT_DIR = Path(__file__).resolve().parent
TRAIN_OUTPUT = OUTPUT_DIR / "mmlu_train.json"
VAL_OUTPUT = OUTPUT_DIR / "mmlu_val.json"

REQUIRED_OUTPUT_FIELDS = [
    "question",
    "choices",
    "answer",
    "answer_letter",
    "subject",
    "dataset_name",
]


def log(msg: str) -> None:
    """Log messages to the console."""
    print(msg, flush=True)


def validate_config() -> None:
    """Validate user-configurable values."""
    if SAMPLES_PER_SUBJECT <= 0:
        raise ValueError("SAMPLES_PER_SUBJECT must be positive.")
    if TRAIN_PER_SUBJECT <= 0:
        raise ValueError("TRAIN_PER_SUBJECT must be positive.")
    if VAL_PER_SUBJECT <= 0:
        raise ValueError("VAL_PER_SUBJECT must be positive.")
    if TRAIN_PER_SUBJECT + VAL_PER_SUBJECT != SAMPLES_PER_SUBJECT:
        raise ValueError(
            "TRAIN_PER_SUBJECT + VAL_PER_SUBJECT must equal SAMPLES_PER_SUBJECT."
        )
    if not SUBJECTS:
        raise ValueError("SUBJECTS must contain at least one subject.")


def get_required_field(example: Dict, field_names: List[str], label: str):
    """Return the first matching field from field_names."""
    for field_name in field_names:
        if field_name in example:
            return example[field_name]

    raise ValueError(f"Example is missing required field for {label}: {field_names}")


def validate_choices(value) -> None:
    """Validate the choices field."""
    if not isinstance(value, list):
        raise ValueError("Expected 'choices' to be a list.")
    if len(value) != 4:
        raise ValueError(f"Expected exactly 4 choices, got {len(value)}.")
    if not all(isinstance(item, str) for item in value):
        raise ValueError("Expected every item in 'choices' to be a string.")


def validate_answer(value) -> None:
    """Validate the answer field."""
    if not isinstance(value, int):
        raise ValueError(f"Expected 'answer' to be an integer, got {type(value)}.")
    if value not in {0, 1, 2, 3}:
        raise ValueError(f"Expected 'answer' to be one of 0, 1, 2, 3, got {value!r}.")


def answer_to_letter(answer: int) -> str:
    """Convert MMLU integer answer index to answer letter."""
    return ["A", "B", "C", "D"][answer]


def convert_example(example: Dict, subject: str) -> Dict:
    """Convert a raw MMLU example to the target schema."""
    question = get_required_field(example, ["question"], "question")
    choices = get_required_field(example, ["choices"], "choices")
    answer = get_required_field(example, ["answer"], "answer")

    if not isinstance(question, str):
        raise ValueError("Expected 'question' to be a string.")

    validate_choices(choices)
    validate_answer(answer)

    raw_subject = example.get("subject", subject)
    if not isinstance(raw_subject, str):
        raise ValueError("Expected 'subject' to be a string.")

    output = {
        "question": question,
        "choices": choices,
        "answer": answer,
        "answer_letter": answer_to_letter(answer),
        "subject": subject,
        "dataset_name": "MMLU",
    }

    for field in REQUIRED_OUTPUT_FIELDS:
        if field not in output:
            raise ValueError(f"Converted example is missing required field: {field}")

    return output


def load_subject_dataset(subject: str) -> List[Dict]:
    """Download and load the MMLU test split for a given subject."""
    log(f"Loading MMLU subject '{subject}' split '{DATASET_SPLIT}' ...")

    try:
        dataset = load_dataset(DATASET_NAME, subject, split=DATASET_SPLIT)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load {DATASET_NAME!r}, subject {subject!r}, "
            f"split {DATASET_SPLIT!r}: {e}"
        ) from e

    rows = [dict(row) for row in dataset]

    if not rows:
        raise ValueError(f"No rows found for subject {subject!r}.")

    return rows


def sample_and_split_subject(
    rows: List[Dict],
    subject: str,
    rng: random.Random,
) -> Dict[str, List[Dict]]:
    """Sample and split a single subject into train and val."""
    if len(rows) < SAMPLES_PER_SUBJECT:
        raise ValueError(
            f"Requested {SAMPLES_PER_SUBJECT} samples for subject {subject!r}, "
            f"but only found {len(rows)} examples."
        )

    log(f"Sampling {SAMPLES_PER_SUBJECT} examples from subject '{subject}' ...")
    selected_indices = rng.sample(range(len(rows)), SAMPLES_PER_SUBJECT)
    sampled_rows = [rows[i] for i in selected_indices]
    sampled_rows = [convert_example(row, subject) for row in sampled_rows]

    log(f"Shuffling subject '{subject}' sampled examples ...")
    rng.shuffle(sampled_rows)

    train_samples = sampled_rows[:TRAIN_PER_SUBJECT]
    val_samples = sampled_rows[TRAIN_PER_SUBJECT:]

    if len(train_samples) != TRAIN_PER_SUBJECT:
        raise RuntimeError(f"Unexpected train split size for subject {subject!r}.")
    if len(val_samples) != VAL_PER_SUBJECT:
        raise RuntimeError(f"Unexpected validation split size for subject {subject!r}.")

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
    log(f"  DATASET_SPLIT = {DATASET_SPLIT}")
    log(f"  SAMPLES_PER_SUBJECT = {SAMPLES_PER_SUBJECT}")
    log(f"  TRAIN_PER_SUBJECT = {TRAIN_PER_SUBJECT}")
    log(f"  VAL_PER_SUBJECT = {VAL_PER_SUBJECT}")
    log(f"  TOTAL_TRAIN = {TRAIN_PER_SUBJECT * len(SUBJECTS)}")
    log(f"  TOTAL_VAL = {VAL_PER_SUBJECT * len(SUBJECTS)}")
    log(f"  SEED = {SEED}")
    log("")

    rng = random.Random(SEED)

    all_train_samples: List[Dict] = []
    all_val_samples: List[Dict] = []

    for subject in SUBJECTS:
        log(f"=== Processing MMLU subject: {subject} ===")
        subject_rows = load_subject_dataset(subject)
        subject_splits = sample_and_split_subject(subject_rows, subject, rng)

        all_train_samples.extend(subject_splits["train"])
        all_val_samples.extend(subject_splits["val"])

        log(
            f"Subject '{subject}' train split complete: "
            f"{len(subject_splits['train'])} samples"
        )
        log(
            f"Subject '{subject}' val split complete: "
            f"{len(subject_splits['val'])} samples"
        )
        log("")

    log("Shuffling final train and val sets ...")
    rng.shuffle(all_train_samples)
    rng.shuffle(all_val_samples)

    expected_train_size = TRAIN_PER_SUBJECT * len(SUBJECTS)
    expected_val_size = VAL_PER_SUBJECT * len(SUBJECTS)

    if len(all_train_samples) != expected_train_size:
        raise RuntimeError("Unexpected final train split size.")
    if len(all_val_samples) != expected_val_size:
        raise RuntimeError("Unexpected final validation split size.")

    train_subject_counts = {
        subject: sum(1 for row in all_train_samples if row["subject"] == subject)
        for subject in SUBJECTS
    }
    val_subject_counts = {
        subject: sum(1 for row in all_val_samples if row["subject"] == subject)
        for subject in SUBJECTS
    }

    for subject in SUBJECTS:
        if train_subject_counts[subject] != TRAIN_PER_SUBJECT:
            raise RuntimeError(
                f"Unexpected train count for subject {subject!r}: "
                f"{train_subject_counts[subject]}"
            )
        if val_subject_counts[subject] != VAL_PER_SUBJECT:
            raise RuntimeError(
                f"Unexpected val count for subject {subject!r}: "
                f"{val_subject_counts[subject]}"
            )

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