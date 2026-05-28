#!/usr/bin/env python3
"""
Create VitaminC train/val JSON files.

Behavior:
- Loads the tals/vitaminc dataset from Hugging Face
- Uses the train split for train samples
- Uses the validation split for validation samples
- Randomly samples exactly TRAIN_SIZE examples from train
- Randomly samples exactly VAL_SIZE examples from validation
- Tries to keep each sampled split stratified/balanced across labels
- Reproducible via fixed seed

Outputs written next to this script:
- vitaminc_train.json
- vitaminc_val.json
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from datasets import load_dataset


# =========================
# User-configurable values
# =========================
TRAIN_SIZE = 320  # Number of examples to sample from the train split
VAL_SIZE = 80  # Number of examples to sample from the validation split
SEED = 42  # Random seed for reproducibility

DATASET_NAME = "tals/vitaminc"
TRAIN_SOURCE_SPLIT = "train"
VAL_SOURCE_SPLIT = "validation"

LABEL_FIELD = "label"

OUTPUT_DIR = Path(__file__).resolve().parent
TRAIN_OUTPUT = OUTPUT_DIR / "vitaminc_train.json"
VAL_OUTPUT = OUTPUT_DIR / "vitaminc_val.json"

REQUIRED_FIELDS = [
    "unique_id",
    "case_id",
    "wiki_revision_id",
    "label",
    "claim",
    "evidence",
    "page",
    "revision_type",
    "FEVER_id",
    "big_bench_canary",
]


def log(msg: str) -> None:
    """Log messages to the console."""
    print(msg, flush=True)


def validate_config() -> None:
    """Validate user-configurable values."""
    if TRAIN_SIZE <= 0:
        raise ValueError("TRAIN_SIZE must be positive.")
    if VAL_SIZE <= 0:
        raise ValueError("VAL_SIZE must be positive.")
    if SEED < 0:
        raise ValueError("SEED must be non-negative.")


def validate_example(example: Dict) -> None:
    """Validate that an example contains the expected fields."""
    for field in REQUIRED_FIELDS:
        if field not in example:
            raise ValueError(f"Example is missing required field: {field}")

    if not isinstance(example[LABEL_FIELD], str):
        raise ValueError(f"Expected '{LABEL_FIELD}' to be a string.")


def convert_examples(rows: List[Dict]) -> List[Dict]:
    """Convert examples to plain dictionaries with the expected keys."""
    output = []
    for row in rows:
        validate_example(row)
        output.append(
            {
                "unique_id": row["unique_id"],
                "case_id": row["case_id"],
                "wiki_revision_id": row["wiki_revision_id"],
                "label": row["label"],
                "claim": row["claim"],
                "evidence": row["evidence"],
                "page": row["page"],
                "revision_type": row["revision_type"],
                "FEVER_id": row["FEVER_id"],
                "big_bench_canary": row["big_bench_canary"],
            }
        )
    return output


def get_stratified_quotas(rows: List[Dict], n: int) -> Dict[str, int]:
    """
    Create quotas that approximately preserve the original label distribution.

    Example:
    If the source split is:
      SUPPORTS: 50%
      REFUTES: 30%
      NOT ENOUGH INFO: 20%

    and n=320, this returns approximately:
      SUPPORTS: 160
      REFUTES: 96
      NOT ENOUGH INFO: 64
    """
    label_counts = Counter(row[LABEL_FIELD] for row in rows)
    total = sum(label_counts.values())

    if total == 0:
        raise ValueError("No rows found when calculating label quotas.")

    raw_quotas = {
        label: (count / total) * n
        for label, count in label_counts.items()
    }

    quotas = {
        label: int(raw_quota)
        for label, raw_quota in raw_quotas.items()
    }

    assigned = sum(quotas.values())
    remainder = n - assigned

    # Add leftover samples to labels with the largest fractional parts.
    fractional_parts = sorted(
        raw_quotas.items(),
        key=lambda item: item[1] - int(item[1]),
        reverse=True,
    )

    for label, _ in fractional_parts[:remainder]:
        quotas[label] += 1

    return quotas


def stratified_sample(rows: List[Dict], n: int, rng: random.Random) -> List[Dict]:
    """
    Sample n examples while trying to stay balanced across labels.

    If one label does not have enough rows, the remaining slots are filled from
    the other labels without duplicates.
    """
    if len(rows) < n:
        raise ValueError(
            f"Requested {n} samples, but only {len(rows)} examples are available."
        )

    rows_by_label = defaultdict(list)
    for row in rows:
        validate_example(row)
        rows_by_label[row[LABEL_FIELD]].append(row)

    labels = sorted(rows_by_label.keys())
    quotas = get_stratified_quotas(rows, n)

    selected = []
    selected_ids = set()

    log("Label counts before sampling:")
    for label in labels:
        log(f"  {label}: {len(rows_by_label[label])}")

    log("Target label quotas, preserving original split ratios:")
    for label in labels:
        log(f"  {label}: {quotas[label]}")

    for label in labels:
        label_rows = rows_by_label[label][:]
        rng.shuffle(label_rows)

        quota = quotas[label]
        take = min(quota, len(label_rows))

        if take < quota:
            log(
                f"Warning: label '{label}' only has {len(label_rows)} rows; "
                f"wanted {quota}. Filling remaining slots from other labels."
            )

        for row in label_rows[:take]:
            selected.append(row)
            selected_ids.add(row["unique_id"])

    if len(selected) < n:
        remaining_rows = [
            row for row in rows
            if row["unique_id"] not in selected_ids
        ]
        rng.shuffle(remaining_rows)

        needed = n - len(selected)
        selected.extend(remaining_rows[:needed])

    if len(selected) != n:
        raise RuntimeError(
            f"Unexpected sample size: got {len(selected)}, expected {n}."
        )

    rng.shuffle(selected)

    log("Label counts after sampling:")
    sampled_counts = Counter(row[LABEL_FIELD] for row in selected)
    for label in sorted(sampled_counts):
        log(f"  {label}: {sampled_counts[label]}")

    return selected


def write_json(path: Path, data: List[Dict]) -> None:
    """Write data to a JSON file."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    """Main function to create train and validation samples."""
    validate_config()

    log("Configuration:")
    log(f"  DATASET_NAME = {DATASET_NAME}")
    log(f"  TRAIN_SOURCE_SPLIT = {TRAIN_SOURCE_SPLIT}")
    log(f"  VAL_SOURCE_SPLIT = {VAL_SOURCE_SPLIT}")
    log(f"  TRAIN_SIZE = {TRAIN_SIZE}")
    log(f"  VAL_SIZE = {VAL_SIZE}")
    log(f"  SEED = {SEED}")
    log("")

    rng = random.Random(SEED)

    log(f"Loading dataset {DATASET_NAME}/{TRAIN_SOURCE_SPLIT} ...")
    train_dataset = load_dataset(DATASET_NAME, split=TRAIN_SOURCE_SPLIT)

    log(f"Loading dataset {DATASET_NAME}/{VAL_SOURCE_SPLIT} ...")
    val_dataset = load_dataset(DATASET_NAME, split=VAL_SOURCE_SPLIT)

    train_rows = [dict(row) for row in train_dataset]
    val_rows = [dict(row) for row in val_dataset]

    log("")
    log(f"Sampling {TRAIN_SIZE} examples from '{TRAIN_SOURCE_SPLIT}' ...")
    train_samples = stratified_sample(train_rows, TRAIN_SIZE, rng)
    train_samples = convert_examples(train_samples)

    log("")
    log(f"Sampling {VAL_SIZE} examples from '{VAL_SOURCE_SPLIT}' ...")
    val_samples = stratified_sample(val_rows, VAL_SIZE, rng)
    val_samples = convert_examples(val_samples)

    if len(train_samples) != TRAIN_SIZE:
        raise RuntimeError("Unexpected train split size.")
    if len(val_samples) != VAL_SIZE:
        raise RuntimeError("Unexpected validation split size.")

    log("")
    log(f"Train split complete: {len(train_samples)} samples")
    log(f"Val split complete: {len(val_samples)} samples")
    log("")

    log("Writing output files...")
    write_json(TRAIN_OUTPUT, train_samples)
    write_json(VAL_OUTPUT, val_samples)

    log("Done.")
    log(f"Created:\n  {TRAIN_OUTPUT}\n  {VAL_OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr, flush=True)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr, flush=True)
        sys.exit(1)