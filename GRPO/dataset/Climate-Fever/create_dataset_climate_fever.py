#!/usr/bin/env python3
"""
Create Climate Fever train/val JSON files.

Behavior:
- Loads the tdiggelm/climate_fever dataset from Hugging Face
- Uses only the test split
- Randomly samples exactly N examples from the test split
- Splits the sampled examples into train and val according to VAL_SIZE
- Reproducible via fixed seed

Outputs written next to this script:
- climate_fever_train.json
- climate_fever_val.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset


# =========================
# User-configurable values
# =========================
N = 400  # Total number of examples to sample
VAL_SIZE = 80  # Number of validation examples
SEED = 42  # Random seed for reproducibility

DATASET_NAME = "tdiggelm/climate_fever"
SOURCE_SPLIT = "test"

OUTPUT_DIR = Path(__file__).resolve().parent
TRAIN_OUTPUT = OUTPUT_DIR / "climate_fever_train.json"
VAL_OUTPUT = OUTPUT_DIR / "climate_fever_val.json"

REQUIRED_FIELDS = [
    "claim",
    "evidences",
    "claim_label",
]


def log(msg: str) -> None:
    """Log messages to the console."""
    print(msg, flush=True)


def validate_config() -> int:
    """Validate user-configurable values and calculate train size."""
    if N <= 0:
        raise ValueError("N must be positive.")
    if VAL_SIZE <= 0:
        raise ValueError("VAL_SIZE must be positive.")
    if VAL_SIZE >= N:
        raise ValueError("VAL_SIZE must be smaller than N.")

    return N - VAL_SIZE


def validate_evidence(evidence: Dict) -> None:
    """Validate that an evidence entry contains the expected fields."""
    required_evidence_fields = [
        "evidence_id",
        "evidence_label",
        "article",
        "evidence",
        "entropy",
        "votes",
    ]

    for field in required_evidence_fields:
        if field not in evidence:
            raise ValueError(f"Evidence is missing required field: {field}")


def validate_example(example: Dict) -> None:
    """Validate that an example contains the expected fields."""
    for field in REQUIRED_FIELDS:
        if field not in example:
            raise ValueError(f"Example is missing required field: {field}")

    if not isinstance(example["evidences"], list):
        raise ValueError("Expected 'evidences' to be a list.")

    for evidence in example["evidences"]:
        validate_evidence(evidence)


def convert_examples(rows: List[Dict]) -> List[Dict]:
    """Convert examples to plain dictionaries with the expected keys."""
    output = []
    for row in rows:
        validate_example(row)
        output.append(
            {
                "claim": row["claim"],
                "evidences": [
                    {
                        "evidence_id": evidence["evidence_id"],
                        "evidence_label": evidence["evidence_label"],
                        "article": evidence["article"],
                        "evidence": evidence["evidence"],
                        "entropy": evidence["entropy"],
                        "votes": evidence["votes"],
                    }
                    for evidence in row["evidences"]
                ],
                "claim_label": row["claim_label"],
            }
        )
    return output


def write_json(path: Path, data: List[Dict]) -> None:
    """Write data to a JSON file."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    """Main function to create train and validation splits."""
    train_size = validate_config()

    log("Configuration:")
    log(f"  DATASET_NAME = {DATASET_NAME}")
    log(f"  SOURCE_SPLIT = {SOURCE_SPLIT}")
    log(f"  N = {N}")
    log(f"  VAL_SIZE = {VAL_SIZE}")
    log(f"  TRAIN_SIZE = {train_size}")
    log(f"  SEED = {SEED}")
    log("")

    log(f"Loading dataset {DATASET_NAME}/{SOURCE_SPLIT} ...")
    dataset = load_dataset(DATASET_NAME, split=SOURCE_SPLIT)

    if len(dataset) < N:
        raise ValueError(
            f"Requested {N} samples, but only {len(dataset)} examples are available "
            f"in split '{SOURCE_SPLIT}'."
        )

    log(f"Sampling {N} examples ...")
    rng = random.Random(SEED)
    selected_indices = rng.sample(range(len(dataset)), N)
    sampled_dataset = dataset.select(selected_indices)

    sampled_rows = [dict(row) for row in sampled_dataset]
    sampled_rows = convert_examples(sampled_rows)

    log("Shuffling sampled examples ...")
    rng.shuffle(sampled_rows)

    train_samples = sampled_rows[:train_size]
    val_samples = sampled_rows[train_size:]

    if len(train_samples) != train_size:
        raise RuntimeError("Unexpected train split size.")
    if len(val_samples) != VAL_SIZE:
        raise RuntimeError("Unexpected validation split size.")

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