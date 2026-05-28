#!/usr/bin/env python3
"""
Create UniADILR abduction train/val JSON files.

Behavior:
- Downloads the UniADILR abduction.jsonl file directly from the GitHub repo
- Uses only abduction examples
- Randomly samples exactly N examples from the source file
- Splits the sampled examples into train and val according to VAL_SIZE
- Reproducible via fixed seed

Outputs written next to this script:
- uniadilr_abduction_train.json
- uniadilr_abduction_val.json
"""

from __future__ import annotations

import json
import random
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List


# =========================
# User-configurable values
# =========================
N = 400  # Total number of examples to sample
VAL_SIZE = 80  # Number of validation examples
SEED = 42  # Random seed for reproducibility

DATA_URL = (
    "https://raw.githubusercontent.com/YuSheng-00/UniADILR/main/"
    "data/UniADILR-HGc/abduction.jsonl"
)
REASONING_TYPE_FILTER = "abduction"

OUTPUT_DIR = Path(__file__).resolve().parent
DOWNLOADED_INPUT = OUTPUT_DIR / "abduction.jsonl"
TRAIN_OUTPUT = OUTPUT_DIR / "uniadilr_abduction_train.json"
VAL_OUTPUT = OUTPUT_DIR / "uniadilr_abduction_val.json"

REQUIRED_FIELDS = [
    "context",
    "hypothesis",
    "proof",
    "reasoning_type",
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


def download_file(url: str, output_path: Path) -> None:
    """Download a file from a URL to a local path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url) as response:
        if response.status != 200:
            raise ValueError(f"Failed to download data. HTTP status: {response.status}")

        data = response.read()

    output_path.write_bytes(data)


def read_jsonl(path: Path) -> List[Dict]:
    """Read a JSONL file into a list of dictionaries."""
    rows: List[Dict] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse JSON on line {line_number} of {path}: {e}"
                ) from e

            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} of {path}."
                )

            rows.append(row)

    return rows


def validate_example(example: Dict) -> None:
    """Validate that an example contains the expected fields."""
    for field in REQUIRED_FIELDS:
        if field not in example:
            raise ValueError(f"Example is missing required field: {field}")

    if example["reasoning_type"] != REASONING_TYPE_FILTER:
        raise ValueError(
            f"Expected reasoning_type == '{REASONING_TYPE_FILTER}', got "
            f"{example['reasoning_type']!r}."
        )

    if not isinstance(example["context"], dict):
        raise ValueError("Expected 'context' to be a dictionary.")


def convert_examples(rows: List[Dict]) -> List[Dict]:
    """Convert examples to plain dictionaries with the expected keys."""
    output = []
    for row in rows:
        validate_example(row)
        output.append({field: row[field] for field in REQUIRED_FIELDS})
    return output


def write_json(path: Path, data: List[Dict]) -> None:
    """Write data to a JSON file."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    """Main function to create train and validation splits."""
    train_size = validate_config()

    log("Configuration:")
    log(f"  DATA_URL = {DATA_URL}")
    log(f"  REASONING_TYPE_FILTER = {REASONING_TYPE_FILTER}")
    log(f"  N = {N}")
    log(f"  VAL_SIZE = {VAL_SIZE}")
    log(f"  TRAIN_SIZE = {train_size}")
    log(f"  SEED = {SEED}")
    log("")

    log(f"Downloading data from {DATA_URL} ...")
    download_file(DATA_URL, DOWNLOADED_INPUT)

    log(f"Loading data from {DOWNLOADED_INPUT} ...")
    dataset = read_jsonl(DOWNLOADED_INPUT)

    log(f"Filtering to reasoning_type == '{REASONING_TYPE_FILTER}' ...")
    filtered_dataset = [
        row for row in dataset if row.get("reasoning_type") == REASONING_TYPE_FILTER
    ]

    if len(filtered_dataset) < N:
        raise ValueError(
            f"Requested {N} samples after filtering, but only {len(filtered_dataset)} "
            f"examples are available with reasoning_type == '{REASONING_TYPE_FILTER}'."
        )

    log(f"Sampling {N} examples ...")
    rng = random.Random(SEED)
    selected_indices = rng.sample(range(len(filtered_dataset)), N)
    sampled_rows = [filtered_dataset[i] for i in selected_indices]
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
    log(
        f"Created:\n  {DOWNLOADED_INPUT}\n  {TRAIN_OUTPUT}\n  {VAL_OUTPUT}"
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
