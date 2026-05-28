#!/usr/bin/env python3
"""
Create List Function train/val JSON files from Robust-Rule-Induction.

Behavior:
- Downloads the list_functions JSONL file directly from GitHub
- Uses datasets/list_functions/list_functions.jsonl as the source dataset
- Randomly samples exactly N examples from the full dataset
- Converts each source JSONL row into the expected task format:
  {
    "idx": <int>,
    "id": <source id>,
    "function": <source function>,
    "train": [{"input": "[...]", "output": "[...]"}, ...],
    "test":  [{"input": "[...]", "output": "[...]"}, ...]
  }
- Uses only train["normal"] and test demonstrations
- Ignores train["ood"] and train["noise"]
- Splits the sampled examples into train and val according to VAL_SIZE
- Reproducible via fixed seed

Outputs written next to this script:
- list_function_train.json
- list_function_val.json

Note:
- The source file currently contains 250 JSONL rows.
- Each row is expected to contain 10 train["normal"] demonstrations and 10 test
  demonstrations. This script validates that at least those counts exist and uses
  exactly the first 10 from each.
"""

from __future__ import annotations

import json
import random
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


# =========================
# User-configurable values
# =========================
N = 200  # Total number of examples to sample
VAL_SIZE = 40  # Number of validation examples
SEED = 42  # Random seed for reproducibility
TRAIN_DEMO_COUNT = 10  # Number of demonstrations stored in each sample's "train"
TEST_DEMO_COUNT = 10  # Number of demonstrations stored in each sample's "test"

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "HKUST-KnowComp/Robust-Rule-Induction/"
    "refs/heads/main/datasets/list_functions/list_functions.jsonl"
)

OUTPUT_DIR = Path(__file__).resolve().parent
DOWNLOADED_DATASET = OUTPUT_DIR / "list_functions.jsonl"
TRAIN_OUTPUT = OUTPUT_DIR / "list_function_train.json"
VAL_OUTPUT = OUTPUT_DIR / "list_function_val.json"

REQUIRED_TOP_LEVEL_FIELDS = [
    "id",
    "function",
    "train",
    "test",
]

REQUIRED_IO_FIELDS = [
    "input",
    "output",
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
    if TRAIN_DEMO_COUNT <= 0:
        raise ValueError("TRAIN_DEMO_COUNT must be positive.")
    if TEST_DEMO_COUNT <= 0:
        raise ValueError("TEST_DEMO_COUNT must be positive.")

    return N - VAL_SIZE


def download_file(url: str, output_path: Path) -> None:
    """Download a file from a URL to a local path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "create-dataset-list-function/1.0",
        },
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        status = getattr(response, "status", None)
        if status is not None and status != 200:
            raise ValueError(f"Failed to download data. HTTP status: {status}")

        data = response.read()

    if not data:
        raise ValueError("Downloaded dataset is empty.")

    output_path.write_bytes(data)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse JSON from {path}, line {line_number}: {e}"
                ) from e

            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path}, line {line_number}.")

            rows.append(row)

    if not rows:
        raise ValueError(f"No JSONL rows found in {path}.")

    return rows


def validate_io_pair(pair: Dict[str, Any], source_name: str) -> None:
    """Validate that one input/output pair contains the expected fields."""
    if not isinstance(pair, dict):
        raise ValueError(f"Expected each demonstration to be a dictionary in {source_name}.")

    for field in REQUIRED_IO_FIELDS:
        if field not in pair:
            raise ValueError(f"A demonstration in {source_name} is missing field: {field}")

    if not isinstance(pair["input"], list):
        raise ValueError(f"Expected demonstration['input'] to be a list in {source_name}.")
    if not isinstance(pair["output"], list):
        raise ValueError(f"Expected demonstration['output'] to be a list in {source_name}.")


def validate_source_example(example: Dict[str, Any], source_name: str) -> None:
    """Validate that a source JSONL row contains the expected fields."""
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in example:
            raise ValueError(f"{source_name} is missing required field: {field}")

    if not isinstance(example["id"], str):
        raise ValueError(f"Expected 'id' to be a string in {source_name}.")
    if not isinstance(example["function"], str):
        raise ValueError(f"Expected 'function' to be a string in {source_name}.")
    if not isinstance(example["train"], dict):
        raise ValueError(f"Expected 'train' to be a dictionary in {source_name}.")
    if "normal" not in example["train"]:
        raise ValueError(f"{source_name} is missing train['normal'].")
    if not isinstance(example["train"]["normal"], list):
        raise ValueError(f"Expected train['normal'] to be a list in {source_name}.")
    if not isinstance(example["test"], list):
        raise ValueError(f"Expected 'test' to be a list in {source_name}.")

    normal_train = example["train"]["normal"]
    test = example["test"]

    if len(normal_train) < TRAIN_DEMO_COUNT:
        raise ValueError(
            f"Expected at least {TRAIN_DEMO_COUNT} train['normal'] demonstrations "
            f"in {source_name}, but found {len(normal_train)}."
        )
    if len(test) < TEST_DEMO_COUNT:
        raise ValueError(
            f"Expected at least {TEST_DEMO_COUNT} test demonstrations "
            f"in {source_name}, but found {len(test)}."
        )

    for demo_index, pair in enumerate(normal_train[:TRAIN_DEMO_COUNT]):
        validate_io_pair(pair, f"{source_name} train['normal'][{demo_index}]")
    for demo_index, pair in enumerate(test[:TEST_DEMO_COUNT]):
        validate_io_pair(pair, f"{source_name} test[{demo_index}]")


def format_io_pair(pair: Dict[str, Any]) -> Dict[str, str]:
    """Convert one source pair into the expected input/output string format."""
    return {
        "input": json.dumps(pair["input"]),
        "output": json.dumps(pair["output"]),
    }


def convert_example(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Convert one source JSONL row into the target sample format."""
    normal_train = example["train"]["normal"]
    test = example["test"]

    train_pairs = [
        format_io_pair(pair) for pair in normal_train[:TRAIN_DEMO_COUNT]
    ]
    test_pairs = [
        format_io_pair(pair) for pair in test[:TEST_DEMO_COUNT]
    ]

    if len(train_pairs) != TRAIN_DEMO_COUNT:
        raise RuntimeError("Converted example has an unexpected train split size.")
    if len(test_pairs) != TEST_DEMO_COUNT:
        raise RuntimeError("Converted example has an unexpected test split size.")

    return {
        "idx": idx,
        "id": example["id"],
        "function": example["function"],
        "train": train_pairs,
        "test": test_pairs,
    }


def convert_examples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert source JSONL rows into the target task format."""
    output: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        source_name = f"JSONL row {idx}"
        validate_source_example(row, source_name)
        output.append(convert_example(row, idx=idx))

    return output


def write_json(path: Path, data: List[Dict[str, Any]]) -> None:
    """Write data to a JSON file."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    """Main function to create train and validation splits."""
    train_size = validate_config()

    log("Configuration:")
    log(f"  DATA_URL = {DATA_URL}")
    log(f"  N = {N}")
    log(f"  VAL_SIZE = {VAL_SIZE}")
    log(f"  TRAIN_SIZE = {train_size}")
    log(f"  TRAIN_DEMO_COUNT = {TRAIN_DEMO_COUNT}")
    log(f"  TEST_DEMO_COUNT = {TEST_DEMO_COUNT}")
    log(f"  SEED = {SEED}")
    log("")

    log(f"Downloading data from {DATA_URL} ...")
    download_file(DATA_URL, DOWNLOADED_DATASET)

    log("Reading JSONL dataset ...")
    rows = read_jsonl(DOWNLOADED_DATASET)

    if len(rows) < N:
        raise ValueError(
            f"Requested {N} samples, but only {len(rows)} rows were found."
        )

    log(f"Sampling {N} examples from the full dataset ...")
    rng = random.Random(SEED)
    selected_indices = rng.sample(range(len(rows)), N)
    sampled_rows = [rows[i] for i in selected_indices]

    log("Converting sampled examples ...")
    sampled_examples = convert_examples(sampled_rows)

    log("Shuffling sampled examples ...")
    rng.shuffle(sampled_examples)

    for idx, row in enumerate(sampled_examples):
        row["idx"] = idx

    train_samples = sampled_examples[:train_size]
    val_samples = sampled_examples[train_size:]

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
        f"Created:\n  {DOWNLOADED_DATASET}\n  {TRAIN_OUTPUT}\n  {VAL_OUTPUT}"
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
