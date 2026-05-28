#!/usr/bin/env python3

from __future__ import annotations

import json
import random
import sys
from pathlib import Path


N_PER_DATASET = 100
VAL_SIZE = 40
SEED = 42

OUTPUT_DIR = Path(__file__).resolve().parent
DATASET_FILES = {
    "caesar": OUTPUT_DIR / "caesar.jsonl",
    "atbash": OUTPUT_DIR / "atbash.jsonl",
}

TRAIN_OUTPUT = OUTPUT_DIR / "crypto_train.json"
VAL_OUTPUT = OUTPUT_DIR / "crypto_val.json"

REQUIRED_FIELDS = ["train", "test", "index"]


def log(msg: str) -> None:
    print(msg, flush=True)


def validate_config() -> int:
    total_samples = len(DATASET_FILES) * N_PER_DATASET

    if N_PER_DATASET <= 0:
        raise ValueError("N_PER_DATASET must be positive.")
    if VAL_SIZE <= 0:
        raise ValueError("VAL_SIZE must be positive.")
    if VAL_SIZE >= total_samples:
        raise ValueError("VAL_SIZE must be smaller than the total number of samples.")

    return total_samples - VAL_SIZE


def validate_example(example: dict) -> None:
    for field in REQUIRED_FIELDS:
        if field not in example:
            raise ValueError(f"Example is missing required field: {field}")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def convert_examples(rows: list[dict], split_name: str) -> list[dict]:
    output = []
    for row in rows:
        validate_example(row)
        item = {field: row[field] for field in REQUIRED_FIELDS}
        item["split"] = split_name
        output.append(item)
    return output


def sample_dataset_rows(name: str, path: Path, rng: random.Random) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    log(f"Loading dataset {name} from {path.name} ...")
    rows = read_jsonl(path)
    rows = convert_examples(rows, name)

    if len(rows) < N_PER_DATASET:
        raise ValueError(
            f"Requested {N_PER_DATASET} samples from {name}, but only {len(rows)} examples are available."
        )

    log(f"Sampling {N_PER_DATASET} examples from {name} ...")
    selected_indices = rng.sample(range(len(rows)), N_PER_DATASET)
    return [rows[i] for i in selected_indices]


def write_json(path: Path, data: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    train_size = validate_config()
    total_samples = len(DATASET_FILES) * N_PER_DATASET

    log("Configuration:")
    log(f"  DATASETS = {', '.join(DATASET_FILES.keys())}")
    log(f"  N_PER_DATASET = {N_PER_DATASET}")
    log(f"  TOTAL_SAMPLES = {total_samples}")
    log(f"  VAL_SIZE = {VAL_SIZE}")
    log(f"  TRAIN_SIZE = {train_size}")
    log(f"  SEED = {SEED}")
    log("")

    rng = random.Random(SEED)

    sampled_rows = []
    for name, path in DATASET_FILES.items():
        sampled_rows.extend(sample_dataset_rows(name, path, rng))

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
