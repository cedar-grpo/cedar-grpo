#!/usr/bin/env python3
"""
Create CauseLogics train/val JSON files.

Behavior:
- Downloads CauseLogics JSONL files directly from the GitHub repo
- Uses exactly 100 random samples from each level (1, 2, 3, 4)
- Splits each level into:
    - 80 train
    - 20 val
- Produces:
    - 320 training examples total
    - 80 validation examples total
- Reproducible via fixed seed

Outputs written next to this script:
- causelogics_train.json
- causelogics_val.json
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
SAMPLES_PER_LEVEL = 100
TRAIN_PER_LEVEL = 80
VAL_PER_LEVEL = 20
SEED = 42

LEVELS = [1, 2, 3, 4]

BASE_URL = "https://raw.githubusercontent.com/sternstude/CauseJudger/main/CauseLogics"

LEVEL_TO_URLS = {
    1: [
        f"{BASE_URL}/Level%201/CauseLogics_Level_1_1.jsonl",
        f"{BASE_URL}/Level%201/CauseLogics_Level_1_2.jsonl",
        f"{BASE_URL}/Level%201/CauseLogics_Level_1_3.jsonl",
    ],
    2: [
        f"{BASE_URL}/Level%202/CauseLogics_Level_2_1.jsonl",
        f"{BASE_URL}/Level%202/CauseLogics_Level_2_2.jsonl",
        f"{BASE_URL}/Level%202/CauseLogics_Level_2_3.jsonl",
    ],
    3: [
        f"{BASE_URL}/Level%203/CauseLogics_Level_3_1.jsonl",
        f"{BASE_URL}/Level%203/CauseLogics_Level_3_2.jsonl",
        f"{BASE_URL}/Level%203/CauseLogics_Level_3_3.jsonl",
    ],
    4: [
        f"{BASE_URL}/Level%204/CauseLogics_Level_4_1.jsonl",
        f"{BASE_URL}/Level%204/CauseLogics_Level_4_2.jsonl",
        f"{BASE_URL}/Level%204/CauseLogics_Level_4_3.jsonl",
    ],
}

OUTPUT_DIR = Path(__file__).resolve().parent
DOWNLOADED_DIR = OUTPUT_DIR / "causelogics_downloaded"
TRAIN_OUTPUT = OUTPUT_DIR / "causelogics_train.json"
VAL_OUTPUT = OUTPUT_DIR / "causelogics_val.json"

REQUIRED_OUTPUT_FIELDS = [
    "Premises",
    "Rules",
    "Phenomenon",
    "PossibleCause",
    "Label",
    "dataset_name",
    "causelogics_level",
]


def log(msg: str) -> None:
    """Log messages to the console."""
    print(msg, flush=True)


def validate_config() -> None:
    """Validate user-configurable values."""
    if SAMPLES_PER_LEVEL <= 0:
        raise ValueError("SAMPLES_PER_LEVEL must be positive.")
    if TRAIN_PER_LEVEL <= 0:
        raise ValueError("TRAIN_PER_LEVEL must be positive.")
    if VAL_PER_LEVEL <= 0:
        raise ValueError("VAL_PER_LEVEL must be positive.")
    if TRAIN_PER_LEVEL + VAL_PER_LEVEL != SAMPLES_PER_LEVEL:
        raise ValueError(
            "TRAIN_PER_LEVEL + VAL_PER_LEVEL must equal SAMPLES_PER_LEVEL."
        )


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


def get_required_field(example: Dict, field_names: List[str], label: str):
    """Return the first matching field from field_names."""
    for field_name in field_names:
        if field_name in example:
            return example[field_name]

    raise ValueError(f"Example is missing required field for {label}: {field_names}")


def validate_list_of_strings(value, field_name: str) -> None:
    """Validate that value is a list of strings."""
    if not isinstance(value, list):
        raise ValueError(f"Expected '{field_name}' to be a list.")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"Expected every item in '{field_name}' to be a string.")


def validate_label(value) -> None:
    """Validate the label field."""
    if not isinstance(value, str):
        raise ValueError("Expected 'Label' to be a string.")

    if value not in {"True", "False"}:
        raise ValueError(f"Expected Label to be 'True' or 'False', got {value!r}.")


def convert_example(example: Dict, level: int) -> Dict:
    """Convert a raw CauseLogics example to the target schema."""
    premises = get_required_field(example, ["Premises"], "Premises")
    rules = get_required_field(example, ["Rules"], "Rules")
    phenomenon = get_required_field(example, ["Phenomenon"], "Phenomenon")
    possible_cause = get_required_field(
        example,
        ["PossibleCause", "Possible Cause"],
        "PossibleCause",
    )
    label = get_required_field(example, ["Label"], "Label")

    validate_list_of_strings(premises, "Premises")
    validate_list_of_strings(rules, "Rules")

    if not isinstance(phenomenon, str):
        raise ValueError("Expected 'Phenomenon' to be a string.")

    if not isinstance(possible_cause, str):
        raise ValueError("Expected 'PossibleCause' to be a string.")

    validate_label(label)

    output = {
        "Premises": premises,
        "Rules": rules,
        "Phenomenon": phenomenon,
        "PossibleCause": possible_cause,
        "Label": label,
        "dataset_name": "CauseLogics",
        "causelogics_level": level,
    }

    for field in REQUIRED_OUTPUT_FIELDS:
        if field not in output:
            raise ValueError(f"Converted example is missing required field: {field}")

    return output


def load_level_dataset(level: int) -> List[Dict]:
    """Download and load all JSONL files for a given CauseLogics level."""
    if level not in LEVEL_TO_URLS:
        raise ValueError(f"Unsupported level: {level}")

    urls = LEVEL_TO_URLS[level]
    rows: List[Dict] = []

    for url in urls:
        filename = Path(url).name
        local_path = DOWNLOADED_DIR / f"level_{level}" / filename

        log(f"Downloading Level {level} file: {url}")
        download_file(url, local_path)

        log(f"Loading Level {level} file: {local_path}")
        rows.extend(read_jsonl(local_path))

    return rows


def sample_and_split_level(rows: List[Dict], level: int, rng: random.Random) -> Dict[str, List[Dict]]:
    """Sample and split a single level into train and val."""
    if len(rows) < SAMPLES_PER_LEVEL:
        raise ValueError(
            f"Requested {SAMPLES_PER_LEVEL} samples for level {level}, "
            f"but only found {len(rows)} examples."
        )

    log(f"Sampling {SAMPLES_PER_LEVEL} examples from Level {level} ...")
    selected_indices = rng.sample(range(len(rows)), SAMPLES_PER_LEVEL)
    sampled_rows = [rows[i] for i in selected_indices]
    sampled_rows = [convert_example(row, level) for row in sampled_rows]

    log(f"Shuffling Level {level} sampled examples ...")
    rng.shuffle(sampled_rows)

    train_samples = sampled_rows[:TRAIN_PER_LEVEL]
    val_samples = sampled_rows[TRAIN_PER_LEVEL:]

    if len(train_samples) != TRAIN_PER_LEVEL:
        raise RuntimeError(f"Unexpected train split size for Level {level}.")
    if len(val_samples) != VAL_PER_LEVEL:
        raise RuntimeError(f"Unexpected validation split size for Level {level}.")

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
    log(f"  SAMPLES_PER_LEVEL = {SAMPLES_PER_LEVEL}")
    log(f"  TRAIN_PER_LEVEL = {TRAIN_PER_LEVEL}")
    log(f"  VAL_PER_LEVEL = {VAL_PER_LEVEL}")
    log(f"  TOTAL_TRAIN = {TRAIN_PER_LEVEL * len(LEVELS)}")
    log(f"  TOTAL_VAL = {VAL_PER_LEVEL * len(LEVELS)}")
    log(f"  SEED = {SEED}")
    log("")

    rng = random.Random(SEED)

    all_train_samples: List[Dict] = []
    all_val_samples: List[Dict] = []

    for level in LEVELS:
        log(f"=== Processing CauseLogics Level {level} ===")
        level_rows = load_level_dataset(level)
        level_splits = sample_and_split_level(level_rows, level, rng)

        all_train_samples.extend(level_splits["train"])
        all_val_samples.extend(level_splits["val"])

        log(f"Level {level} train split complete: {len(level_splits['train'])} samples")
        log(f"Level {level} val split complete: {len(level_splits['val'])} samples")
        log("")

    log("Shuffling final train and val sets ...")
    rng.shuffle(all_train_samples)
    rng.shuffle(all_val_samples)

    expected_train_size = TRAIN_PER_LEVEL * len(LEVELS)
    expected_val_size = VAL_PER_LEVEL * len(LEVELS)

    if len(all_train_samples) != expected_train_size:
        raise RuntimeError("Unexpected final train split size.")
    if len(all_val_samples) != expected_val_size:
        raise RuntimeError("Unexpected final validation split size.")

    train_level_counts = {
        level: sum(1 for row in all_train_samples if row["causelogics_level"] == level)
        for level in LEVELS
    }
    val_level_counts = {
        level: sum(1 for row in all_val_samples if row["causelogics_level"] == level)
        for level in LEVELS
    }

    for level in LEVELS:
        if train_level_counts[level] != TRAIN_PER_LEVEL:
            raise RuntimeError(
                f"Unexpected train count for level {level}: {train_level_counts[level]}"
            )
        if val_level_counts[level] != VAL_PER_LEVEL:
            raise RuntimeError(
                f"Unexpected val count for level {level}: {val_level_counts[level]}"
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