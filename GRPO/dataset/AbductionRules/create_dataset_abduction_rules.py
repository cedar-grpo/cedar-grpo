#!/usr/bin/env python3
"""
Create balanced AbductionRules train/val JSON files.

Behavior:
- Downloads only the needed JSONL files from GitHub raw URLs
- TRAIN is sampled from train.jsonl
- VAL is sampled from dev.jsonl
- Exactly N/4 contexts from each of:
    * Abduction-Animal-Simple
    * Abduction-Animal
    * Abduction-Person-Simple
    * Abduction-Person
- Validation gets exactly (N/4) * VAL_SPLIT contexts from each dataset
- No duplicate contexts within a split, where context identity is (dataset_name, id)
- For each chosen context, one query is sampled uniformly from the available Q1, Q2, ...
  for that example
- Reproducible via fixed seed

Outputs written next to this script:
- abduction_rules_train.json
- abduction_rules_val.json
"""

from __future__ import annotations

import json
import random
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple


# =========================
# User-configurable values
# =========================
N = 320  # Total number of contexts to sample
VAL_SPLIT = 0.25  # Proportion of validation split
SEED = 42  # Random seed for reproducibility

BASE_RAW_URL = "https://raw.githubusercontent.com/Strong-AI-Lab/AbductionRules/main/datasets"

DATASET_DIRS = [
    "Abduction-Animal-Simple",
    "Abduction-Animal",
    "Abduction-Person-Simple",
    "Abduction-Person",
]

OUTPUT_DIR = Path(__file__).resolve().parent
TRAIN_OUTPUT = OUTPUT_DIR / "abduction_rules_train.json"
VAL_OUTPUT = OUTPUT_DIR / "abduction_rules_val.json"

def log(msg: str) -> None:
    """Log messages to the console."""
    print(msg, flush=True)

def validate_config() -> Tuple[int, int]:
    """Validate user-configurable values and calculate per-dataset counts."""
    if N <= 0:
        raise ValueError("N must be positive.")
    if N % 4 != 0:
        raise ValueError("N must be divisible by 4.")
    if not (0 < VAL_SPLIT < 1):
        raise ValueError("VAL_SPLIT must be between 0 and 1.")

    per_dataset_train = N // 4
    per_dataset_val_float = per_dataset_train * VAL_SPLIT

    if abs(per_dataset_val_float - round(per_dataset_val_float)) > 1e-9:
        raise ValueError(
            f"(N/4) * VAL_SPLIT must be an integer. "
            f"Got {per_dataset_val_float} from N={N}, VAL_SPLIT={VAL_SPLIT}."
        )

    return per_dataset_train, int(round(per_dataset_val_float))

def download_text(url: str) -> str:
    """Download text content from a URL."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Failed to download {url}: HTTP {resp.status}")
        return resp.read().decode("utf-8")

def read_jsonl_from_url(url: str) -> List[Dict]:
    """Read and parse JSONL data from a URL."""
    text = download_text(url)
    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {url} on line {line_no}") from e
    return rows

def parse_q_number(question_id: str) -> int | None:
    """Extract question number from IDs ending in '-Qk'."""
    if not isinstance(question_id, str):
        return None
    marker = "-Q"
    if marker not in question_id:
        return None
    suffix = question_id.rsplit(marker, 1)[1]
    if not suffix.isdigit():
        return None
    return int(suffix)

def get_available_questions(example: Dict) -> List[Tuple[int, Dict]]:
    """Return available questions as a sorted list of (qnum, question_dict)."""
    questions = example.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"Example {example.get('id')} has no valid 'questions' list.")

    q_map = {}
    for q in questions:
        qid = q.get("id")
        qnum = parse_q_number(qid)
        if qnum is not None:
            q_map[qnum] = q

    if not q_map:
        raise ValueError(
            f"Example {example.get('id')} does not contain any parseable Q-numbered questions."
        )

    return sorted(q_map.items(), key=lambda x: x[0])

def choose_one_question(example: Dict, rng: random.Random) -> Dict:
    """Randomly select one question from the available questions."""
    if "id" not in example or "context" not in example:
        raise ValueError("Each example must contain 'id' and 'context'.")

    available_questions = get_available_questions(example)
    chosen_qnum, chosen_q = rng.choice(available_questions)

    return {
        "context_id": example["id"],
        "context": example["context"],
        "query_id": chosen_q.get("id"),
        "query": chosen_q.get("text"),
        "answer": chosen_q.get("label"),
        "qcat": chosen_q.get("QCat"),
        "question_number": chosen_qnum,
    }

def dedupe_examples_by_id(rows: List[Dict]) -> List[Dict]:
    """Remove duplicate examples based on their IDs."""
    seen = set()
    unique_rows = []
    for row in rows:
        row_id = row.get("id")
        if row_id is None:
            raise ValueError("Encountered row without 'id'.")
        if row_id not in seen:
            seen.add(row_id)
            unique_rows.append(row)
    return unique_rows

def sample_from_dataset_split(
    dataset_name: str,
    source_filename: str,
    count: int,
    rng: random.Random,
) -> List[Dict]:
    """Sample a specified number of examples from a dataset split."""
    url = f"{BASE_RAW_URL}/{dataset_name}/{source_filename}"
    log(f"Downloading {dataset_name}/{source_filename} ...")

    rows = read_jsonl_from_url(url)
    rows = dedupe_examples_by_id(rows)

    if count > len(rows):
        raise ValueError(
            f"Requested {count} samples from {dataset_name}/{source_filename}, "
            f"but only {len(rows)} unique context IDs exist."
        )

    selected = rng.sample(rows, count)

    output = []
    for ex in selected:
        item = choose_one_question(ex, rng)
        item["source_dataset"] = dataset_name
        output.append(item)

    keys = [(x["source_dataset"], x["context_id"]) for x in output]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"Duplicate sampled contexts detected in {dataset_name}.")

    return output

def build_split(
    source_filename: str,
    per_dataset_count: int,
    rng: random.Random,
) -> List[Dict]:
    """Build a dataset split by sampling from all datasets."""
    all_samples = []

    for dataset_name in DATASET_DIRS:
        sampled = sample_from_dataset_split(
            dataset_name=dataset_name,
            source_filename=source_filename,
            count=per_dataset_count,
            rng=rng,
        )
        log(f"  sampled {len(sampled)} examples from {dataset_name}")
        all_samples.extend(sampled)

    rng.shuffle(all_samples)

    keys = [(x["source_dataset"], x["context_id"]) for x in all_samples]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate (dataset, context_id) detected in final split.")

    return all_samples

def write_json(path: Path, data: List[Dict]) -> None:
    """Write data to a JSON file."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main() -> None:
    """Main function to create train and validation splits."""
    per_dataset_train, per_dataset_val = validate_config()

    log("Configuration:")
    log(f"  N = {N}")
    log(f"  VAL_SPLIT = {VAL_SPLIT}")
    log(f"  SEED = {SEED}")
    log(f"  train per dataset = {per_dataset_train}")
    log(f"  val per dataset = {per_dataset_val}")
    log("")

    master_rng = random.Random(SEED)
    train_rng = random.Random(master_rng.randint(0, 10**9))
    val_rng = random.Random(master_rng.randint(0, 10**9))

    log("Building train split from train.jsonl ...")
    train_samples = build_split(
        source_filename="train.jsonl",
        per_dataset_count=per_dataset_train,
        rng=train_rng,
    )
    log(f"Train split complete: {len(train_samples)} samples")
    log("")

    log("Building val split from dev.jsonl ...")
    val_samples = build_split(
        source_filename="dev.jsonl",
        per_dataset_count=per_dataset_val,
        rng=val_rng,
    )
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