import json
import os
import random
from collections import Counter
from typing import Any, Dict, List, Tuple

# =========================
# Config
# =========================
TRAIN_PER_DATASET = 380
VAL_PER_DATASET = 100
SEED = 42

OUT_TRAIN = "train_split.json"
OUT_VAL = "val_split.json"

# Folder name -> datasetName to add into each sample
DATASET_NAME_MAP = {
    "AbductionRules": "AbductionRules",
    "Balanced COPA": "BalancedCOPA",
    "CauseLogics": "CauseLogics",
    "Climate-Fever": "ClimateFever",
    "Crypto": "Crypto",
    "List Function": "ListFunction",
    "UniADILR": "UniADILR",
}


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def sample_up_to(items: List[Dict[str, Any]], k: int, rng: random.Random) -> List[Dict[str, Any]]:
    if len(items) <= k:
        return list(items)
    return rng.sample(items, k)


def add_dataset_name(sample: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    new_sample = dict(sample)
    new_sample["datasetName"] = dataset_name
    return new_sample


def infer_split_file(folder_path: str, split_name: str) -> str:
    """
    Finds the JSON file for a split inside a dataset subfolder.
    Expected patterns include files like:
      *_train.json
      *_val.json
    """
    candidates = []
    for fname in os.listdir(folder_path):
        lower = fname.lower()
        if not lower.endswith(".json"):
            continue
        if split_name == "train" and "_train_enhanced.json" in lower:
            candidates.append(fname)
        elif split_name == "val" and "_val_enhanced.json" in lower:
            candidates.append(fname)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find a {split_name} json file in: {folder_path}"
        )

    if len(candidates) > 1:
        # Prefer shortest name if multiple matches exist
        candidates.sort(key=len)

    return os.path.join(folder_path, candidates[0])


def load_dataset_split(folder_name: str, split_name: str) -> List[Dict[str, Any]]:
    folder_path = os.path.join(os.getcwd(), folder_name)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Missing dataset folder: {folder_path}")

    split_file = infer_split_file(folder_path, split_name)
    data = load_json(split_file)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected top-level JSON array in {split_file}, got {type(data).__name__}"
        )

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"Expected every sample in {split_file} to be a JSON object, "
                f"but item {i} is {type(item).__name__}"
            )

    return data


def build_mixed_split(
    split_name: str,
    max_per_dataset: int,
    rng: random.Random
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    mixed = []
    counts = {}

    for folder_name, dataset_name in DATASET_NAME_MAP.items():
        samples = load_dataset_split(folder_name, split_name)
        picked = sample_up_to(samples, max_per_dataset, rng)
        picked = [add_dataset_name(x, dataset_name) for x in picked]

        mixed.extend(picked)
        counts[dataset_name] = len(picked)

    rng.shuffle(mixed)
    return mixed, counts


def print_summary(split_name: str, counts: Dict[str, int], total: int) -> None:
    print(f"\n{split_name} summary:")
    for dataset_name in sorted(counts):
        print(f"  {dataset_name}: {counts[dataset_name]}")
    print(f"  TOTAL: {total}")


def main() -> None:
    rng = random.Random(SEED)

    train_split, train_counts = build_mixed_split(
        split_name="train",
        max_per_dataset=TRAIN_PER_DATASET,
        rng=rng,
    )

    val_split, val_counts = build_mixed_split(
        split_name="val",
        max_per_dataset=VAL_PER_DATASET,
        rng=rng,
    )

    save_json(OUT_TRAIN, train_split)
    save_json(OUT_VAL, val_split)

    print(f"Wrote {OUT_TRAIN}")
    print(f"Wrote {OUT_VAL}")

    print_summary("train_split", train_counts, len(train_split))
    print_summary("val_split", val_counts, len(val_split))

    # Optional sanity check
    print("\nSanity check from written objects:")
    print("  train:", dict(Counter(x["datasetName"] for x in train_split)))
    print("  val:  ", dict(Counter(x["datasetName"] for x in val_split)))


if __name__ == "__main__":
    main()