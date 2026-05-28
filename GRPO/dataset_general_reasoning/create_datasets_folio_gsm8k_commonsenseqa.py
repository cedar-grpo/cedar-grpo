import json
import os
import random
import requests
from typing import Any, Dict, List
from datasets import load_dataset, get_dataset_split_names

# =========================
# Config
# =========================
TRAIN_PER_DATASET = 320
VAL_PER_DATASET = 80
SEED = 42

OUTPUT_DIR = "dataset"


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def add_dataset_name(samples: List[Dict[str, Any]], dataset_name: str) -> List[Dict[str, Any]]:
    out =[]
    for s in samples:
        new_sample = dict(s)
        new_sample["datasetName"] = dataset_name
        out.append(new_sample)
    return out


def fetch_jsonl_from_github(url: str) -> List[Dict[str, Any]]:
    print(f"Downloading {url} ...")
    response = requests.get(url)
    response.raise_for_status()
    
    data =[]
    for line in response.text.strip().split("\n"):
        if line.strip():
            data.append(json.loads(line))
    return data


def process_gsm8k(rng: random.Random) -> None:
    dataset_name = "gsm8k"
    dataset_config = "main"
    
    available_splits = get_dataset_split_names(dataset_name, dataset_config)
    selected_split = "train"
    
    print(f"\nLoading GSM8K dataset (split={selected_split})...")
    dataset = load_dataset(dataset_name, dataset_config, split=selected_split)
    
    # 1. Convert to list and shuffle
    data_list = list(dataset)
    rng.shuffle(data_list)
    
    # 2. Sample 400 total
    total_needed = TRAIN_PER_DATASET + VAL_PER_DATASET
    sampled = data_list[:total_needed]
    
    # 3. Split 80/20 (320 Train / 80 Val)
    train_split = sampled[:TRAIN_PER_DATASET]
    val_split = sampled[TRAIN_PER_DATASET:]
    
    # 4. Add dataset name tag
    train_split = add_dataset_name(train_split, dataset_name)
    val_split = add_dataset_name(val_split, dataset_name)
    
    # 5. Save files
    save_json(os.path.join(OUTPUT_DIR, f"{dataset_name}_train.json"), train_split)
    save_json(os.path.join(OUTPUT_DIR, f"{dataset_name}_val.json"), val_split)
    print(f"  -> Saved {dataset_name}: {len(train_split)} train, {len(val_split)} val.")


def process_commonsense_qa(rng: random.Random) -> None:
    dataset_name = "tau/commonsense_qa"
    safe_name = "commonsense_qa"
    
    print(f"\nLoading {dataset_name} dataset...")
    train_ds = load_dataset(dataset_name, split="train")
    val_ds = load_dataset(dataset_name, split="validation")
    
    train_list = list(train_ds)
    val_list = list(val_ds)
    
    # Shuffle splits
    rng.shuffle(train_list)
    rng.shuffle(val_list)
    
    # Extract exactly 320 from train, and 80 from validation
    train_split = train_list[:TRAIN_PER_DATASET]
    val_split = val_list[:VAL_PER_DATASET]
    
    train_split = add_dataset_name(train_split, safe_name)
    val_split = add_dataset_name(val_split, safe_name)
    
    save_json(os.path.join(OUTPUT_DIR, f"{safe_name}_train.json"), train_split)
    save_json(os.path.join(OUTPUT_DIR, f"{safe_name}_val.json"), val_split)
    print(f"  -> Saved {safe_name}: {len(train_split)} train, {len(val_split)} val.")


def process_folio(rng: random.Random) -> None:
    safe_name = "folio"
    print(f"\nLoading {safe_name} dataset from GitHub...")
    
    # Convert GitHub blob URLs to raw URLs to fetch data directly
    train_url = "https://raw.githubusercontent.com/Yale-LILY/FOLIO/main/data/v0.0/folio-train.jsonl"
    val_url = "https://raw.githubusercontent.com/Yale-LILY/FOLIO/main/data/v0.0/folio-validation.jsonl"
    
    train_list = fetch_jsonl_from_github(train_url)
    val_list = fetch_jsonl_from_github(val_url)
    
    # Shuffle splits
    rng.shuffle(train_list)
    rng.shuffle(val_list)
    
    # Extract exactly 320 from train, and 80 from validation
    train_split = train_list[:TRAIN_PER_DATASET]
    val_split = val_list[:VAL_PER_DATASET]
    
    train_split = add_dataset_name(train_split, safe_name)
    val_split = add_dataset_name(val_split, safe_name)
    
    save_json(os.path.join(OUTPUT_DIR, f"{safe_name}_train.json"), train_split)
    save_json(os.path.join(OUTPUT_DIR, f"{safe_name}_val.json"), val_split)
    print(f"  -> Saved {safe_name}: {len(train_split)} train, {len(val_split)} val.")


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    print(f"Targeting directory: ./{OUTPUT_DIR}/")
    
    # To maintain pure reproducibility across runs without order-dependency, 
    # we inject a fresh Random instance seeded identically into each processing block.
    process_gsm8k(random.Random(SEED))
    process_commonsense_qa(random.Random(SEED))
    process_folio(random.Random(SEED))
    
    print("\nAll datasets have been successfully processed, sampled, and saved!")


if __name__ == "__main__":
    main()
