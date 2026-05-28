"""
One-time migration script: standardizes datasets/ into consistent
JSONL + dataset_info.json folders with snake_case names.

Run from project root:
    python datasets/migrate.py

After verifying output, pass --delete to remove old folders.
"""
import sys
# Remove cwd ('') from sys.path so 'import datasets' resolves to the
# HuggingFace package and not this datasets/ directory.
sys.path = [p for p in sys.path if p not in ("", ".")]

import argparse
import json
import os
import shutil
from pathlib import Path
import pyarrow.ipc as ipc

DATASETS_DIR = Path(__file__).parent
TV_DIR = DATASETS_DIR / "training_validation"


# ── helpers ──────────────────────────────────────────────────────────────────

def write_jsonl(records, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return dest


def copy_jsonl(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def json_array_to_jsonl(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "r", encoding="utf-8") as f:
        records = json.load(f)
    with open(dest, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def arrow_to_jsonl(arrow_dir: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    arrow_file = next(arrow_dir.glob("*.arrow"))
    with ipc.open_stream(arrow_file) as reader:
        table = reader.read_all()
    records = table.to_pylist()
    with open(dest, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def write_info(dest_dir: Path, info: dict):
    with open(dest_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
        f.write("\n")


def count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


# ── per-dataset migration ─────────────────────────────────────────────────────

def migrate_medqa():
    print("medqa ... ", end="", flush=True)
    out = DATASETS_DIR / "medqa"
    copy_jsonl(DATASETS_DIR / "MedQA" / "test.jsonl", out / "test.jsonl")
    copy_jsonl(DATASETS_DIR / "MedQA" / "phrases_no_exclude_test.jsonl", out / "test_with_phrases.jsonl")
    n = count_lines(out / "test.jsonl")
    write_info(out, {
        "name": "MedQA",
        "source": "https://github.com/jind11/MedQA",
        "description": "US Medical Licensing Exam (USMLE) multiple-choice questions.",
        "files": {
            "test.jsonl": {"split": "test", "num_examples": n,
                           "fields": ["question", "answer", "options", "meta_info", "answer_idx"]},
            "test_with_phrases.jsonl": {"split": "test", "num_examples": n,
                                        "fields": ["question", "answer", "options", "meta_info", "answer_idx", "metamap_phrases"],
                                        "note": "Augmented with MetaMap phrase extraction"},
        }
    })
    print(f"done ({n} examples)")


def migrate_neulr():
    print("neulr ... ", end="", flush=True)
    out = DATASETS_DIR / "neulr"
    splits = [
        ("abductive_neutral.json",  "abductive.jsonl",  "abductive"),
        ("deductive_neutral.json",  "deductive.jsonl",  "deductive"),
        ("inductive_neutral.json",  "inductive.jsonl",  "inductive"),
    ]
    counts = {}
    for src_name, dst_name, split in splits:
        n = json_array_to_jsonl(DATASETS_DIR / "NeuLR" / src_name, out / dst_name)
        counts[split] = n
    write_info(out, {
        "name": "NeuLR",
        "source": "Neural Logic Reasoning dataset",
        "description": "Abductive, deductive, and inductive reasoning over synthetic logical facts and rules.",
        "files": {
            "abductive.jsonl":  {"num_examples": counts["abductive"],  "fields": ["id", "context", "label", "explain"]},
            "deductive.jsonl":  {"num_examples": counts["deductive"],  "fields": ["id", "context", "question", "label", "explain"]},
            "inductive.jsonl":  {"num_examples": counts["inductive"],  "fields": ["id", "context", "question", "label", "explain"]},
        }
    })
    print(f"done (abductive={counts['abductive']}, deductive={counts['deductive']}, inductive={counts['inductive']})")


def migrate_musr():
    print("musr ... ", end="", flush=True)
    out = DATASETS_DIR / "musr"
    files = [
        ("murder_mystery.json",    "murder_mystery.jsonl"),
        ("object_placements.json", "object_placements.jsonl"),
        ("team_allocation.json",   "team_allocation.jsonl"),
    ]
    counts = {}
    for src_name, dst_name in files:
        key = dst_name.replace(".jsonl", "")
        print(f"\n  converting {src_name} ...", end="", flush=True)
        n = json_array_to_jsonl(DATASETS_DIR / "musr" / src_name, out / dst_name)
        counts[key] = n
        print(f" {n} examples", end="", flush=True)
    write_info(out, {
        "name": "MuSR",
        "source": "MuSR: Multi-Step Soft Reasoning dataset",
        "description": "Long-form narrative reasoning tasks requiring multi-hop deductive inference.",
        "files": {
            "murder_mystery.jsonl":    {"num_examples": counts["murder_mystery"],    "fields": ["context", "questions"]},
            "object_placements.jsonl": {"num_examples": counts["object_placements"], "fields": ["context", "questions"]},
            "team_allocation.jsonl":   {"num_examples": counts["team_allocation"],   "fields": ["context", "questions"]},
        }
    })
    print(f"\n  musr done")


def migrate_uniadilr():
    print("uniadilr ... ", end="", flush=True)
    out = DATASETS_DIR / "uniadilr"
    copy_jsonl(DATASETS_DIR / "UniADILR-HGc" / "abduction.jsonl", out / "abduction.jsonl")
    copy_jsonl(DATASETS_DIR / "UniADILR-HGc" / "abduction-multi-choice.jsonl", out / "abduction_multi_choice.jsonl")
    n = count_lines(out / "abduction.jsonl")
    write_info(out, {
        "name": "UniADILR-HGc",
        "source": "https://github.com/YuSheng-00/UniADILR/tree/main/data/UniADILR-HGc",
        "description": "Unified Abductive, Deductive, Inductive Logical Reasoning dataset (HGc subset).",
        "files": {
            "abduction.jsonl":             {"num_examples": n, "fields": ["context"]},
            "abduction_multi_choice.jsonl": {"num_examples": n, "note": "Multi-choice version of abduction.jsonl"},
        }
    })
    print(f"done ({n} examples)")


def migrate_arrow(tv_folder: str, out_name: str, info: dict):
    print(f"{out_name} ... ", end="", flush=True)
    out = DATASETS_DIR / out_name
    n = arrow_to_jsonl(TV_DIR / tv_folder, out / "data.jsonl")
    info["files"] = {"data.jsonl": {"num_examples": n, **info.get("file_meta", {})}}
    info.pop("file_meta", None)
    write_info(out, info)
    print(f"done ({n} examples)")


# ── main ──────────────────────────────────────────────────────────────────────

def main(delete_old: bool):
    print("=== Dataset migration ===\n")

    migrate_medqa()
    migrate_neulr()
    migrate_musr()
    migrate_uniadilr()

    migrate_arrow("art", "art", {
        "name": "ART",
        "source": "https://huggingface.co/datasets/allenai/art",
        "description": "Abductive NLI (αNLI): choose the more plausible hypothesis given two observations.",
        "file_meta": {"fields": ["observation_1", "observation_2", "hypothesis_1", "hypothesis_2", "label"]},
    })

    migrate_arrow("copa_guess_effect", "copa", {
        "name": "Balanced COPA",
        "source": "https://huggingface.co/datasets/pkavumba/balanced-copa",
        "description": "Balanced COPA: causal commonsense reasoning (cause/effect) as binary choice.",
        "file_meta": {"fields": ["id", "premise", "question", "choice1", "choice2", "label", "mirrored"]},
    })

    migrate_arrow("go_emotions", "go_emotions", {
        "name": "GoEmotions",
        "source": "https://huggingface.co/datasets/google-research-datasets/go_emotions",
        "description": "Reddit comments labeled with 28 fine-grained emotion categories.",
        "file_meta": {"fields": ["text", "labels", "id"]},
    })

    migrate_arrow("gsm8k", "gsm8k", {
        "name": "GSM8K",
        "source": "https://huggingface.co/datasets/gsm8k",
        "description": "Grade School Math 8K: math word problems requiring multi-step arithmetic reasoning.",
        "file_meta": {"fields": ["question", "answer"]},
    })

    migrate_arrow("miniarc", "miniarc", {
        "name": "MiniARC",
        "source": "Local dataset (miniarc.jsonl)",
        "description": "Mini version of ARC (Abstraction and Reasoning Corpus) visual grid puzzles.",
        "file_meta": {"fields": ["idx", "train", "test"]},
    })

    print("\n=== All datasets migrated ===\n")

    # Verify all targets exist before deleting
    expected = [
        "medqa/test.jsonl", "medqa/test_with_phrases.jsonl",
        "neulr/abductive.jsonl", "neulr/deductive.jsonl", "neulr/inductive.jsonl",
        "musr/murder_mystery.jsonl", "musr/object_placements.jsonl", "musr/team_allocation.jsonl",
        "uniadilr/abduction.jsonl", "uniadilr/abduction_multi_choice.jsonl",
        "art/data.jsonl", "copa/data.jsonl", "go_emotions/data.jsonl",
        "gsm8k/data.jsonl", "miniarc/data.jsonl",
    ]
    missing = [p for p in expected if not (DATASETS_DIR / p).exists()]
    if missing:
        print(f"ERROR: missing output files: {missing}")
        return

    print("All output files verified.\n")

    if delete_old:
        old = ["MedQA", "NeuLR", "musr/murder_mystery.json", "musr/object_placements.json",
               "musr/team_allocation.json", "UniADILR-HGc", "training_validation"]
        for name in old:
            p = DATASETS_DIR / name
            if p.exists():
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                print(f"  deleted {p}")
        # Remove musr/ old JSON files; the folder itself stays (it has new .jsonl files)
        print("\nOld folders deleted.")
    else:
        print("Dry run — old folders kept. Re-run with --delete to remove them.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true",
                        help="Delete old folders after migration")
    args = parser.parse_args()
    main(args.delete)
