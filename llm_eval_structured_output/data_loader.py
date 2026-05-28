"""
data_loader.py
--------------
Checkpoint discovery, dataset file resolution, item loading, and sampling.

All file-system concerns live here so the rest of the pipeline is agnostic to
the on-disk layout.
"""

from __future__ import annotations

import glob
import json
import os
import random
import sys
from typing import Any

import config


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Item = dict[str, Any]
PidMap = dict[Any, tuple[str, Item]]  # problem_id → (reasoning_text, raw_item)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

_CKPT_PATTERNS = [
    "checkpoint-*",
    "raw_model",
    "checkpoints/checkpoint-*",
    "checkpoints/raw_model",
    "checkpoints/*",
    "evaluation-llm/checkpoints/checkpoint-*",
    "evaluation-llm/checkpoints/raw_model",
    "../checkpoint-*",
    "../raw_model",
]


def find_checkpoint_dirs() -> list[str]:
    """Return a sorted, de-duplicated list of checkpoint directory paths.

    Checkpoints whose base directory name appears in
    ``config.EXCLUDED_CHECKPOINTS`` are silently dropped.
    """
    found: list[str] = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for pattern in _CKPT_PATTERNS:
        # Search relative to CWD and to the package directory so runs from
        # repo root or from this folder both work.
        found.extend(glob.glob(pattern))
        if not os.path.isabs(pattern):
            found.extend(glob.glob(os.path.join(base_dir, pattern)))

    excluded = {e.strip() for e in (config.EXCLUDED_CHECKPOINTS or [])}
    canonical_paths: dict[str, str] = {}
    for path in found:
        abs_path = os.path.abspath(path)
        norm_path = os.path.normpath(abs_path)
        key = os.path.normcase(norm_path)
        canonical_paths[key] = norm_path

    result: list[str] = []
    for path in sorted(canonical_paths.values()):
        basename = os.path.basename(path)
        if basename in excluded:
            print(f"[INFO] Skipping excluded checkpoint: {path}")
            continue
        result.append(path)

    if len(result) > 2 and sys.stdin.isatty():
        print("\n[INFO] More than 2 checkpoints found.")
        print("Select one checkpoint to evaluate:")
        for idx, path in enumerate(result, start=1):
            print(f"  {idx}. {path}")

        while True:
            choice = input("Enter checkpoint number: ").strip()
            if choice.isdigit():
                i = int(choice)
                if 1 <= i <= len(result):
                    result = [result[i - 1]]
                    break
            print("[WARN] Invalid selection. Please enter a valid number.")

    return result


def parse_checkpoint_number(ckpt_dir: str) -> int | None:
    """Extract the integer checkpoint step from a directory name like ``checkpoint-500``.

    ``raw_model`` is treated as an alias for ``checkpoint-0``.
    """
    basename = os.path.basename(os.path.normpath(ckpt_dir))
    if basename == "raw_model":
        return 0
    try:
        return int(basename.split("-")[-1])
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Dataset file discovery inside a checkpoint
# ---------------------------------------------------------------------------

def _target_filename(ckpt_num: int) -> str:
    """Checkpoint-0 uses a different file than trained checkpoints."""
    return "raw_results_train_all.json" if ckpt_num == 0 else "all_cases.json"


def find_dataset_files(ckpt_dir: str, ckpt_num: int) -> list[dict[str, str]]:
    """
    Return a list of ``{"path": ..., "dataset": ...}`` dicts for all dataset
    files found under ``ckpt_dir``.

    Only datasets listed in ``config.ACTIVE_DATASETS`` are returned;
    if that list is empty, all datasets are returned.
    """
    target = _target_filename(ckpt_num)
    pattern = os.path.join(ckpt_dir, "*", target)
    active_ds = [d.lower() for d in config.ACTIVE_DATASETS] if config.ACTIVE_DATASETS else []
    results: list[dict[str, str]] = []
    for path in sorted(glob.glob(pattern)):
        parts = os.path.normpath(path).split(os.sep)
        dataset_name = parts[-2].lower()
        if active_ds and dataset_name not in active_ds:
            continue
        results.append({"path": path, "dataset": dataset_name})
    return results


# ---------------------------------------------------------------------------
# Item loading
# ---------------------------------------------------------------------------

def load_items(path: str) -> list[Item]:
    """Load the list of evaluation items from a JSON file.

    If any item is missing a ``problem_id``, sequential integer IDs (0-based)
    are assigned in the order the items appear in the file so that the sampling
    logic can treat them as normal items.
    """
    with open(path, encoding="utf-8") as fh:
        content = json.load(fh)
    if isinstance(content, dict) and "results" in content:
        items = content["results"]
    elif isinstance(content, list):
        items = content
    else:
        items = []

    # Assign sequential IDs to items that are missing problem_id
    for idx, item in enumerate(items):
        if item.get("problem_id") is None:
            item["problem_id"] = idx

    return items


# get_labels function removed since there's no ground truth


def extract_reasoning(item: Item) -> str | None:
    """
    Pull the reasoning text from an item regardless of the schema variant.
    Returns None if no usable reasoning is found.
    """
    # Schema variant 1: {"finetuned": {"reasoning": ...}}
    ft = item.get("finetuned")
    if isinstance(ft, dict):
        r = ft.get("reasoning")
        if r:
            return r
    if isinstance(ft, str) and ft:
        return ft
    # Schema variant 2: {"reasoning": ...}
    r = item.get("reasoning")
    if r:
        return r
    # Schema variant 3: Extract reasoning from full_response
    full_response = item.get("full_response", "")
    if full_response:
        # Try to extract reasoning text between <reasoning> tags
        import re
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', full_response, re.DOTALL)
        if reasoning_match:
            return reasoning_match.group(1).strip()
    return None


def _coerce_text(value: Any) -> str:
    """Convert a value into a display-safe text block (possibly empty)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False).strip()
        except Exception:
            return str(value).strip()
    return str(value).strip()


def extract_full_input(item: Item) -> str:
    """
    Extract source question/input text from ``item["user_input"]``.

    If missing, return an empty string.
    """
    if not isinstance(item, dict):
        return ""
    return _coerce_text(item.get("user_input"))


def _is_placeholder(text: str) -> bool:
    """True if the reasoning text is a template placeholder, not real reasoning."""
    return "here you write your chain-of-thought" in text.lower()


# get_labels function removed since there's no ground truth


def build_pid_map(items: list[Item]) -> PidMap:
    """
    Build a mapping ``problem_id → (reasoning_text, raw_item)`` for all items
    that have a valid, non-placeholder reasoning trace.
    """
    out: PidMap = {}
    for item in items:
        reasoning = extract_reasoning(item)
        if not reasoning or _is_placeholder(reasoning):
            continue
        pid = item.get("problem_id")
        if pid is not None:
            out[pid] = (reasoning, item)
    return out


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _load_pinned_sample(n_samples: int) -> list | None:
    """
    Look for ``{config.RANDOM_SAMPLES_DIR}/samples_{n_samples}.json``.

    If the file exists, load and return its contents as a list of integers.
    Returns ``None`` when no matching file is found or the directory is not
    configured, so the caller falls back to seeded random sampling.
    """
    samples_dir = getattr(config, "RANDOM_SAMPLES_DIR", None)
    if not samples_dir:
        return None
    candidate = os.path.join(samples_dir, f"samples_{n_samples}.json")
    if not os.path.isfile(candidate):
        return None
    with open(candidate, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        print(f"[WARN] {candidate}: expected a JSON array, got {type(data).__name__} – ignoring.")
        return None
    print(f"[INFO] Using pinned sample from '{candidate}' ({len(data)} indices).")
    return [int(x) for x in data]


def _sort_pids(pids: set) -> list:
    def _key(x: Any) -> tuple:
        try:
            return (0, int(str(x)))
        except (ValueError, TypeError):
            return (1, str(x))
    return sorted(pids, key=_key)


def compute_sampled_pids(
    checkpoint_dirs: list[str],
    *,
    n_samples: int = config.N_SAMPLES,
    seed: int = config.RANDOM_SEED,
) -> dict[str, list]:
    """
    Pre-compute ONE shared sample set per dataset across ALL checkpoints.

    Benefits:
    - The same problem IDs are evaluated in every checkpoint → fair cross-checkpoint
      comparison.

    Returns a dict ``{dataset_name: [pid, ...]}`` for every dataset that is
    present across *all* checkpoints.
    """
    ckpt_nums_by_dir: dict[str, int] = {}
    for ckpt_dir in checkpoint_dirs:
        n = parse_checkpoint_number(ckpt_dir)
        if n is not None:
            ckpt_nums_by_dir[ckpt_dir] = n

    expected_ckpt_count = len(ckpt_nums_by_dir)

    intersection: dict[str, set] = {}
    dataset_ckpt_count: dict[str, int] = {}

    for ckpt_dir, ckpt_num in ckpt_nums_by_dir.items():
        for f_info in find_dataset_files(ckpt_dir, ckpt_num):
            ds = f_info["dataset"]
            try:
                items = load_items(f_info["path"])
                pid_map = build_pid_map(items)
            except Exception as exc:
                print(f"[WARN] Could not load {f_info['path']}: {exc}")
                continue

            valid_pids = set(pid_map)

            dataset_ckpt_count[ds] = dataset_ckpt_count.get(ds, 0) + 1

            if ds not in intersection:
                intersection[ds] = valid_pids
            else:
                intersection[ds] &= valid_pids

    # Drop datasets not present in every checkpoint
    for ds in list(intersection):
        if dataset_ckpt_count.get(ds, 0) < expected_ckpt_count:
            print(
                f"[WARN] Dataset '{ds}' found in only "
                f"{dataset_ckpt_count[ds]}/{expected_ckpt_count} checkpoints – skipping."
            )
            del intersection[ds]

    # Check for a pre-generated pinned sample file for this N
    pinned_indices: list | None = _load_pinned_sample(n_samples)

    random.seed(seed)
    sampled: dict[str, list] = {}

    for ds, pid_set in intersection.items():
        if not pid_set:
            print(f"[WARN] Dataset '{ds}': no common valid items across checkpoints.")
            continue

        # ── Pinned-sample path ───────────────────────────────────────────
        if pinned_indices is not None:
            # Keep only indices that are valid PIDs for this dataset
            available = [p for p in pinned_indices if p in pid_set]
            if not available:
                print(
                    f"[WARN] Dataset '{ds}': pinned sample has no overlap with valid PIDs – "
                    "falling back to random sampling."
                )
            else:
                sampled[ds] = _sort_pids(set(available))
                print(
                    f"[OK] '{ds}': using {len(sampled[ds])} pinned indices "
                    f"({n_samples - len(sampled[ds])} requested indices not present in dataset)"
                    if len(sampled[ds]) < n_samples else
                    f"[OK] '{ds}': using {len(sampled[ds])} pinned indices"
                )
                continue

        k = min(n_samples, len(pid_set))
        sampled[ds] = _sort_pids(random.sample(_sort_pids(pid_set), k))
        print(f"[OK] '{ds}': sampled {len(sampled[ds])} items (no stratification)")

    return sampled
