"""
main.py
-------
Entry point for the LLM evaluation pipeline.

Usage
-----
    python main.py

All settings are controlled via config.py.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure Unicode characters print correctly on Windows terminals (cp1252 → utf-8)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

import config
import llm_client
import results as results_mod
from data_loader import (
    find_checkpoint_dirs,
    find_dataset_files,
    build_pid_map,
    load_items,
    parse_checkpoint_number,
    compute_sampled_pids,
)
from evaluator import process_single_item
from metrics.registry import get_active_metrics
from reporting.comparison_logs import write_comparison_logs, COMPARISON_LOG_DIR


# ---------------------------------------------------------------------------
# Bootstrap output directories
# ---------------------------------------------------------------------------

def _setup_dirs() -> None:
    for d in (
        config.BASE_OUTPUT_DIR,
        config.LOG_DIR,
        config.UNNORM_DIR,
        config.NORM_DIR,
        COMPARISON_LOG_DIR
    ):
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run() -> None:
    print("=" * 60)
    print(" LLM EVALUATION PIPELINE")
    print("=" * 60)
    print(f"  Run ID     : {config.RUN_ID}")
    print(f"  Judge model: {config.JUDGE_MODEL}")
    print(f"  N samples  : {config.N_SAMPLES}")
    print(f"  Seed       : {config.RANDOM_SEED}")
    if config.EXCLUDED_CHECKPOINTS:
        print(f"  Excluded   : {', '.join(config.EXCLUDED_CHECKPOINTS)}")

    _setup_dirs()

    active_metrics = get_active_metrics(config.ACTIVE_METRICS)
    print(f"\nActive metrics ({len(active_metrics)}):")
    for name, m in active_metrics.items():
        print(f"  [{m.metric_type:8s}] {name}: {m.description}")

    # ── 1. Find checkpoints ──────────────────────────────────────────────
    checkpoint_dirs = find_checkpoint_dirs()
    if not checkpoint_dirs:
        print("\n[ERROR] No checkpoint directories found. Adjust _CKPT_PATTERNS in data_loader.py.")
        sys.exit(1)
    print(f"\nFound {len(checkpoint_dirs)} checkpoint(s):")
    for d in checkpoint_dirs:
        print(f"  {d}")

    # ── Write config snapshot now that checkpoints are resolved ──────────
    results_mod.write_config_snapshot(
        checkpoint_dirs=checkpoint_dirs,
        active_metric_names=list(active_metrics),
        active_datasets=config.ACTIVE_DATASETS,
    )

    # ── 2. API connectivity check ─────────────────────────────────────────
    if not llm_client.test_connection():
        if config.CONTINUE_ON_API_FAILURE:
            print("[WARN] API unreachable; continuing because CONTINUE_ON_API_FAILURE is set.")
        elif sys.stdin.isatty():
            answer = input("\n[!] API unreachable. Continue anyway? (yes/no): ").strip().lower()
            if answer not in ("yes", "y"):
                sys.exit(1)
        else:
            print("[ERROR] API unreachable and no TTY available for prompt.")
            print("        Set CONTINUE_ON_API_FAILURE=1 to force continuation.")
            sys.exit(1)

    # ── 3. Pre-compute stable shared sample set ───────────────────────────
    sampled_pids = compute_sampled_pids(checkpoint_dirs)

    # ── 4. Main loop over checkpoints ─────────────────────────────────────
    all_results: list[dict] = []          # Every item result across all checkpoints
    global_unnorm: list = []
    global_norm: list = []

    for ckpt_dir in checkpoint_dirs:
        ckpt_num = parse_checkpoint_number(ckpt_dir)
        if ckpt_num is None:
            print(f"[WARN] Cannot parse checkpoint number from '{ckpt_dir}' – skipping.")
            continue

        ckpt_name = os.path.basename(ckpt_dir)
        print(f"\n{'─'*60}")
        print(f"  Checkpoint: {ckpt_name}  (step {ckpt_num})")
        print(f"{'─'*60}")

        dataset_files = find_dataset_files(ckpt_dir, ckpt_num)
        if not dataset_files:
            print(f"  [WARN] No dataset files found in {ckpt_dir}")
            continue

        tasks: list[tuple] = []
        for f_info in dataset_files:
            ds = f_info["dataset"]
            wanted_pids = sampled_pids.get(ds, [])
            if not wanted_pids:
                print(f"  {ds}: skipped (no sampled PIDs)")
                continue

            try:
                items = load_items(f_info["path"])
                pid_map = build_pid_map(items)
            except Exception as exc:
                print(f"  [ERROR] Loading {f_info['path']}: {exc}")
                continue

            count = 0
            for pid in wanted_pids:
                rec = pid_map.get(pid)
                if rec:
                    reasoning, item = rec
                    tasks.append((ds, ckpt_num, pid, reasoning, item))
                    count += 1
            print(f"  {ds}: queued {count}/{len(wanted_pids)} items")

        if not tasks:
            print("  No tasks – skipping checkpoint.")
            continue

        # Run in parallel
        ckpt_results: list[dict] = []
        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
            futures = {pool.submit(process_single_item, t): t for t in tasks}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    ckpt_results.append(result)
                except Exception as exc:
                    task = futures[future]
                    print(f"  [ERROR] Task {task[0]}/{task[2]}: {exc}")
                    print(f"  ERROR TYPE: {type(exc)}")
                    import traceback
                    print(f"  TRACEBACK: {traceback.format_exc()}")

        all_results.extend(ckpt_results)

        # Save per-checkpoint CSVs and collect summary rows
        unnorm_summary, norm_summary = results_mod.save_checkpoint_csvs(
            checkpoint_num=ckpt_num,
            checkpoint_dir_name=ckpt_name,
            result_rows=ckpt_results,
            active_metric_names=list(active_metrics),
        )
        global_unnorm.append(unnorm_summary)
        global_norm.append(norm_summary)

    # ── 5. Post-run outputs ───────────────────────────────────────────────
    if all_results:
        results_mod.write_debug_logs(all_results)
        write_comparison_logs(all_results, active_metrics)

    print("\n[Generating comparison tables …]")
    results_mod.generate_comparison_tables(config.UNNORM_DIR, "Unnormalized")
    results_mod.generate_comparison_tables(config.NORM_DIR, "Normalized")

    print("\n" + "=" * 60)
    print(" DONE".center(60))
    print("=" * 60)
    print(f"\nOutputs in: {config.BASE_OUTPUT_DIR}/")
    print(f"  {config.UNNORM_DIR}/")
    print(f"  {config.NORM_DIR}/")
    print(f"  {config.LOG_DIR}/")
    print(f"  {COMPARISON_LOG_DIR}/  (pairwise, 2-checkpoint runs only)")


if __name__ == "__main__":
    run()
