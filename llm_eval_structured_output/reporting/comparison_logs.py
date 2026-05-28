"""
reporting/comparison_logs.py
----------------------------
Pairwise comparison logs produced when the run has **exactly two checkpoints**.

For every (dataset × metric) combination, CSV files are written to
``results/comparison_logs/<dataset>/<metric_name>/``:

Binary metrics
~~~~~~~~~~~~~~
* ``match.csv``    – instances where both checkpoints give the *same* detected value
* ``mismatch.csv`` – instances where the detected value differs between the two

Counting metrics
~~~~~~~~~~~~~~~~
(comparison is on ``example_count = len(examples)``)
* ``A_gt_B.csv`` – instances where checkpoint A count > checkpoint B count
* ``A_eq_B.csv`` – instances where checkpoint A count == checkpoint B count
* ``A_lt_B.csv`` – instances where checkpoint A count < checkpoint B count

Coverage metrics
~~~~~~~~~~~~~~~~
(comparison is on ``score``, a float 0.0–1.0)
* ``A_gt_B.csv`` – instances where checkpoint A score > checkpoint B score
* ``A_eq_B.csv`` – instances where checkpoint A score == checkpoint B score
* ``A_lt_B.csv`` – instances where checkpoint A score < checkpoint B score

In every file the two checkpoints are labeled A and B in checkpoint-number order
(lower checkpoint number = A).  Each row contains:
  dataset, problem_id, status, checkpoint_A, checkpoint_B,
  <metric>_A_value, <metric>_B_value,
  <metric>_A_analysis, <metric>_B_analysis,
  <metric>_A_examples (JSON), <metric>_B_examples (JSON)
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

import pandas as pd

import config


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

COMPARISON_LOG_DIR: str = os.path.join(config.BASE_OUTPUT_DIR, "comparison_logs")


def write_comparison_logs(all_results: list[dict], active_metrics: dict) -> None:
    """Generate pairwise comparison CSVs when len(checkpoints) == 2.

    Parameters
    ----------
    all_results:
        The flat list of per-item result dicts assembled in ``main.py``.
    active_metrics:
        The ``{name: BaseMetric}`` dict from the registry.
    """
    # ── Guard: exactly two checkpoints ────────────────────────────────────
    checkpoints = sorted({r["checkpoint"] for r in all_results})
    if len(checkpoints) != 2:
        if len(checkpoints) > 2:
            print(
                f"[comparison_logs] Skipped: {len(checkpoints)} checkpoints found "
                "(pairwise comparison logs require exactly 2)."
            )
        return

    ckpt_a, ckpt_b = checkpoints
    print(
        f"\n[comparison_logs] Generating pairwise logs  "
        f"A=ckpt{ckpt_a}  B=ckpt{ckpt_b}"
    )

    os.makedirs(COMPARISON_LOG_DIR, exist_ok=True)

    # ── Index results by (dataset, pid) → {checkpoint: result_dict} ───────
    index: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for r in all_results:
        key = (r["dataset"], r["pid"])
        index[key][r["checkpoint"]] = r

    # ── Collect (dataset, pid) pairs present in BOTH checkpoints ──────────
    paired_keys = [k for k, v in index.items() if ckpt_a in v and ckpt_b in v]
    if not paired_keys:
        print("[comparison_logs] No matching (dataset, pid) pairs found across the two checkpoints.")
        return

    print(f"[comparison_logs] {len(paired_keys)} paired items across all datasets.")

    # ── Build rows per metric ──────────────────────────────────────────────
    for metric_name, metric in active_metrics.items():
        _process_metric(
            metric_name=metric_name,
            metric_type=metric.metric_type,
            paired_keys=paired_keys,
            index=index,
            ckpt_a=ckpt_a,
            ckpt_b=ckpt_b,
        )


# ---------------------------------------------------------------------------
# Per-metric helper
# ---------------------------------------------------------------------------

def _process_metric(
    metric_name: str,
    metric_type: str,
    paired_keys: list[tuple],
    index: dict,
    ckpt_a: int,
    ckpt_b: int,
) -> None:
    """Build and write comparison CSVs for one metric."""

    # Group rows into buckets
    if metric_type == "binary":
        buckets: dict[str, list[dict]] = {"match": [], "mismatch": []}
    else:
        buckets = {"A_gt_B": [], "A_eq_B": [], "A_lt_B": []}

    for ds, pid in paired_keys:
        rec_a = index[(ds, pid)][ckpt_a]
        rec_b = index[(ds, pid)][ckpt_b]

        mdata_a: dict[str, Any] = (rec_a.get("metrics") or {}).get(metric_name, {})
        mdata_b: dict[str, Any] = (rec_b.get("metrics") or {}).get(metric_name, {})

        # Skip if either checkpoint had an error for this metric
        if mdata_a.get("error") or mdata_b.get("error"):
            continue

        # ── Extract the comparison value ───────────────────────────────────
        if metric_type == "binary":
            val_a: Any = mdata_a.get("detected", False)
            val_b: Any = mdata_b.get("detected", False)
        elif metric_type == "counting":
            val_a = mdata_a.get("example_count", 0) or 0
            val_b = mdata_b.get("example_count", 0) or 0
        elif metric_type in ("coverage", "scorebased"):  # score is the primary value
            val_a = mdata_a.get("score") or 0.0
            val_b = mdata_b.get("score") or 0.0
        else:
            val_a = mdata_a.get("score") or 0.0
            val_b = mdata_b.get("score") or 0.0

        # ── Build base row ─────────────────────────────────────────────────
        row = {
            "dataset": ds,
            "problem_id": pid,
            "status": rec_a.get("status", ""),
            "checkpoint_A": ckpt_a,
            "checkpoint_B": ckpt_b,
            f"{metric_name}_A_value": val_a,
            f"{metric_name}_B_value": val_b,
            f"{metric_name}_A_analysis": mdata_a.get("reasoning", ""),
            f"{metric_name}_B_analysis": mdata_b.get("reasoning", ""),
            f"{metric_name}_A_examples": _safe_json(mdata_a.get("examples", [])),
            f"{metric_name}_B_examples": _safe_json(mdata_b.get("examples", [])),
        }

        # ── Route into bucket ──────────────────────────────────────────────
        if metric_type == "binary":
            bucket_key = "match" if val_a == val_b else "mismatch"
        else:
            if val_a > val_b:
                bucket_key = "A_gt_B"
            elif val_a == val_b:
                bucket_key = "A_eq_B"
            else:
                bucket_key = "A_lt_B"

        buckets[bucket_key].append(row)

    # ── Sort A_gt_B and A_lt_B by abs(A - B) descending ───────────────────
    if metric_type != "binary":
        val_col_a = f"{metric_name}_A_value"
        val_col_b = f"{metric_name}_B_value"
        for bk in ("A_gt_B", "A_lt_B"):
            buckets[bk].sort(
                key=lambda r, ca=val_col_a, cb=val_col_b: abs(r[ca] - r[cb]),
                reverse=True,
            )

    # ── Write CSVs (one per bucket, per dataset) ───────────────────────────
    _write_buckets(metric_name, buckets, ckpt_a, ckpt_b)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _write_buckets(
    metric_name: str,
    buckets: dict[str, list[dict]],
    ckpt_a: int,
    ckpt_b: int,
) -> None:
    """Split each bucket by dataset and write one CSV per (dataset × bucket)."""

    for bucket_name, rows in buckets.items():
        if not rows:
            continue

        # Group by dataset
        by_ds: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_ds[row["dataset"]].append(row)

        for ds, ds_rows in by_ds.items():
            out_dir = os.path.join(COMPARISON_LOG_DIR, ds, metric_name)
            os.makedirs(out_dir, exist_ok=True)

            fname = f"{bucket_name}.csv"
            out_path = os.path.join(out_dir, fname)
            pd.DataFrame(ds_rows).to_csv(out_path, index=False, encoding="utf-8-sig")

    # Summary per metric
    summary_parts = [f"{k}={len(v)}" for k, v in buckets.items()]
    print(
        f"  [{metric_name}]  "
        + "  ".join(summary_parts)
        + f"  → {os.path.join(COMPARISON_LOG_DIR, '(per dataset)', metric_name)}"
    )


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)
