"""
results.py
----------
Backward-compatible facade.

All logic has been moved into the ``reporting`` package:
  - reporting/csv.py   → CSV saving and debug logs
  - reporting/excel.py → colour-coded Excel workbooks
  - reporting/plots.py → evaluation line plots and tier bar charts

This module re-exports the public API so that existing call-sites
(``main.py`` and any scripts) continue to work without modification.
"""

from __future__ import annotations

import glob
import os

import pandas as pd

from reporting.csv import save_checkpoint_csvs, write_debug_logs, write_config_snapshot  # noqa: F401
from reporting.excel import build_excel_workbook
from reporting.plots import build_evaluation_plots  # noqa: F401



# ---------------------------------------------------------------------------
# Comparison tables orchestrator (called by main.py)
# ---------------------------------------------------------------------------

def generate_comparison_tables(base_dir: str, label: str) -> None:
    """Combine per-checkpoint summary CSVs, build Excel workbook and line plots."""
    print(f"\n{'='*60}\nGenerating comparison tables: {label}\n{'='*60}")

    ckpt_dirs = [
        d for d in (
            glob.glob(os.path.join(base_dir, "checkpoint-*")) +
            glob.glob(os.path.join(base_dir, "raw_model"))
        )
        if os.path.isdir(d)
    ]
    if not ckpt_dirs:
        print(f"[WARN] No checkpoint-* or raw_model dirs in {base_dir}")
        return

    frames: list[pd.DataFrame] = []
    for d in ckpt_dirs:
        f = os.path.join(d, "summary_metrics.csv")
        if os.path.exists(f):
            frames.append(pd.read_csv(f))

    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["Dataset", "Checkpoint"], inplace=True)
    combined.to_csv(os.path.join(base_dir, "all_checkpoints_summary.csv"), index=False)

    # Compact non-graph summary: only key columns, only those actually evaluated.
    _wanted_compact = [
        "Dataset",
        "Checkpoint",
        "branchiness_count",
        "backtracking_count",
        "uncertainty_markers_count",
        "prior_count",
        "differential_elimination_count",
        "evidence_explanation_directionality_scorebased_score",
        "observation_coverage_score",
        "Word Count",
    ]
    compact_columns = [c for c in _wanted_compact if c in combined.columns]
    compact_df = combined[compact_columns]
    compact_df.to_csv(os.path.join(base_dir, "all_checkpoints_summary_compact_non_graph.csv"), index=False)

    build_excel_workbook(combined, base_dir)
    build_evaluation_plots(combined, base_dir)
