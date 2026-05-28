"""
reporting/plots.py
------------------
Evaluation line plots.

Plots are generated directly from the statuses present in the summary data.
If multiple statuses are present, one plot is produced per status plus an
aggregate "mix" plot. If there is only one status, only that status is plotted.
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")


def _metric_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"Checkpoint", "Dataset", "Status", "Word Count"}
    cols: list[str] = []
    for c in df.columns:
        if c in excluded:
            continue
        if c.endswith("_analysis") or c.endswith("_examples"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _status_slug(status: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in status).strip("_")
    return normalized or "status"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evaluation_plot(df: pd.DataFrame, metric_col: str, base_dir: str, suffix: str = "") -> None:
    if metric_col not in df.columns:
        return
    try:
        pivot = df.pivot_table(index="Checkpoint", columns="Dataset", values=metric_col, aggfunc="mean")
    except Exception:
        return
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for ds in pivot.columns:
        ax.plot(pivot.index, pivot[ds], marker="o", label=ds, linewidth=2)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel(metric_col)
    ax.set_title(f"{metric_col} – evaluation across checkpoints" + (f" ({suffix})" if suffix else ""))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fname = f"evaluation_{metric_col}{'_' + suffix if suffix else ''}.png"
    os.makedirs(os.path.join(base_dir, suffix if suffix else "None"), exist_ok=True)
    plt.savefig(os.path.join(os.path.join(base_dir, suffix if suffix else "None"), fname), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation line plots
# ---------------------------------------------------------------------------

def build_evaluation_plots(combined: pd.DataFrame, base_dir: str) -> None:
    """Produce evaluation PNGs from the statuses actually present in the data."""
    metric_cols = _metric_columns(combined)
    has_status = "Status" in combined.columns
    statuses = []
    if has_status:
        statuses = [status for status in combined["Status"].dropna().unique()]

    for col in metric_cols:
        if not has_status:
            _evaluation_plot(combined, col, base_dir, suffix="mix")
        elif len(statuses) <= 1:
            suffix = _status_slug(str(statuses[0])) if statuses else "status"
            _evaluation_plot(combined, col, base_dir, suffix=suffix)
        else:
            for status in statuses:
                sub = combined[combined["Status"] == status]
                if not sub.empty:
                    _evaluation_plot(sub, col, base_dir, suffix=_status_slug(str(status)))
            _evaluation_plot(combined, col, base_dir, suffix="mix")
