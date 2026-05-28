"""
reporting/csv.py
----------------
Per-checkpoint CSV writing, the end-of-run full debug log, and the run
configuration snapshot.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

import config


def _example_excerpt(example: dict[str, Any]) -> str:
    return example.get("excerpt", example.get("text", ""))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(obj)

def _ensure(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _raw_question(rec: dict[str, Any]) -> str:
    """Return the normalized raw-question value with backward-compatible fallback."""
    return rec.get("raw_question", rec.get("full_input", ""))


# ---------------------------------------------------------------------------
# Per-checkpoint CSV saving
# ---------------------------------------------------------------------------

def save_checkpoint_csvs(
    checkpoint_num: int,
    checkpoint_dir_name: str,
    result_rows: list[dict],
    active_metric_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Write per-item and summary CSVs for one checkpoint.

    Returns (unnorm_summary_df, norm_summary_df) with a ``Checkpoint`` column
    already set, ready to be appended to the global summary list.
    """
    _ensure(config.UNNORM_DIR, config.NORM_DIR)

    unnorm_rows: list[dict] = []
    norm_rows: list[dict] = []

    for row in result_rows:
        wc = row.get("word_count", 0) or 0
        norm_factor = 100.0 / wc if wc > 0 else 0.0

        base = {
            "Dataset": row["dataset"],
            "Status": row["status"],
            "Problem ID": row["pid"],
            "raw_question": _raw_question(row),
            "Word Count": wc,
        }

        unnorm_item = dict(base)
        norm_item = dict(base)

        for mname in active_metric_names:
            mdata = row.get("metrics", {}).get(mname, {})
            mtype = mdata.get("type", "")
            count = mdata.get("example_count", 0) or 0
            detected = mdata.get("detected", False)
            analysis = mdata.get("reasoning", "")
            score = mdata.get("score")  # float | None
            scalar_metrics = mdata.get("scalar_metrics") or {}
            normalized_scalar_metrics = mdata.get("normalized_scalar_metrics") or {}

            # For coverage metrics, detail_str lists each observation detail + addressed flag
            if mtype == "coverage":
                examples_str = "; ".join(
                    f"[{'Y' if e.get('addressed') else 'N'}] {e.get('detail', '')}"
                    for e in mdata.get("examples", []) if isinstance(e, dict)
                )
            else:
                examples_str = "; ".join(
                    _example_excerpt(e) for e in mdata.get("examples", []) if isinstance(e, dict)
                )

            # For scorebased metrics, score is the primary column; skip the
            # boolean _detected and the always-zero _count so they don't clutter
            # the summary CSVs and plots.
            if mtype != "scorebased":
                unnorm_item[f"{mname}_count"] = count
                unnorm_item[f"{mname}_detected"] = detected
            unnorm_item[f"{mname}_analysis"] = analysis
            unnorm_item[f"{mname}_examples"] = examples_str
            if score is not None:
                unnorm_item[f"{mname}_score"] = score

            if mtype != "scorebased":
                norm_item[f"{mname}_count"] = count * norm_factor
                norm_item[f"{mname}_detected"] = detected
            norm_item[f"{mname}_analysis"] = analysis
            norm_item[f"{mname}_examples"] = examples_str
            if score is not None:
                norm_item[f"{mname}_score"] = score

            for scalar_name, scalar_value in scalar_metrics.items():
                unnorm_item[f"{mname}__{scalar_name}"] = scalar_value

            for scalar_name, scalar_value in scalar_metrics.items():
                norm_item[f"{mname}__{scalar_name}"] = scalar_value * norm_factor
            for scalar_name, scalar_value in normalized_scalar_metrics.items():
                norm_item[f"{mname}__{scalar_name}"] = scalar_value

        unnorm_rows.append(unnorm_item)
        norm_rows.append(norm_item)

    unnorm_df = pd.DataFrame(unnorm_rows)
    norm_df = pd.DataFrame(norm_rows)

    count_cols = [f"{m}_count" for m in active_metric_names if f"{m}_count" in unnorm_df.columns]
    detected_cols = [f"{m}_detected" for m in active_metric_names if f"{m}_detected" in unnorm_df.columns]
    score_cols = [f"{m}_score" for m in active_metric_names if f"{m}_score" in unnorm_df.columns]

    def _summarise(df: pd.DataFrame) -> pd.DataFrame:
        # Check if required columns exist
        if "Dataset" not in df.columns:
            print("ERROR: Dataset column missing from DataFrame")
            print("Available columns:", df.columns.tolist())
            print("First few rows:")
            print(df.head())
            raise KeyError("Dataset column missing")
        
        scalar_cols = [c for c in df.columns if "__" in c]

        agg: dict[str, str] = {c: "mean" for c in count_cols if c in df.columns}
        agg.update({c: "mean" for c in detected_cols if c in df.columns})
        agg.update({c: "mean" for c in score_cols if c in df.columns})
        agg.update({c: "mean" for c in scalar_cols if c in df.columns})
        if "Word Count" in df.columns:
            agg["Word Count"] = "mean"
        summary = df.groupby(["Dataset", "Status"]).agg(agg).reset_index()
        summary["Checkpoint"] = checkpoint_num
        return summary

    unnorm_summary = _summarise(unnorm_df)
    norm_summary = _summarise(norm_df)

    for data_df, base_dir, summary_df in [
        (unnorm_df, config.UNNORM_DIR, unnorm_summary),
        (norm_df, config.NORM_DIR, norm_summary),
    ]:
        ckpt_out = os.path.join(base_dir, checkpoint_dir_name)
        _ensure(ckpt_out)
        data_df.to_csv(os.path.join(ckpt_out, "detailed_metrics_log.csv"), index=False)
        summary_df.to_csv(os.path.join(ckpt_out, "summary_metrics.csv"), index=False)

    return unnorm_summary, norm_summary


# ---------------------------------------------------------------------------
# Full debug log (written once at the end across all checkpoints)
# ---------------------------------------------------------------------------

def write_debug_logs(all_results: list[dict]) -> None:
    """
    Write two consolidated debug files per dataset across all checkpoints:

    1. ``{dataset}_full_debug.jsonl`` – one JSON object per row combining the
       full source-item fields (observation_1/2, hypothesis_1/2, true_label,
       predicted_label, the model reasoning, correct) with the evaluated metric
       results.  ``timestamp`` and ``run_id`` are intentionally omitted so the
       file is stable across re-runs.

    2. ``{dataset}_full_debug_{RUN_ID}.csv`` – same content as a spreadsheet,
       without ``timestamp`` / ``run_id`` columns, with the source-item fields
       prepended.
    """
    import os
    
    by_dataset: dict[str, list[dict]] = {}
    for r in all_results:
        ds = r.get("dataset", "unknown")
        by_dataset.setdefault(ds, []).append(r)

    for ds, records in by_dataset.items():
        rows = []
        jsonl_lines: list[str] = []

        for rec in records:
            status = rec.get("status", "")

            # All raw source-item fields, exactly as they appear in the JSON
            # file (works for ART, copa_guess_effect, or any other schema).
            raw_item: dict = rec.get("item") or {}

            # ── JSONL record ───────────────────────────────────────────────
            jsonl_obj: dict = {
                **raw_item,          # all source fields first
                "raw_question": _raw_question(rec),
                "checkpoint":  rec.get("checkpoint"),
                "dataset":     rec.get("dataset"),
                "problem_id":  rec.get("pid"),
                "word_count":  rec.get("word_count"),
                "metrics": {
                    mname: {
                        "type":          mdata.get("type"),
                        "detected":      mdata.get("detected"),
                        "score":         mdata.get("score"),  # primary for scorebased
                        "example_count": mdata.get("example_count"),
                        "analysis":      mdata.get("reasoning"),
                        "examples":      mdata.get("examples", []),
                        "scalar_metrics": mdata.get("scalar_metrics") or {},
                        "normalized_scalar_metrics": mdata.get("normalized_scalar_metrics") or {},
                        "tokens":        mdata.get("tokens") or {},
                        "error":         mdata.get("error") or "",
                    }
                    for mname, mdata in (rec.get("metrics") or {}).items()
                },
            }
            jsonl_lines.append(json.dumps(jsonl_obj, ensure_ascii=False))

            # ── CSV row ────────────────────────────────────────────────────
            row: dict = {
                "checkpoint": rec.get("checkpoint"),
                "dataset":    rec.get("dataset"),
                "problem_id": rec.get("pid"),
                "status":     status,
                **raw_item,          # all source fields
                "raw_question": _raw_question(rec),
                "word_count": rec.get("word_count"),
            }
            for mname, mdata in (rec.get("metrics") or {}).items():
                tok = mdata.get("tokens") or {}
                row[f"{mname}_type"]             = mdata.get("type")
                row[f"{mname}_detected"]         = mdata.get("detected")
                _score_val = mdata.get("score")
                if _score_val is not None:
                    row[f"{mname}_score"] = _score_val  # primary for scorebased
                row[f"{mname}_example_count"]    = mdata.get("example_count")
                row[f"{mname}_analysis"]         = mdata.get("reasoning")
                row[f"{mname}_examples"]         = _safe_json(mdata.get("examples", []))
                row[f"{mname}_scalar_metrics"]   = _safe_json(mdata.get("scalar_metrics") or {})
                row[f"{mname}_normalized_scalar_metrics"] = _safe_json(mdata.get("normalized_scalar_metrics") or {})
                row[f"{mname}_tokens_input"]     = tok.get("input")
                row[f"{mname}_tokens_output"]    = tok.get("output")
                # Only written when the model separates reasoning tokens
                if "reasoning" in tok:
                    row[f"{mname}_tokens_reasoning"] = tok["reasoning"]
                if "cached_input" in tok:
                    row[f"{mname}_tokens_cached_input"] = tok["cached_input"]
                row[f"{mname}_error"]            = mdata.get("error") or ""
            rows.append(row)

        # Write JSONL
        jsonl_path = os.path.join(config.LOG_DIR, f"{ds}_full_debug.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(jsonl_lines) + ("\n" if jsonl_lines else ""))
        print(f"[OK] Full-debug JSONL → {jsonl_path}  ({len(jsonl_lines)} records)")

        # Write CSV
        csv_path = os.path.join(config.LOG_DIR, f"{ds}_full_debug_{config.RUN_ID}.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[OK] Full-debug CSV  → {csv_path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Run configuration snapshot
# ---------------------------------------------------------------------------

def write_config_snapshot(
    checkpoint_dirs: list[str],
    active_metric_names: list[str],
    active_datasets: list[str],
) -> None:
    """Write a JSON snapshot of the current run configuration to the results folder.

    The file is named ``run_config_{RUN_ID}.json`` and is placed in
    ``config.BASE_OUTPUT_DIR``.  It captures every setting that materially
    affects the results so that any run can be reproduced or audited later.

    Parameters
    ----------
    checkpoint_dirs:
        The resolved (post-exclusion) checkpoint directories being evaluated.
    active_metric_names:
        Names of the metrics actually used in this run.
    active_datasets:
        Names of the datasets being evaluated (empty = all discovered).
    """
    _ensure(config.BASE_OUTPUT_DIR)

    snapshot: dict[str, Any] = {
        "run_id": config.RUN_ID,
        "run_date": config.RUN_ID[:8],          # YYYYMMDD prefix
        # ── Judge model ───────────────────────────────────────────────────
        "judge_model": config.JUDGE_MODEL,
        "reasoning_effort": config.REASONING_EFFORT,
        "api_base_url": config.GEMINI_BASE_URL if config.JUDGE_MODEL.lower().startswith("gemini") else config.OPENAI_BASE_URL,
        "api_timeout_s": config.API_TIMEOUT,
        "api_max_retries": config.API_MAX_RETRIES,
        # ── Sampling ─────────────────────────────────────────────────────
        "n_samples": config.N_SAMPLES,
        "random_seed": config.RANDOM_SEED,
        "max_workers": config.MAX_WORKERS,
        # ── Scope ────────────────────────────────────────────────────────
        "active_metrics": active_metric_names,
        "active_datasets": active_datasets if active_datasets else ["(all discovered)"],
        "excluded_checkpoints": list(config.EXCLUDED_CHECKPOINTS),
        "evaluated_checkpoints": [os.path.basename(os.path.normpath(d)) for d in checkpoint_dirs],
        # ── I/O ──────────────────────────────────────────────────────────
        "output_base_dir": config.BASE_OUTPUT_DIR,
        "unnorm_dir": config.UNNORM_DIR,
        "norm_dir": config.NORM_DIR,
        "log_dir": config.LOG_DIR,
        "comparison_log_dir": os.path.join(config.BASE_OUTPUT_DIR, "comparison_logs"),
        "latex_slides_dir": os.path.join(config.BASE_OUTPUT_DIR, "latex_slides"),
        "clear_previous_outputs": config.CLEAR_PREVIOUS_OUTPUTS,
    }

    out_path = os.path.join(config.BASE_OUTPUT_DIR, f"run_config_{config.RUN_ID}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, indent=2, ensure_ascii=False)

    print(f"[OK] Config snapshot  → {out_path}")
