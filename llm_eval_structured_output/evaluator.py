"""
evaluator.py
------------
Orchestrates per-item evaluation across all active metrics.

``process_single_item`` is designed to be called from a ThreadPoolExecutor –
it is the only place that calls into the metrics and assembles results.
"""

from __future__ import annotations

import datetime
from typing import Any

import config
from data_loader import extract_full_input
from metrics.base import MetricResult
from metrics.registry import get_active_metrics

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process_single_item(
    task: tuple,
) -> dict[str, Any]:
    """
    Evaluate all active metrics for a single reasoning item.

    Parameters
    ----------
    task : tuple
        ``(dataset_name, checkpoint_num, problem_id, reasoning_text, raw_item)``

    Returns
    -------
    dict
        A flat result dict ready to be assembled into DataFrames / logs.
    """
    dataset_name, checkpoint, pid, reasoning, item = task
    print(f"  [Processing] {dataset_name}  ckpt={checkpoint}  pid={pid}")

    run_id = config.RUN_ID

    # ── Item metadata ──────────────────────────────────────────────────────
    # Since there's no ground truth, we don't classify as correct/incorrect
    status = "No Ground Truth"

    # Word count (no regex – just whitespace split)
    word_count = len(reasoning.split()) if reasoning else 0

    raw_question = extract_full_input(item) if isinstance(item, dict) else ""

    # Build context for metrics that need source input info
    metric_context = {
        "full_input": raw_question,
    }

    # ── Metric evaluation ─────────────────────────────────────────────────
    active_metrics = get_active_metrics(config.ACTIVE_METRICS)
    metric_results: dict[str, MetricResult] = {}

    for metric_name, metric in active_metrics.items():
        result = metric.evaluate(
            reasoning,
            dataset=dataset_name,
            problem_id=str(pid),
            checkpoint=str(checkpoint),
            run_id=run_id,
            context=metric_context,
        )
        metric_results[metric_name] = result
        status_icon = "OK" if not result.error else "FAIL"
        if metric.metric_type == "counting":
            detail = f"count={len(result.examples)}"
        elif metric.metric_type == "coverage":
            detail = f"score={result.score:.2%}  ({sum(1 for e in result.examples if e.get('addressed'))}/{len(result.examples)} details)"
        elif metric.metric_type == "scorebased":
            detail = f"score={result.score}"
        elif metric.metric_type == "graph":
            detail = (
                f"nodes={sum(1 for e in result.examples if e.get('kind') == 'vertex')}  "
                f"edges={sum(1 for e in result.examples if e.get('kind') == 'edge')}"
            )
        else:
            detail = f"detected={result.detected}"
        err_suffix = f"  ERR: {result.error[:80]}" if result.error else ""
        print(f"    [{status_icon:4s}] {metric_name}: {detail}{err_suffix}")

    print(
        f"  [Done] {dataset_name}  ckpt={checkpoint}  pid={pid}  "
        f"({len(metric_results)} metrics, "
        f"{sum(1 for r in metric_results.values() if not r.error)} OK, "
        f"{sum(1 for r in metric_results.values() if r.error)} failed)"
    )

    # ── Assemble output ───────────────────────────────────────────────────
    output = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "run_id": run_id,
        "dataset": dataset_name,
        "checkpoint": checkpoint,
        "pid": pid,
        "status": status,
        "raw_question": raw_question,
        "reasoning": reasoning,
        "word_count": word_count,        # Full raw item – used by write_debug_logs to emit all source fields
        # regardless of dataset schema (ART, copa_guess_effect, etc.)
        "item": item if isinstance(item, dict) else {},        # Per-metric structured results
        "metrics": {
            name: {
                "type": active_metrics[name].metric_type,
                "detected": r.detected,
                "reasoning": r.reasoning,
                "examples": r.examples,
                "example_count": len(r.examples),
                "score": r.score,
                "scalar_metrics": r.scalar_metrics,
                "normalized_scalar_metrics": r.normalized_scalar_metrics,
                "tokens": r.tokens,   # {input, output, reasoning?, cached_input?}
                "error": r.error,
            }
            for name, r in metric_results.items()
        },
    }
    
    # Debug: Print the output structure
    # Only print debug if we actually have metrics
    if output['metrics']:
        print(f"DEBUG: Output keys: {list(output.keys())}")
        print(f"DEBUG: Metrics keys: {list(output['metrics'].keys())}")
        for name, metric_data in output['metrics'].items():
            print(f"DEBUG: {name} - type: {metric_data['type']}, detected: {metric_data['detected']}, error: {metric_data['error']}")
    else:
        print("DEBUG: No metrics in output!")
        print(f"DEBUG: Full output: {output}")
    
    return output
