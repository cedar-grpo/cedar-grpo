#!/usr/bin/env python3
"""
scripts/generate_latex_report.py
---------------------------------
Generate a LaTeX article PDF report comparing a single evaluated item
across two checkpoints for ALL available metrics.

Each metric gets its own section in one continuous document so the reader
can flip through question → reasoning → every metric result side-by-side.

Usage
-----
    python scripts/generate_latex_report.py \\
        --dataset medqa \\
        --problem_id 12 \\
        --checkpoint_a 0 \\
        --checkpoint_b 2560

    # Custom paths:
    python scripts/generate_latex_report.py \\
        --dataset art \\
        --problem_id 5 \\
        --checkpoint_a 0 \\
        --checkpoint_b 2560 \\
        --output results/latex_reports \\
        --log_dir results/llm_logs \\
        --checkpoints_dir checkpoints

Data sources (tried in order)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1.  results/llm_logs/{dataset}_full_debug.jsonl
      → richest source: contains the raw item, reasoning, AND evaluated
        metric results for every checkpoint.
2.  checkpoints/{ckpt_dir}/{dataset}/[all_cases|raw_results_train_all].json
      → fallback: contains the raw item and reasoning only (no metrics).

Output
------
    results/latex_reports/{dataset}_pid{id}_ckpt{a}vs{b}_report.tex

Document layout
---------------
Title       Problem #<pid>  [<DATASET>]
            Checkpoint <A> vs Checkpoint <B>

§1  Problem Statement
§2  Reasoning Trace
    §2.1  <Checkpoint A>
    §2.2  <Checkpoint B>
§3  Metric Analysis  (one subsection per metric; two sub-sub-sections per checkpoint)
    §3.1  Backtracking  (counting)           §3.1.1  Checkpoint A
                                             §3.1.2  Checkpoint B
    §3.2  Branchiness   (counting)           ...
    §3.3  Uncertainty Markers  (counting)
    §3.4  Prior         (counting)
    §3.5  Differential Elimination  (counting)
    §3.6  Observation Coverage  (coverage)
    §3.7  Evidence-Explanation Directionality  (binary)
    §3.8  Evidence-Explanation Directionality Score  (score-based)
    §3.9  Rationale Graph  (graph)
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Allow running the script from the repo root without installing the package
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from data_loader import extract_full_input  # noqa: E402

# ---------------------------------------------------------------------------
# Prompt template imports (to display the exact prompt sent to the judge LLM)
# ---------------------------------------------------------------------------
try:
    from prompts.counting.backtracking import USER_PROMPT_TEMPLATE as _PT_backtracking
    from prompts.counting.branchiness import USER_PROMPT_TEMPLATE as _PT_branchiness
    from prompts.counting.uncertainty_markers import USER_PROMPT_TEMPLATE as _PT_uncertainty_markers
    from prompts.counting.prior import USER_PROMPT_TEMPLATE as _PT_prior
    from prompts.counting.differential_elimination import USER_PROMPT_TEMPLATE as _PT_differential_elimination
    from prompts.coverage.observation_coverage import USER_PROMPT_TEMPLATE as _PT_observation_coverage
    from prompts.binary.evidence_explanation_directionality import USER_PROMPT_TEMPLATE as _PT_evidence_explanation_directionality
    from prompts.scorebased.evidence_explanation_directionality_scorebased import USER_PROMPT_TEMPLATE as _PT_evidence_explanation_directionality_scorebased
    from prompts.graph_structure.rationale_graph import USER_PROMPT_TEMPLATE as _PT_rationale_graph
except ImportError:
    _PT_backtracking = _PT_branchiness = _PT_uncertainty_markers = _PT_prior = None
    _PT_differential_elimination = _PT_observation_coverage = None
    _PT_evidence_explanation_directionality = None
    _PT_evidence_explanation_directionality_scorebased = None
    _PT_rationale_graph = None

_METRIC_USER_PROMPTS: dict[str, str | None] = {
    "backtracking":                                   _PT_backtracking,
    "branchiness":                                    _PT_branchiness,
    "uncertainty_markers":                            _PT_uncertainty_markers,
    "prior":                                          _PT_prior,
    "differential_elimination":                       _PT_differential_elimination,
    "observation_coverage":                           _PT_observation_coverage,
    "evidence_explanation_directionality":            _PT_evidence_explanation_directionality,
    "evidence_explanation_directionality_scorebased": _PT_evidence_explanation_directionality_scorebased,
    "rationale_graph":                                _PT_rationale_graph,
}


# ============================================================
# LaTeX helpers
# ============================================================

_LATEX_ESCAPE = str.maketrans(
    {
        "&":  r"\&",
        "%":  r"\%",
        "$":  r"\$",
        "#":  r"\#",
        "_":  r"\_",
        "{":  r"\{",
        "}":  r"\}",
        "~":  r"\textasciitilde{}",
        "^":  r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
)


def esc(text: str) -> str:
    """Escape a plain string for safe use inside LaTeX body text."""
    if not isinstance(text, str):
        text = str(text)
    return text.translate(_LATEX_ESCAPE)


def wrap(text: str, width: int = 100) -> str:
    """Escape and soft-wrap long lines for LaTeX body flow."""
    escaped = esc(text)
    lines = escaped.splitlines()
    out: list[str] = []
    for line in lines:
        if len(line) <= width:
            out.append(line)
        else:
            out.extend(textwrap.wrap(line, width=width, break_long_words=True))
    return "\n".join(out)


def _bool_mark(val: bool) -> str:
    return r"\cmark" if val else r"\xmark"


# ============================================================
# Data loading  (identical logic to original script)
# ============================================================

def _ckpt_dir_name(checkpoint_num: int) -> str:
    return "raw_model" if checkpoint_num == 0 else f"checkpoint-{checkpoint_num}"


def _ckpt_json_filename(checkpoint_num: int) -> str:
    return "raw_results_train_all.json" if checkpoint_num == 0 else "all_cases.json"


def _load_raw_checkpoint_item(
    checkpoints_dir: str,
    dataset: str,
    checkpoint_num: int,
    problem_id: Any,
) -> dict | None:
    ckpt_dir = os.path.join(checkpoints_dir, _ckpt_dir_name(checkpoint_num))
    if not os.path.isdir(ckpt_dir):
        return None
    ds_dir = None
    for entry in os.listdir(ckpt_dir):
        if entry.lower() == dataset.lower():
            ds_dir = os.path.join(ckpt_dir, entry)
            break
    if ds_dir is None:
        return None

    fpath = os.path.join(ds_dir, _ckpt_json_filename(checkpoint_num))
    if not os.path.exists(fpath):
        return None

    with open(fpath, encoding="utf-8") as fh:
        data = json.load(fh)

    results = data.get("results", data) if isinstance(data, dict) else data
    if not isinstance(results, list):
        return None

    for idx, item in enumerate(results):
        if item.get("problem_id") is None and item.get("sample_id") is None and item.get("qid") is None:
            item["_seq_id"] = idx

    pid_str = str(problem_id)
    for item in results:
        for pid_key in ("problem_id", "sample_id", "qid", "_seq_id"):
            raw_pid = item.get(pid_key)
            if raw_pid is not None and str(raw_pid) == pid_str:
                return item
    return None


def _load_jsonl(log_dir: str, dataset: str) -> list[dict]:
    path = os.path.join(log_dir, f"{dataset}_full_debug.jsonl")
    if not os.path.exists(path):
        return []
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _find_jsonl_record(records: list[dict], checkpoint_num: int, problem_id: Any) -> dict | None:
    pid_str = str(problem_id)
    for rec in records:
        if (
            str(rec.get("checkpoint", "")) == str(checkpoint_num)
            and str(rec.get("problem_id", "")) == pid_str
        ):
            return rec
    return None


def _extract_reasoning(item: dict) -> str:
    ft = item.get("finetuned")
    if isinstance(ft, dict):
        r = ft.get("reasoning")
        if r:
            return r
    if isinstance(ft, str) and ft:
        return ft
    r = item.get("reasoning") or item.get("full_response", "")
    return r or ""


# ============================================================
# Question section
# ============================================================

def _section_question(item: dict, pid: Any, dataset: str) -> str:
    """Return the §Problem Statement section body."""
    raw_question = item.get("raw_question", "") or extract_full_input(item)
    lines: list[str] = [
        r"\begin{description}",
        r"  \item[Dataset] \texttt{" + esc(dataset.upper()) + r"}",
        r"  \item[Problem ID] \texttt{" + esc(str(pid)) + r"}",
        r"\end{description}",
        r"\vspace{6pt}",
        r"\noindent\textbf{Question / Observation:}\\[4pt]",
        r"\begin{quote}",
        r"  \small " + wrap(raw_question),
        r"\end{quote}",
    ]

    # MedQA: try to split the embedded options dict for cleaner rendering
    last_brace = raw_question.rfind("{")
    if last_brace > 0:
        stem = raw_question[:last_brace].strip()
        dict_part = raw_question[last_brace:].strip()
        try:
            opts = ast.literal_eval(dict_part)
            if isinstance(opts, dict):
                lines = [
                    r"\begin{description}",
                    r"  \item[Dataset] \texttt{" + esc(dataset.upper()) + r"}",
                    r"  \item[Problem ID] \texttt{" + esc(str(pid)) + r"}",
                    r"\end{description}",
                    r"\vspace{6pt}",
                    r"\noindent\textbf{Question stem:}\\[4pt]",
                    r"\begin{quote}",
                    r"  \small " + wrap(stem),
                    r"\end{quote}",
                    r"\vspace{4pt}",
                    r"\noindent\textbf{Answer choices:}",
                    r"\begin{enumerate}[label=\Alph*.]",
                ]
                for letter in sorted(opts.keys()):
                    lines.append(
                        r"  \item[\textbf{" + esc(str(letter)) + r"}]"
                        + r" \small " + wrap(str(opts[letter]))
                    )
                lines.append(r"\end{enumerate}")
        except Exception:
            pass

    # Correct answer if available
    correct = item.get("correct_answer") or item.get("label") or item.get("answer")
    if correct is not None:
        lines += [
            r"\vspace{4pt}",
            r"\noindent\textbf{Correct answer:} \texttt{" + esc(str(correct)) + r"}",
        ]

    return "\n".join(lines)


# ============================================================
# Reasoning section
# ============================================================

def _format_reasoning(reasoning: str) -> str:
    if not reasoning:
        return r"\textit{(no reasoning trace found)}"

    cleaned = re.sub(r"^\s*<reasoning>\s*", "", reasoning, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*</reasoning>.*$",  "", cleaned,  flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"\s*<answer>.*?</answer>\s*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = cleaned.strip()

    if not cleaned:
        return r"\textit{(empty reasoning trace)}"

    paragraphs = re.split(r"\n{2,}", cleaned)
    tex_paras: list[str] = []
    for para in paragraphs:
        lines = [esc(ln) for ln in para.splitlines() if ln.strip()]
        if lines:
            tex_paras.append(" ".join(lines))

    return r"\small " + "\n\n\\medskip\n".join(tex_paras)


# ============================================================
# Per-metric-type formatters  (article style — no frame width limits)
# ============================================================

# ── Binary ──────────────────────────────────────────────────────────────────

def _format_binary(mdata: dict) -> str:
    """
    Binary metric (e.g. evidence_explanation_directionality).

    Fields:
      detected   : bool   — is the phenomenon present?
      reasoning  : str    — step-by-step justification
      evidence   : str    — supporting quote (if detected)
    """
    detected: bool = bool(mdata.get("detected", False))
    reasoning: str = mdata.get("reasoning", mdata.get("analysis", ""))
    evidence:  str = mdata.get("evidence", "")
    examples: list = mdata.get("examples", [])

    lines: list[str] = [
        r"\begin{center}",
        r"  {\LARGE " + _bool_mark(detected) + r"}\\[4pt]",
        r"  \textbf{Detected:} \quad \texttt{" + esc(str(detected)) + r"}",
        r"\end{center}",
        r"\vspace{8pt}",
        r"\noindent\textbf{Reasoning / Justification:}\\[4pt]",
        r"\begin{quote}\small " + wrap(reasoning) + r"\end{quote}",
    ]

    # Primary evidence field
    if evidence:
        lines += [
            r"\vspace{4pt}",
            r"\noindent\textbf{Supporting evidence quote:}",
            r"\begin{quote}\small\itshape ``" + wrap(evidence) + r"''\end{quote}",
        ]

    # Additional examples list (if present)
    if examples:
        lines += [r"\vspace{4pt}", r"\noindent\textbf{Additional examples:}",
                  r"\begin{enumerate}\small"]
        for ex in examples:
            excerpt = ex.get("excerpt", ex.get("text", ""))
            explanation = ex.get("explanation", "")
            if excerpt or explanation:
                item_text = ""
                if excerpt:
                    item_text += r"\textbf{Excerpt:} \textit{``" + wrap(excerpt) + r"''}\\"
                if explanation:
                    item_text += r"\textbf{Why:} " + wrap(explanation)
                lines.append(r"  \item " + item_text)
        lines.append(r"\end{enumerate}")

    return "\n".join(lines)


# ── Counting ─────────────────────────────────────────────────────────────────

def _format_counting(mdata: dict) -> str:
    """
    Counting metric (backtracking, branchiness, uncertainty_markers, prior,
    differential_elimination).

    Fields:
      analysis / overall_analysis  : str        — holistic commentary
      examples                     : list[dict] — each has excerpt + explanation
      example_count                : int        — may be pre-computed
    """
    analysis: str  = mdata.get("analysis", mdata.get("overall_analysis", mdata.get("reasoning", "")))
    examples: list = mdata.get("examples", [])
    count: int     = mdata.get("example_count", len(examples))

    lines: list[str] = [
        r"\begin{center}",
        r"  \textbf{Instance count:} {\Large\bfseries " + esc(str(count)) + r"}",
        r"\end{center}",
        r"\vspace{6pt}",
        r"\noindent\textbf{Overall analysis:}\\[4pt]",
        r"\begin{quote}\small " + wrap(analysis) + r"\end{quote}",
    ]

    if examples:
        lines += [
            r"\vspace{8pt}",
            r"\noindent\textbf{Instances found (" + esc(str(count)) + r"):}",
            r"\begin{enumerate}\small",
        ]
        for ex in examples:
            excerpt     = wrap(ex.get("excerpt", ex.get("text", "")), 90)
            explanation = wrap(ex.get("explanation", ""), 90)
            entry = r"  \item"
            if excerpt:
                entry += r" \textbf{Excerpt:} \textit{``" + excerpt + r"''}\\"
            if explanation:
                entry += r"\\[2pt]  \quad\textbf{Why:} " + explanation
            lines.append(entry)
        lines.append(r"\end{enumerate}")
    else:
        lines += [
            r"\vspace{6pt}",
            r"\begin{center}\textit{No instances found.}\end{center}",
        ]

    return "\n".join(lines)


# ── Coverage ─────────────────────────────────────────────────────────────────

def _format_coverage(mdata: dict) -> str:
    """
    Coverage metric (observation_coverage).

    Fields:
      examples / observation_details : list[dict]
          detail    : str   — one specific fact from the observation
          addressed : bool  — was it explicitly addressed?
          evidence  : str   — supporting quote
      analysis / overall_analysis    : str
    """
    analysis: str  = mdata.get("analysis", mdata.get("overall_analysis", mdata.get("reasoning", "")))
    examples: list = mdata.get("examples", [])

    n_total    = len(examples)
    n_addressed = sum(1 for e in examples if e.get("addressed", False))
    score_pct  = f"{n_addressed / n_total * 100:.1f}\\%" if n_total > 0 else r"N/A"

    lines: list[str] = [
        r"\begin{center}",
        r"  \textbf{Coverage score:} {\Large\bfseries "
        + esc(f"{n_addressed}/{n_total}") + r" = " + score_pct + r"}",
        r"\end{center}",
        r"\vspace{6pt}",
        r"\noindent\textbf{Overall analysis:}\\[4pt]",
        r"\begin{quote}\small " + wrap(analysis) + r"\end{quote}",
    ]

    if examples:
        lines += [
            r"\vspace{8pt}",
            r"\noindent\textbf{Observation-detail breakdown:}\\[4pt]",
            r"\begin{longtable}{>{\centering\arraybackslash}p{0.8cm} p{7cm} p{6.5cm}}",
            r"  \toprule",
            r"  \textbf{OK?} & \textbf{Observation detail} & \textbf{Evidence from reasoning} \\",
            r"  \midrule \endhead",
        ]
        for ex in examples:
            addressed = ex.get("addressed", False)
            detail    = wrap(ex.get("detail",    ""), 70)
            evidence  = wrap(ex.get("evidence",  ""), 60)
            mark = _bool_mark(addressed)
            ev_cell = r"\small\itshape " + evidence if evidence else r"\textit{---}"
            lines.append(
                f"  {mark} & \\small {detail} & {ev_cell} \\\\"
            )
        lines += [r"  \bottomrule", r"\end{longtable}"]
    else:
        lines += [
            r"\begin{center}\textit{No observation details found.}\end{center}",
        ]

    return "\n".join(lines)


# ── Score-based ───────────────────────────────────────────────────────────────

_SCORE_LABELS = {
    0.0: r"Backward / circular / missing directionality",
    0.5: r"Ambiguous directionality",
    1.0: r"Correct directionality (observation $\rightarrow$ explanation)",
}


def _format_scorebased(mdata: dict) -> str:
    """
    Score-based metric (evidence_explanation_directionality_scorebased).

    Fields:
      score      : float  — one of {0.0, 0.5, 1.0}
      reasoning  : str    — 1–2 sentence justification
      detected   : bool   — True when score > 0.0
    """
    raw_score = mdata.get("score", mdata.get("directionality_score"))
    try:
        score = float(raw_score) if raw_score is not None else None
    except (ValueError, TypeError):
        score = None

    reasoning: str = mdata.get("reasoning", mdata.get("reasoning_analysis", mdata.get("analysis", "")))

    if score is not None:
        score_label = _SCORE_LABELS.get(score, esc(str(score)))
        score_txt = esc(f"{score:.1f}")
    else:
        score_label = r"\textit{not available}"
        score_txt = r"—"

    lines: list[str] = [
        r"\begin{center}",
        r"  \textbf{Directionality score:} {\Large\bfseries " + score_txt + r"} / 1.0",
        r"  \\[4pt]",
        r"  \textit{" + score_label + r"}",
        r"\end{center}",
        r"\vspace{8pt}",
        r"\noindent\textbf{Analysis:}\\[4pt]",
        r"\begin{quote}\small " + wrap(reasoning) + r"\end{quote}",
    ]

    return "\n".join(lines)


# ── Graph structure ───────────────────────────────────────────────────────────

_GRAPH_STAT_LABELS: dict[str, str] = {
    "mean_out_degree":           "Mean out-degree",
    "max_out_degree":            "Max out-degree",
    "std_out_degree":            "Std out-degree",
    "mean_in_degree":            "Mean in-degree",
    "average_depth_length":      "Average depth length",
    "maximum_depth":             "Maximum depth",
    "std_depth":                 "Std depth",
    "depth_to_width_ratio":      "Depth-to-width ratio",
    "in_degree_skewness":        "In-degree skewness",
    "in_degree_centralization":  "In-degree centralization",
    "number_of_sink_nodes":      "Number of sink nodes",
    "number_of_source_nodes":    "Number of source nodes",
    "cycle_count":               "Cycle count",
    "number_of_isolated_subgraphs": "Isolated subgraphs",
    "mean_betweenness_centrality":  "Mean betweenness centrality",
    "max_betweenness_centrality":   "Max betweenness centrality",
}


def _format_graph(mdata: dict) -> str:
    """
    Graph-structure metric (rationale_graph).

    Raw data (from MetricResult.raw) contains:
      general_reasoning   : str
      vertices            : list[dict]  — vertex_id, label, description, text_correspondence
      edges               : list[dict]  — source_vertex_label, target_vertex_label,
                                          edge_label, description, text_correspondence
      scalar_metrics      : dict[str, float]
      normalized_scalar_metrics : dict[str, float]
    """
    general_reasoning: str = mdata.get("reasoning", mdata.get("analysis", ""))
    raw: dict = mdata.get("raw", mdata)  # fall back to the whole mdata dict

    vertices: list[dict] = raw.get("vertices", []) or mdata.get("examples_vertices", [])
    edges:    list[dict] = raw.get("edges",    []) or mdata.get("examples_edges", [])

    # Support flat examples list (as stored in MetricResult.examples)
    if not vertices and not edges:
        for ex in mdata.get("examples", []):
            if ex.get("kind") == "vertex":
                vertices.append(ex)
            elif ex.get("kind") == "edge":
                edges.append(ex)

    scalar_metrics: dict = raw.get("scalar_metrics", {}) or {}
    n_vertices = len(vertices)
    n_edges    = len(edges)

    lines: list[str] = [
        r"\noindent\textbf{General reasoning from the LLM evaluator:}\\[4pt]",
        r"\begin{quote}\small " + wrap(general_reasoning) + r"\end{quote}",
        r"\vspace{8pt}",
        # ── Summary statistics table ─────────────────────────────────────
        r"\noindent\textbf{Graph statistics:}\\[4pt]",
        r"\begin{center}",
        r"\begin{tabular}{l r}",
        r"  \toprule",
        r"  \textbf{Statistic} & \textbf{Value} \\",
        r"  \midrule",
        r"  Vertices & " + esc(str(n_vertices)) + r" \\",
        r"  Edges    & " + esc(str(n_edges))    + r" \\",
    ]
    if scalar_metrics:
        lines.append(r"  \midrule")
        for key, label in _GRAPH_STAT_LABELS.items():
            val = scalar_metrics.get(key)
            if val is not None:
                lines.append(
                    r"  " + esc(label) + r" & " + esc(f"{float(val):.3f}") + r" \\"
                )
    lines += [r"  \bottomrule", r"\end{tabular}", r"\end{center}"]

    # ── Vertices ──────────────────────────────────────────────────────────
    if vertices:
        lines += [
            r"\vspace{8pt}",
            r"\noindent\textbf{Vertices (" + esc(str(n_vertices)) + r"):}\\[4pt]",
            r"\begin{longtable}{>{\bfseries}p{1cm} p{3cm} p{5cm} p{5.5cm}}",
            r"  \toprule",
            r"  ID & Label & Description & Text correspondence \\",
            r"  \midrule \endhead",
        ]
        for v in vertices:
            vid   = esc(str(v.get("vertex_id", v.get("id", "—"))))
            label = esc(str(v.get("label", "—")))
            desc  = wrap(str(v.get("description", "")), 60)
            tc    = wrap(str(v.get("text_correspondence", "")), 60)
            lines.append(
                f"  {vid} & \\small {label} & \\small {desc} & \\small\\itshape {tc} \\\\"
            )
        lines += [r"  \bottomrule", r"\end{longtable}"]

    # ── Edges ─────────────────────────────────────────────────────────────
    if edges:
        lines += [
            r"\vspace{8pt}",
            r"\noindent\textbf{Edges (" + esc(str(n_edges)) + r"):}\\[4pt]",
            r"\begin{longtable}{p{3.5cm} p{2.5cm} p{4cm} p{4.5cm}}",
            r"  \toprule",
            r"  \textbf{From $\rightarrow$ To} & \textbf{Relation} & \textbf{Description} & \textbf{Text correspondence} \\",
            r"  \midrule \endhead",
        ]
        for e in edges:
            src   = esc(str(e.get("source", e.get("source_vertex_label", "—"))))
            tgt   = esc(str(e.get("target", e.get("target_vertex_label", "—"))))
            elbl  = esc(str(e.get("edge_label", "—")))
            desc  = wrap(str(e.get("description", "")), 50)
            tc    = wrap(str(e.get("text_correspondence", "")), 50)
            lines.append(
                f"  \\small {src} $\\rightarrow$ {tgt} & \\small {elbl} & \\small {desc} & \\small\\itshape {tc} \\\\"
            )
        lines += [r"  \bottomrule", r"\end{longtable}"]

    return "\n".join(lines)


# ============================================================
# Dispatcher
# ============================================================

def _format_metric(mdata: dict | None) -> str:
    """Format any metric type; return the section body as a LaTeX string."""
    if mdata is None or mdata.get("error"):
        err = mdata.get("error", "Metric data not available.") if mdata else "Metric data not available."
        return r"\begin{center}\textcolor{red}{\textbf{Error:} " + esc(err) + r"}\end{center}"

    mtype = mdata.get("type", "")

    if mtype == "binary":
        return _format_binary(mdata)
    if mtype == "counting":
        return _format_counting(mdata)
    if mtype == "coverage":
        return _format_coverage(mdata)
    if mtype in ("scorebased", "score_based"):
        return _format_scorebased(mdata)
    if mtype == "graph":
        return _format_graph(mdata)

    # Unknown — graceful raw dump
    lines = [r"\noindent\textbf{Raw metric data} (type: \texttt{" + esc(mtype or "unknown") + r"}):"]
    lines.append(r"\begin{description}\small")
    for k, v in mdata.items():
        if k not in ("tokens",):
            lines.append(
                r"  \item[\texttt{" + esc(str(k)) + r"}] " + wrap(str(v), 100)
            )
    lines.append(r"\end{description}")
    return "\n".join(lines)


# ============================================================
# Prompt display helpers
# ============================================================

def _fill_prompt(template: str | None, dataset: str, reasoning: str, full_input: str = "") -> str:
    """Fill a USER_PROMPT_TEMPLATE with the actual runtime values."""
    if not template:
        return "(prompt template not available)"
    filled = template
    filled = filled.replace("{dataset}", dataset)
    filled = filled.replace("{text}", reasoning)
    filled = filled.replace("{full_input}", full_input)
    # Remove any unfilled placeholders gracefully
    filled = re.sub(r"\{[a-z_]+\}", "(not provided)", filled)
    return filled


def _format_prompt_box(prompt_text: str) -> str:
    """Render a filled prompt as a shaded framed LaTeX box (mdframed)."""
    if not prompt_text:
        return (
            r"\begin{mdframed}[style=promptbox]\small"
            r"\textit{(prompt not available)}\end{mdframed}"
        )
    # Split into paragraphs; escape each line, join with LaTeX paragraph breaks
    paragraphs = re.split(r"\n{2,}", prompt_text.strip())
    tex_parts: list[str] = []
    for para in paragraphs:
        lines = [esc(ln) for ln in para.splitlines()]
        joined = "\\\\\ \n".join(lines)  # line breaks within a paragraph
        tex_parts.append(joined)
    body = "\n\n".join(tex_parts)
    return (
        "\\begin{mdframed}[style=promptbox]\n"
        "\\small\n"
        + body + "\n"
        "\\end{mdframed}"
    )


# ============================================================
# Document preamble
# ============================================================

ARTICLE_PREAMBLE = r"""\documentclass[12pt,a4paper]{article}

% ── Encoding & fonts ───────────────────────────────────────────────────────
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{microtype}

% ── Page geometry ──────────────────────────────────────────────────────────
\usepackage[
  top=2.2cm, bottom=2.5cm,
  left=2.5cm, right=2.5cm
]{geometry}

% ── Tables ─────────────────────────────────────────────────────────────────
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{longtable}
\usepackage{array}

% ── List customisation ─────────────────────────────────────────────────────
\usepackage{enumitem}

% ── Math & symbols ─────────────────────────────────────────────────────────
\usepackage{amsmath}
\usepackage{pifont}

% ── Colours & hyperlinks ───────────────────────────────────────────────────
\usepackage{xcolor}
\usepackage[colorlinks=true, linkcolor=blue!60!black, urlcolor=blue!60!black]{hyperref}

% ── Section styling ────────────────────────────────────────────────────────
\usepackage{titlesec}
\titleformat{\section}{\large\bfseries}{\thesection.}{0.5em}{}[\titlerule]
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection.}{0.5em}{}
\titleformat{\subsubsection}{\normalsize\itshape}{\thesubsubsection.}{0.5em}{}

% ── Header / footer ────────────────────────────────────────────────────────
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0.4pt}
\fancyhead[L]{\small\leftmark}
\fancyhead[R]{\small\thepage}

% ── Prompt boxes ──────────────────────────────────────────────────────────
\usepackage{mdframed}
\mdfdefinestyle{promptbox}{%
  backgroundcolor=gray!7,
  linecolor=gray!45,
  linewidth=0.5pt,
  innerleftmargin=8pt,
  innerrightmargin=8pt,
  innertopmargin=6pt,
  innerbottommargin=6pt
}

% ── Misc ───────────────────────────────────────────────────────────────────
\usepackage{parskip}
\setlength{\parskip}{6pt}

% ── Checkmark / cross marks ────────────────────────────────────────────────
\newcommand{\cmark}{\textcolor{green!50!black}{\ding{51}}}
\newcommand{\xmark}{\textcolor{red!80!black}{\ding{55}}}
"""

# ── Human-readable metric type labels ──────────────────────────────────────
_MTYPE_LABELS = {
    "counting":   "Counting",
    "binary":     "Binary (yes/no)",
    "coverage":   "Coverage",
    "scorebased": "Score-based",
    "score_based": "Score-based",
    "graph":      "Graph structure",
}

# ── Brief one-line description for every known metric ──────────────────────
_METRIC_DESCRIPTIONS: dict[str, str] = {
    "backtracking": (
        "Counts moments where the model explicitly revises or rejects a previous "
        "reasoning step, indicating self-correction."
    ),
    "branchiness": (
        "Counts distinct hypothesis branches or alternative paths explored before "
        "converging on a conclusion."
    ),
    "uncertainty_markers": (
        "Counts hedging expressions (e.g.~\\textit{perhaps}, \\textit{might}, "
        "\\textit{likely}) that signal epistemic humility rather than overconfidence."
    ),
    "prior": (
        "Counts explicit references to background knowledge, base-rates, or prior "
        "probability judgements incorporated into the reasoning."
    ),
    "differential_elimination": (
        "Counts moments where competing hypotheses are explicitly ruled out by "
        "testing them against available evidence before accepting the best explanation."
    ),
    "observation_coverage": (
        "Measures the fraction of specific observation details in the prompt that the "
        "chosen hypothesis explicitly addresses.  Score~=~addressed / total."
    ),
    "evidence_explanation_directionality": (
        "Binary check: does the model reason \\textit{from} observations "
        "\\textit{to} explanations (correct direction) rather than assuming the answer "
        "and working backwards?"
    ),
    "evidence_explanation_directionality_scorebased": (
        "Graded version of directionality awareness: "
        "1.0~=~correct, 0.5~=~ambiguous, 0.0~=~backward/circular/missing."
    ),
    "rationale_graph": (
        "Extracts a directed graph of reasoning steps from the trace and computes "
        "structural metrics (depth, branching, cycles, betweenness centrality, \\ldots)."
    ),
}


def _checkpoint_label(num: int) -> str:
    return "raw\\_model (checkpoint 0)" if num == 0 else f"checkpoint-{num}"


def _section_label(num: int) -> str:
    """Safe LaTeX label component for a checkpoint number."""
    return "raw" if num == 0 else str(num)


# ============================================================
# Full document assembly
# ============================================================

def build_latex(
    dataset: str,
    pid: Any,
    ckpt_a: int,
    ckpt_b: int | None,
    item_a: dict,
    item_b: dict | None,
    reasoning_a: str,
    reasoning_b: str | None,
    all_metrics_a: dict[str, dict | None],
    all_metrics_b: dict[str, dict | None],
) -> str:
    """Assemble the complete .tex source for the analysis report."""

    ds_esc  = esc(dataset.upper())
    pid_esc = esc(str(pid))
    lbl_a   = _checkpoint_label(ckpt_a)
    lbl_b   = _checkpoint_label(ckpt_b) if ckpt_b is not None else ""

    full_input_a = item_a.get("raw_question", "") or extract_full_input(item_a)
    full_input_b = item_b.get("raw_question", "") or extract_full_input(item_b) if item_b else ""

    # Collect union of all metric names present in either checkpoint
    all_metric_names: list[str] = []
    seen: set[str] = set()
    for mname in list(all_metrics_a.keys()) + list(all_metrics_b.keys()):
        if mname not in seen:
            all_metric_names.append(mname)
            seen.add(mname)

    doc = ARTICLE_PREAMBLE
    doc += "\n\\begin{document}\n"

    # ── Title ─────────────────────────────────────────────────────────────
    title_sub = (r"{\normalsize " + lbl_a + r" \quad vs \quad " + lbl_b + r"}") if ckpt_b is not None else (r"{\normalsize " + lbl_a + r"}")
    doc += r"""
\begin{center}
  {\LARGE\bfseries Abductive Reasoning Evaluation Report}\\[8pt]
  {\large Dataset: \texttt{""" + ds_esc + r"""} \quad Problem \#""" + pid_esc + r"""}\\[4pt]
  """ + title_sub + r"""
\end{center}
\vspace{4pt}
\hrule
\vspace{12pt}
\tableofcontents
\vspace{12pt}
\hrule
\newpage
"""

    # ── §1  Problem Statement ────────────────────────────────────────────
    doc += "\n\\section{Problem Statement}\n"
    doc += _section_question(item_a, pid, dataset) + "\n"

    # ── §2  Reasoning Traces ─────────────────────────────────────────────
    doc += "\n\\section{Reasoning Traces}\n"

    if ckpt_b is not None:
        doc += f"\n\\subsection{{{esc(lbl_a)}}}\n"
    doc += _format_reasoning(reasoning_a) + "\n"

    if ckpt_b is not None and reasoning_b is not None:
        doc += f"\n\\subsection{{{esc(lbl_b)}}}\n"
        doc += _format_reasoning(reasoning_b) + "\n"

    # ── §3  Metric Analysis ───────────────────────────────────────────────
    doc += "\n\\section{Metric Analysis}\n"
    
    if ckpt_b is not None:
        doc += (
            r"Each subsection covers one metric.  "
            r"Two sub-subsections give the result for each checkpoint, "
            r"followed by a brief side-by-side summary where applicable." + "\n"
        )
    else:
        doc += (
            r"Each subsection covers one metric, optionally with a prompt box "
            r"and the model's response." + "\n"
        )

    if not all_metric_names:
        doc += (
            r"\begin{center}\textit{No metric data found.  "
            r"Run the evaluation pipeline first to populate "
            r"\texttt{results/llm\_logs/<dataset>\_full\_debug.jsonl}.}"
            r"\end{center}" + "\n"
        )
    else:
        for idx, mname in enumerate(all_metric_names, start=1):
            mdata_a = all_metrics_a.get(mname)
            mdata_b = all_metrics_b.get(mname)

            # Infer display name and type
            display_name = esc(mname.replace("_", " ").title())
            mtype = ""
            if mdata_a and mdata_a.get("type"):
                mtype = mdata_a["type"]
            elif mdata_b and mdata_b.get("type"):
                mtype = mdata_b["type"]
            mtype_label = _MTYPE_LABELS.get(mtype, esc(mtype)) if mtype else "unknown"

            # Optional description
            description = _METRIC_DESCRIPTIONS.get(mname, "")

            doc += f"\n\\subsection{{{display_name}}}\n"
            doc += (
                r"\noindent\textit{Metric type:} \textbf{"
                + esc(mtype_label)
                + r"}"
            )
            if description:
                doc += r"  \hfill" + "\n"
                doc += r"\noindent " + description + "\n"
            doc += "\n\n"

            # ── Build filled prompts for each checkpoint ──────────────
            tmpl    = _METRIC_USER_PROMPTS.get(mname)
            prompt_a = _fill_prompt(tmpl, dataset, reasoning_a, full_input_a)
            prompt_b = _fill_prompt(tmpl, dataset, reasoning_b, full_input_b) if reasoning_b else ""

            # ── Checkpoint A ──────────────────────────────────────────
            if ckpt_b is not None:
                doc += f"\\subsubsection{{{esc(lbl_a)}}}\n"
            doc += "\n\\noindent\\textbf{\\sffamily Prompt sent to LLM evaluator:}\\par\\vspace{2pt}\n"
            doc += _format_prompt_box(prompt_a) + "\n\n"
            doc += "\\vspace{4pt}\\noindent\\textbf{\\sffamily Model response:}\\par\\vspace{4pt}\n"
            doc += _format_metric(mdata_a) + "\n"

            # ── Checkpoint B ──────────────────────────────────────────
            if ckpt_b is not None:
                doc += f"\n\\subsubsection{{{esc(lbl_b)}}}\n"
                doc += "\n\\noindent\\textbf{\\sffamily Prompt sent to LLM evaluator:}\\par\\vspace{2pt}\n"
                doc += _format_prompt_box(prompt_b) + "\n\n"
                doc += "\\vspace{4pt}\\noindent\\textbf{\\sffamily Model response:}\\par\\vspace{4pt}\n"
                doc += _format_metric(mdata_b) + "\n"

                # ── Delta summary for counting / coverage / scorebased ────
                doc += _maybe_delta_summary(mname, mtype, mdata_a, mdata_b)

    doc += "\n\\end{document}\n"
    return doc


# ============================================================
# Delta / comparison summary box
# ============================================================

def _maybe_delta_summary(
    mname: str,
    mtype: str,
    mdata_a: dict | None,
    mdata_b: dict | None,
) -> str:
    """
    Return a small LaTeX comparison paragraph if we can compute a meaningful
    delta between checkpoint A and checkpoint B.
    """
    if mdata_a is None or mdata_b is None:
        return ""

    lines: list[str] = []

    if mtype == "counting":
        ex_a = mdata_a.get("examples", [])
        ex_b = mdata_b.get("examples", [])
        cnt_a = mdata_a.get("example_count", len(ex_a))
        cnt_b = mdata_b.get("example_count", len(ex_b))
        delta = int(cnt_b) - int(cnt_a)
        sign  = "+" if delta > 0 else ""
        lines = [
            r"\vspace{6pt}\noindent\rule{\linewidth}{0.4pt}",
            r"\noindent\textbf{Comparison:} ",
            esc(f"{lbl_escape(mname)}: {cnt_a} (ckpt A) → {cnt_b} (ckpt B)   "
                f"[delta: {sign}{delta}]"),
            r"\vspace{4pt}",
        ]

    elif mtype == "coverage":
        def _score(md: dict) -> float | None:
            exs = md.get("examples", [])
            if not exs:
                return None
            addressed = sum(1 for e in exs if e.get("addressed", False))
            return round(addressed / len(exs), 3)

        sa = _score(mdata_a)
        sb = _score(mdata_b)
        if sa is not None and sb is not None:
            delta = round(sb - sa, 3)
            sign  = "+" if delta > 0 else ""
            lines = [
                r"\vspace{6pt}\noindent\rule{\linewidth}{0.4pt}",
                r"\noindent\textbf{Comparison:} ",
                esc(f"Coverage: {sa:.1%} (ckpt A) → {sb:.1%} (ckpt B)   "
                    f"[delta: {sign}{delta:.3f}]"),
                r"\vspace{4pt}",
            ]

    elif mtype in ("scorebased", "score_based"):
        sa_raw = mdata_a.get("score")
        sb_raw = mdata_b.get("score")
        try:
            sa = float(sa_raw) if sa_raw is not None else None  # type: ignore[arg-type]
            sb = float(sb_raw) if sb_raw is not None else None  # type: ignore[arg-type]
        except (ValueError, TypeError):
            sa = sb = None
        if sa is not None and sb is not None:
            delta = round(sb - sa, 2)
            sign  = "+" if delta > 0 else ""
            lines = [
                r"\vspace{6pt}\noindent\rule{\linewidth}{0.4pt}",
                r"\noindent\textbf{Comparison:} ",
                esc(f"Score: {sa:.1f} (ckpt A) → {sb:.1f} (ckpt B)   "
                    f"[delta: {sign}{delta:.2f}]"),
                r"\vspace{4pt}",
            ]

    elif mtype == "binary":
        det_a = bool(mdata_a.get("detected", False))
        det_b = bool(mdata_b.get("detected", False))
        if det_a != det_b:
            change = "gained" if (not det_a and det_b) else "lost"
            lines = [
                r"\vspace{6pt}\noindent\rule{\linewidth}{0.4pt}",
                r"\noindent\textbf{Comparison:} The phenomenon was \textbf{"
                + esc(change) + r"} between checkpoint A and checkpoint B.",
                r"\vspace{4pt}",
            ]

    return ("\n".join(lines) + "\n") if lines else ""


def lbl_escape(s: str) -> str:
    return s.replace("_", " ")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a LaTeX article PDF report for one evaluated item (all metrics).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset",       required=True,  help="Dataset name (e.g. medqa, art)")
    p.add_argument("--problem_id",    required=True,  help="Problem/sample ID from the JSON data")
    p.add_argument("--checkpoint_a",  required=True,  type=int, help="First checkpoint (0 = raw_model)")
    p.add_argument("--checkpoint_b",  required=False, type=int, default=None, help="Second checkpoint (optional)")
    p.add_argument("--output",        default=os.path.join("results", "latex_reports"),
                   help="Output directory for .tex files  [default: results/latex_reports]")
    p.add_argument("--log_dir",       default=os.path.join("results", "llm_logs"),
                   help="Directory with *_full_debug.jsonl files  [default: results/llm_logs]")
    p.add_argument("--checkpoints_dir", default="checkpoints",
                   help="Root checkpoints directory  [default: checkpoints]")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.lower()
    pid_raw = args.problem_id
    ckpt_a  = args.checkpoint_a
    ckpt_b  = args.checkpoint_b

    try:
        pid: Any = int(pid_raw)
    except ValueError:
        pid = pid_raw

    print(f"[generate_latex_report] dataset={dataset}  pid={pid}  "
          f"ckpt_a={ckpt_a}  ckpt_b={ckpt_b}")

    # ── 1. Load JSONL logs ─────────────────────────────────────────────────
    jsonl_records = _load_jsonl(args.log_dir, dataset)
    rec_a = _find_jsonl_record(jsonl_records, ckpt_a, pid)
    rec_b = _find_jsonl_record(jsonl_records, ckpt_b, pid) if ckpt_b is not None else None

    if jsonl_records:
        if rec_a is None:
            rec_a = _find_jsonl_record(jsonl_records, ckpt_a, pid_raw)
        if ckpt_b is not None and rec_b is None:
            rec_b = _find_jsonl_record(jsonl_records, ckpt_b, pid_raw)

    # ── 2. Load raw checkpoint items ───────────────────────────────────────
    item_a = _load_raw_checkpoint_item(args.checkpoints_dir, dataset, ckpt_a, pid)
    item_b = _load_raw_checkpoint_item(args.checkpoints_dir, dataset, ckpt_b, pid) if ckpt_b is not None else None

    if item_a is None and rec_a:
        item_a = rec_a
    if item_b is None and rec_b:
        item_b = rec_b

    if item_a is None and (item_b is None if ckpt_b is not None else True):
        print(
            f"[ERROR] Cannot find problem_id={pid!r} in checkpoint A.\n"
            f"  JSONL dir       : {args.log_dir}\n"
            f"  Checkpoints dir : {args.checkpoints_dir}"
        )
        sys.exit(1)

    item_a = item_a or item_b
    if ckpt_b is not None:
        item_b = item_b or item_a

    # ── 3. Extract reasoning ───────────────────────────────────────────────
    reasoning_a = (rec_a.get("reasoning", "") if rec_a else "") or _extract_reasoning(item_a)
    reasoning_b = ((rec_b.get("reasoning", "") if rec_b else "") or _extract_reasoning(item_b)) if ckpt_b is not None else None

    # ── 4. Collect ALL metric data from JSONL records ──────────────────────
    all_metrics_a: dict[str, dict | None] = {}
    all_metrics_b: dict[str, dict | None] = {}

    if rec_a and rec_a.get("metrics"):
        all_metrics_a = {k: v for k, v in rec_a["metrics"].items()}
    if rec_b and rec_b.get("metrics"):
        all_metrics_b = {k: v for k, v in rec_b["metrics"].items()}

    if not all_metrics_a and not all_metrics_b:
        if jsonl_records:
            print(
                "[WARN] No metric data found in the JSONL records for the requested "
                "problem_id.  The report will contain the question and reasoning only."
            )
        else:
            print(
                f"[WARN] JSONL log not found at "
                f"'{args.log_dir}/{dataset}_full_debug.jsonl'.  "
                "Run the evaluation pipeline first to populate metric results."
            )

    # ── 5. Build LaTeX source ─────────────────────────────────────────────
    latex_source = build_latex(
        dataset=dataset,
        pid=pid,
        ckpt_a=ckpt_a,
        ckpt_b=ckpt_b,
        item_a=item_a,
        item_b=item_b,
        reasoning_a=reasoning_a,
        reasoning_b=reasoning_b,
        all_metrics_a=all_metrics_a,
        all_metrics_b=all_metrics_b,
    )

    # ── 6. Write .tex file ────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    out_fname = f"{dataset}_pid{pid}_ckpt{ckpt_a}vs{ckpt_b}_report.tex" if ckpt_b is not None else f"{dataset}_pid{pid}_ckpt{ckpt_a}_report.tex"
    out_path  = os.path.join(args.output, out_fname)

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(latex_source)

    print(f"[OK] LaTeX report written → {out_path}")
    print(f"     Compile with:  pdflatex \"{out_path}\"")
    print(f"     (run twice for correct table of contents)")


if __name__ == "__main__":
    main()
