#!/usr/bin/env python3
"""
scripts/generate_latex_slides.py
---------------------------------
Generate a LaTeX Beamer presentation comparing a single evaluated item
across two checkpoints for one metric.

Usage
-----
    python scripts/generate_latex_slides.py \\
        --dataset medqa \\
        --problem_id 12 \\
        --checkpoint_a 0 \\
        --checkpoint_b 2560 \\
        --metric backtracking

    # Custom paths:
    python scripts/generate_latex_slides.py \\
        --dataset art \\
        --problem_id 5 \\
        --checkpoint_a 0 \\
        --checkpoint_b 2560 \\
        --metric observation_coverage \\
        --output results/latex_slides \\
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
    results/latex_slides/{dataset}_pid{id}_ckpt{a}vs{b}_{metric}.tex
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

# ============================================================
# LaTeX escaping
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
    """Escape a plain string for use inside LaTeX body text."""
    if not isinstance(text, str):
        text = str(text)
    return text.translate(_LATEX_ESCAPE)


def wrap(text: str, width: int = 90) -> str:
    """Escape and soft-wrap long lines so they fit on a beamer slide."""
    escaped = esc(text)
    lines = escaped.splitlines()
    wrapped_lines: list[str] = []
    for line in lines:
        if len(line) <= width:
            wrapped_lines.append(line)
        else:
            # textwrap on already-escaped text (safe, no further escaping)
            wrapped_lines.extend(textwrap.wrap(line, width=width, break_long_words=True))
    return "\n".join(wrapped_lines)


# ============================================================
# Data loading
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
    """Load a single item from the raw checkpoint JSON file."""
    ckpt_dir = os.path.join(checkpoints_dir, _ckpt_dir_name(checkpoint_num))
    # Dataset directory lookup is case-insensitive
    ds_dir = None
    for entry in os.listdir(ckpt_dir):
        if entry.lower() == dataset.lower():
            ds_dir = os.path.join(ckpt_dir, entry)
            break
    if ds_dir is None:
        return None

    fname = _ckpt_json_filename(checkpoint_num)
    fpath = os.path.join(ds_dir, fname)
    if not os.path.exists(fpath):
        return None

    with open(fpath, encoding="utf-8") as fh:
        data = json.load(fh)

    results = data.get("results", data) if isinstance(data, dict) else data
    if not isinstance(results, list):
        return None

    # Assign sequential IDs if missing (mirrors data_loader.py)
    for idx, item in enumerate(results):
        if item.get("problem_id") is None and item.get("sample_id") is None and item.get("qid") is None:
            item["_seq_id"] = idx

    # Resolve problem_id key variants (use 'is not None' to avoid 0 == falsy)
    pid_str = str(problem_id)
    for item in results:
        for pid_key in ("problem_id", "sample_id", "qid", "_seq_id"):
            raw_pid = item.get(pid_key)
            if raw_pid is not None and str(raw_pid) == pid_str:
                return item
    return None


def _load_jsonl(log_dir: str, dataset: str) -> list[dict]:
    """Load the full_debug JSONL for a dataset (created by write_debug_logs)."""
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
        if str(rec.get("checkpoint", "")) == str(checkpoint_num) and str(rec.get("problem_id", "")) == pid_str:
            return rec
    return None


def _extract_reasoning(item: dict) -> str:
    """Mirror data_loader.extract_reasoning – works on any schema variant."""
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
# Question / problem context formatting
# ============================================================

def _parse_medqa_options(question_str: str) -> tuple[str, dict | None]:
    """
    Split a MedQA question string that embeds an options dict.

    Returns (stem, options_dict).
    The options dict is like {'A': '...', 'B': '...', ...}.
    If no dict is found, returns (question_str, None).
    """
    # Find the last '{' that starts an embedded dict
    last_brace = question_str.rfind("{")
    if last_brace == -1:
        return question_str.strip(), None
    stem = question_str[:last_brace].strip()
    dict_part = question_str[last_brace:].strip()
    try:
        opts = ast.literal_eval(dict_part)
        if isinstance(opts, dict):
            return stem, opts
    except Exception:
        pass
    return question_str.strip(), None


def _format_question_slide(item: dict, pid: Any, dataset: str) -> str:
    """Return the LaTeX body content for the Question slide."""
    raw_question = item.get("raw_question", "") or extract_full_input(item)
    full_input = wrap(raw_question)
    return "\n    ".join(
        [
            r"\textbf{Raw question:}\\[4pt]",
            rf"\small {full_input}",
        ]
    )


# ============================================================
# Reasoning slide body
# ============================================================

def _format_reasoning_slide(reasoning: str) -> str:
    """Return the LaTeX body content for a Reasoning slide."""
    if not reasoning:
        return r"\textit{(no reasoning trace found)}"

    # Strip common XML-style wrappers: <reasoning>, </reasoning>, <answer>…</answer>
    cleaned = re.sub(r"^\s*<reasoning>\s*", "", reasoning, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*</reasoning>.*$",  "", cleaned,  flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"\s*<answer>.*?</answer>\s*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = cleaned.strip()

    if not cleaned:
        return r"\textit{(empty reasoning trace)}"

    # Split into paragraphs, escape each line, join with LaTeX paragraph breaks
    paragraphs = re.split(r"\n{2,}", cleaned)
    tex_paragraphs: list[str] = []
    for para in paragraphs:
        lines = [esc(ln) for ln in para.splitlines() if ln.strip()]
        if lines:
            tex_paragraphs.append(" ".join(lines))

    return r"\small " + "\n\n\\medskip\n".join(tex_paragraphs)


# ============================================================
# Metric slide body (3 types)
# ============================================================

def _bool_mark(val: bool) -> str:
    return r"\cmark" if val else r"\xmark"


def _format_metric_binary(mdata: dict, metric_name: str) -> str:
    """Render a binary metric result as LaTeX."""
    detected: bool = bool(mdata.get("detected", False))
    analysis: str = mdata.get("analysis", mdata.get("reasoning", ""))
    examples: list = mdata.get("examples", [])

    lines: list[str] = [
        r"\begin{center}",
        r"{\Large " + _bool_mark(detected) + r"}\\[4pt]",
        r"\textbf{Detected: } \texttt{" + esc(str(detected)) + r"}",
        r"\end{center}",
        r"\vspace{6pt}",
        r"\textbf{Reasoning:}\\[4pt]",
        r"\small " + wrap(analysis, 85),
    ]

    if examples:
        ev = examples[0]
        excerpt = ev.get("excerpt", ev.get("text", ""))
        if excerpt:
            lines += [
                r"\vspace{6pt}",
                r"\textbf{Evidence quote:}",
                r"\begin{quote}\small\itshape " + wrap(excerpt, 80) + r"\end{quote}",
            ]

    return "\n    ".join(lines)


def _format_metric_counting(mdata: dict, metric_name: str) -> str:
    """Render a counting metric result as LaTeX."""
    analysis: str  = mdata.get("analysis", mdata.get("reasoning", ""))
    examples: list = mdata.get("examples", [])
    count: int     = mdata.get("example_count", len(examples))

    lines: list[str] = [
        r"\begin{center}",
        r"\textbf{Instance count: } {\Large\textbf{" + esc(str(count)) + r"}}",
        r"\end{center}",
        r"\vspace{4pt}",
        r"\textbf{Overall analysis:}\\[4pt]",
        r"\small " + wrap(analysis, 85),
    ]

    if examples:
        lines += [
            r"\vspace{8pt}",
            r"\textbf{Examples found:}",
            r"\begin{enumerate}\small",
        ]
        for ex in examples:
            excerpt     = wrap(ex.get("excerpt", ex.get("text", "")), 75)
            explanation = wrap(ex.get("explanation", ""), 75)
            lines.append(
                r"  \item \textbf{Excerpt:} \textit{``" + excerpt + r"''}\\"
                + "\n        \\textbf{Why:} " + explanation
            )
        lines.append(r"\end{enumerate}")

    return "\n    ".join(lines)


def _format_metric_coverage(mdata: dict, metric_name: str) -> str:
    """Render a coverage metric result as LaTeX."""
    analysis: str  = mdata.get("analysis", mdata.get("reasoning", ""))
    examples: list = mdata.get("examples", [])

    n_total    = len(examples)
    n_addressed = sum(1 for e in examples if e.get("addressed", False))
    score = round(n_addressed / n_total, 3) if n_total > 0 else 0.0

    score_pct = f"{score * 100:.1f}\\%"
    lines: list[str] = [
        r"\begin{center}",
        r"\textbf{Coverage score: } {\Large\textbf{" + esc(f"{n_addressed}/{n_total}") + r" = " + score_pct + r"}}",
        r"\end{center}",
        r"\vspace{4pt}",
        r"\textbf{Overall analysis:}\\[4pt]",
        r"\small " + wrap(analysis, 85),
    ]

    if examples:
        lines += [
            r"\vspace{8pt}",
            r"\textbf{Observation details:}\\[4pt]",
            r"\begin{tabularx}{\textwidth}{>{\centering\arraybackslash}p{0.9cm} X p{4.5cm}}",
            r"\toprule",
            r"\textbf{OK?} & \textbf{Detail} & \textbf{Evidence} \\",
            r"\midrule",
        ]
        for ex in examples:
            addressed = ex.get("addressed", False)
            detail    = wrap(ex.get("detail",    ""), 60)
            evidence  = wrap(ex.get("evidence",  ""), 40)
            mark = _bool_mark(addressed)
            lines.append(f"  {mark} & \\small {detail} & \\small\\itshape {evidence} \\\\")
        lines += [r"\bottomrule", r"\end{tabularx}"]

    return "\n    ".join(lines)


def _format_metric_slide(mdata: dict | None, metric_name: str, metric_type_hint: str) -> str:
    """Dispatch to the right formatter; return body LaTeX."""
    if mdata is None or mdata.get("error"):
        err = mdata.get("error", "Metric data not available.") if mdata else "Metric data not available."
        return r"\textit{\textcolor{red}{Error: " + esc(err) + r"}}"

    mtype = mdata.get("type", metric_type_hint)

    if mtype == "binary":
        return _format_metric_binary(mdata, metric_name)
    if mtype == "counting":
        return _format_metric_counting(mdata, metric_name)
    if mtype == "coverage":
        return _format_metric_coverage(mdata, metric_name)

    # Unknown type — raw dump
    lines = [r"\small\textbf{Raw metric data (unknown type):}\\[4pt]"]
    for k, v in mdata.items():
        if k not in ("tokens",):
            lines.append(rf"\textbf{{{esc(k)}:}} {wrap(str(v), 80)}\\[2pt]")
    return "\n    ".join(lines)


# ============================================================
# Full document assembly
# ============================================================

BEAMER_PREAMBLE = r"""\documentclass{beamer}

% Packages
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{pifont}
\usepackage{microtype}
\usepackage{xcolor}

% Clean minimalist white theme
\usetheme{default}
\setbeamertemplate{navigation symbols}{}
\setbeamercolor{background canvas}{bg=white}
\setbeamercolor{frametitle}{fg=black, bg=white}
\setbeamercolor{normal text}{fg=black}
\setbeamertemplate{frametitle}{%
    \vspace{0.4cm}\textbf{\insertframetitle}\par\vskip-0.15cm\hrulefill}

% Check / cross marks
\newcommand{\cmark}{\textcolor{green!60!black}{\ding{51}}}
\newcommand{\xmark}{\textcolor{red}{\ding{55}}}
"""


def _checkpoint_label(num: int) -> str:
    return "raw\\_model (0)" if num == 0 else f"checkpoint-{num}"


def build_latex(
    dataset: str,
    pid: Any,
    ckpt_a: int,
    ckpt_b: int,
    item_a: dict,
    item_b: dict,
    reasoning_a: str,
    reasoning_b: str,
    metric_name: str,
    mdata_a: dict | None,
    mdata_b: dict | None,
    metric_type_hint: str = "",
) -> str:
    """Assemble the complete .tex document string."""

    ds_esc   = esc(dataset.upper())
    pid_esc  = esc(str(pid))
    m_esc    = esc(metric_name.replace("_", " ").title())
    lbl_a    = _checkpoint_label(ckpt_a)
    lbl_b    = _checkpoint_label(ckpt_b)

    # Use item_a for question (should be identical across checkpoints)
    q_body = _format_question_slide(item_a, pid, dataset)
    r_body_a = _format_reasoning_slide(reasoning_a)
    r_body_b = _format_reasoning_slide(reasoning_b)
    m_body_a = _format_metric_slide(mdata_a, metric_name, metric_type_hint)
    m_body_b = _format_metric_slide(mdata_b, metric_name, metric_type_hint)

    doc = BEAMER_PREAMBLE
    doc += "\n\\begin{document}\n"

    # ── Slide 1: Question ────────────────────────────────────────────────
    doc += f"\n\\begin{{frame}}{{Problem {pid_esc} [{ds_esc}]: Question}}\n"
    doc += f"    {q_body}\n"
    doc += "\\end{frame}\n"

    # ── Slide 2: Reasoning A ─────────────────────────────────────────────
    doc += f"\n\\begin{{frame}}[allowframebreaks]{{Problem {pid_esc}: Reasoning ({lbl_a})}}\n"
    doc += f"    {r_body_a}\n"
    doc += "\\end{frame}\n"

    # ── Slide 3: Metric A ────────────────────────────────────────────────
    doc += f"\n\\begin{{frame}}[allowframebreaks]{{Problem {pid_esc}: {m_esc} ({lbl_a})}}\n"
    doc += f"    {m_body_a}\n"
    doc += "\\end{frame}\n"

    # ── Slide 4: Reasoning B ─────────────────────────────────────────────
    doc += f"\n\\begin{{frame}}[allowframebreaks]{{Problem {pid_esc}: Reasoning ({lbl_b})}}\n"
    doc += f"    {r_body_b}\n"
    doc += "\\end{frame}\n"

    # ── Slide 5: Metric B ────────────────────────────────────────────────
    doc += f"\n\\begin{{frame}}[allowframebreaks]{{Problem {pid_esc}: {m_esc} ({lbl_b})}}\n"
    doc += f"    {m_body_b}\n"
    doc += "\\end{frame}\n"

    doc += "\n\\end{document}\n"
    return doc


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a LaTeX Beamer comparison presentation for one evaluated item.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset",       required=True, help="Dataset name (e.g. medqa, art)")
    p.add_argument("--problem_id",    required=True, help="Problem/sample ID from the JSON data")
    p.add_argument("--checkpoint_a",  required=True, type=int, help="First checkpoint number (0 = raw_model)")
    p.add_argument("--checkpoint_b",  required=True, type=int, help="Second checkpoint number")
    p.add_argument("--metric",        required=True, help="Metric name (e.g. backtracking, observation_coverage)")
    p.add_argument("--output",        default=os.path.join("results", "latex_slides"),
                   help="Output directory for .tex files  [default: results/latex_slides]")
    p.add_argument("--log_dir",       default=os.path.join("results", "llm_logs"),
                   help="Directory containing *_full_debug.jsonl files  [default: results/llm_logs]")
    p.add_argument("--checkpoints_dir", default="checkpoints",
                   help="Root checkpoints directory  [default: checkpoints]")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset     = args.dataset.lower()
    pid_raw     = args.problem_id
    ckpt_a      = args.checkpoint_a
    ckpt_b      = args.checkpoint_b
    metric_name = args.metric

    # Try to cast problem_id to int (most datasets use integer IDs)
    try:
        pid: Any = int(pid_raw)
    except ValueError:
        pid = pid_raw

    print(f"[generate_latex_slides] dataset={dataset}  pid={pid}  "
          f"ckpt_a={ckpt_a}  ckpt_b={ckpt_b}  metric={metric_name}")

    # ── 1. Attempt to load from JSONL ─────────────────────────────────────
    jsonl_records = _load_jsonl(args.log_dir, dataset)
    rec_a = _find_jsonl_record(jsonl_records, ckpt_a, pid)
    rec_b = _find_jsonl_record(jsonl_records, ckpt_b, pid)

    if jsonl_records and (rec_a is None or rec_b is None):
        # Try string pid fallback
        rec_a = rec_a or _find_jsonl_record(jsonl_records, ckpt_a, pid_raw)
        rec_b = rec_b or _find_jsonl_record(jsonl_records, ckpt_b, pid_raw)

    # ── 2. Load raw items from checkpoint files (for item metadata) ────────
    item_a = _load_raw_checkpoint_item(args.checkpoints_dir, dataset, ckpt_a, pid)
    item_b = _load_raw_checkpoint_item(args.checkpoints_dir, dataset, ckpt_b, pid)

    # Fallback: use JSONL record as the item source
    if item_a is None and rec_a:
        item_a = rec_a
    if item_b is None and rec_b:
        item_b = rec_b

    if item_a is None and item_b is None:
        print(
            f"[ERROR] Could not find problem_id={pid!r} in either checkpoint "
            f"(tried JSONL logs and raw checkpoint files).\n"
            f"  JSONL dir: {args.log_dir}\n"
            f"  Checkpoints dir: {args.checkpoints_dir}"
        )
        sys.exit(1)

    item_a = item_a or item_b
    item_b = item_b or item_a

    # ── 3. Extract reasoning ───────────────────────────────────────────────
    reasoning_a = (rec_a.get("reasoning", "") if rec_a else "") or _extract_reasoning(item_a)
    reasoning_b = (rec_b.get("reasoning", "") if rec_b else "") or _extract_reasoning(item_b)

    # ── 4. Extract metric data ─────────────────────────────────────────────
    def _get_mdata(rec: dict | None) -> dict | None:
        if rec is None:
            return None
        metrics = rec.get("metrics") or {}
        return metrics.get(metric_name) or metrics.get(metric_name.lower())

    mdata_a = _get_mdata(rec_a)
    mdata_b = _get_mdata(rec_b)

    if mdata_a is None and mdata_b is None:
        if jsonl_records:
            print(
                f"[WARN] Metric '{metric_name}' not found in the JSONL for pid={pid}. "
                f"Slides will be generated without metric data.\n"
                f"  Available metrics in record: "
                + str(list((rec_a or rec_b or {}).get("metrics", {}).keys()))
            )
        else:
            print(
                f"[WARN] JSONL log not found at '{args.log_dir}/{dataset}_full_debug.jsonl'. "
                "Run the evaluation pipeline first to populate metric results."
            )

    # Infer metric type from whichever mdata is available
    mtype_hint = ""
    if mdata_a:
        mtype_hint = mdata_a.get("type", "")
    elif mdata_b:
        mtype_hint = mdata_b.get("type", "")

    # ── 5. Build LaTeX ────────────────────────────────────────────────────
    latex_source = build_latex(
        dataset=dataset,
        pid=pid,
        ckpt_a=ckpt_a,
        ckpt_b=ckpt_b,
        item_a=item_a,
        item_b=item_b,
        reasoning_a=reasoning_a,
        reasoning_b=reasoning_b,
        metric_name=metric_name,
        mdata_a=mdata_a,
        mdata_b=mdata_b,
        metric_type_hint=mtype_hint,
    )

    # ── 6. Write to file ──────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    out_fname = f"{dataset}_pid{pid}_ckpt{ckpt_a}vs{ckpt_b}_{metric_name}.tex"
    out_path  = os.path.join(args.output, out_fname)

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(latex_source)

    print(f"[OK] LaTeX slides written -> {out_path}")
    print(f"     Compile with:  pdflatex \"{out_path}\"")


if __name__ == "__main__":
    main()
