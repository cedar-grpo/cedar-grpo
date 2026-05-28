#!/usr/bin/env python3
"""
Generate rationale-enhanced CauseLogics train/val JSON files using OpenRouter.

Usage:
  for linux/macOS:
  export OPENROUTER_API_KEY="your_openrouter_key"

  for windows cmd:
  set OPENROUTER_API_KEY=your_key_here

  python enhance_causelogics.py \
    --train "../../dataset/CauseLogics/causelogics_train.json" \
    --val "../../dataset/CauseLogics/causelogics_val.json" \
    --output-dir . \
    --threads 8 \
    --resume

Outputs:
  ./causelogics_train_enhanced.json
  ./causelogics_val_enhanced.json

Each output sample keeps the original fields and adds:
  "rationale": "...inner text from <think>...</think>..."

The script only accepts model outputs that exactly match:

<think>
...
</think>

<answer>
TRUE
</answer>

or

<think>
...
</think>

<answer>
FALSE
</answer>
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import copy
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =========================
# Config defaults
# =========================

MODEL = "google/gemini-3-flash-preview"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Can be overridden by CLI flags --resume / --no-resume.
RESUME_FROM_EXISTING_OUTPUTS = True

DEFAULT_THREADS = 8
DEFAULT_MAX_RETRIES = 6
DEFAULT_REQUEST_TIMEOUT_SECONDS = 90

TEMPERATURE = 0.2
TOP_P = 0.9
MAX_COMPLETION_TOKENS = 450

# Optional OpenRouter app-identifying headers.
# Leave as None unless you want OpenRouter analytics attribution.
HTTP_REFERER = None
X_TITLE = "causelogics-rationale-generation"


SYSTEM_PROMPT_SYNTHETIC_CAUSELOGICS = """\
You are generating supervised fine-tuning data for an abductive logical reasoning model.

You will receive:
1. A set of premises
2. A set of rules
3. An observed phenomenon
4. A possible cause
5. The gold answer

Write a concise rationale explaining why adding the possible cause does or does not make the phenomenon logically inferable.

The rationale should:
- Start from the phenomenon.
- Assume the possible cause is added to the premises.
- Identify the relevant rule or rule chain.
- State which required facts are already present or can be derived.
- Explain whether forward reasoning can infer the phenomenon.
- If the phenomenon cannot be inferred, briefly state where the proof path fails.
- Use only the given premises, rules, and possible cause.
- Avoid unrelated facts, unrelated rules, and external knowledge.

Output exactly:

<think>
[A concise abductive logical rationale, usually 3–6 sentences.]
</think>

<answer>
[The gold answer only: TRUE or FALSE.]
</answer>

Do not change the gold answer in the answer section.
Do not include any text, punctuation, or explanation in the answer section.
""".strip()


STRICT_RESPONSE_RE = re.compile(
    r"\A<think>\n(?P<rationale>.+?)\n</think>\n\n<answer>\n(?P<answer>TRUE|FALSE)\n</answer>\Z",
    re.DOTALL,
)


# =========================
# Prompt + validation
# =========================

def _get_any(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return default


def normalize_label(label_raw: Any) -> str:
    """
    Normalize CauseLogics labels to the exact target answer strings:
      TRUE or FALSE
    """
    if isinstance(label_raw, bool):
        return "TRUE" if label_raw else "FALSE"

    label_text = str(label_raw).strip().lower()
    if label_text == "true":
        return "TRUE"
    if label_text == "false":
        return "FALSE"

    raise ValueError(f"Invalid CauseLogics label: {label_raw!r}. Expected True/False.")


def create_synthetic_causelogics_prompt(example: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Returns:
      system_prompt, user_prompt, gold_answer_as_TRUE_or_FALSE
    """
    system_prompt = SYSTEM_PROMPT_SYNTHETIC_CAUSELOGICS

    premises_raw = _get_any(example, ["Premises", "premises"], default=[])
    rules_raw = _get_any(example, ["Rules", "rules"], default=[])
    phenomenon = _get_any(example, ["Phenomenon", "phenomenon"], default=None)
    possible_cause = _get_any(example, ["PossibleCause", "possible_cause"], default=None)
    label_raw = _get_any(example, ["Label", "label"], default=None)

    if isinstance(premises_raw, list):
        premises_text = "\n".join([f"- {x}" for x in premises_raw])
    else:
        premises_text = f"- {premises_raw}" if premises_raw is not None else ""

    if isinstance(rules_raw, list):
        rules_text = "\n".join([f"- {x}" for x in rules_raw])
    else:
        rules_text = f"- {rules_raw}" if rules_raw is not None else ""

    if phenomenon is None or possible_cause is None or label_raw is None:
        missing = []
        if phenomenon is None:
            missing.append("Phenomenon")
        if possible_cause is None:
            missing.append("PossibleCause")
        if label_raw is None:
            missing.append("Label/label")
        raise KeyError(f"CauseLogics example missing required field(s): {', '.join(missing)}")

    gold_answer = normalize_label(label_raw)

    user_prompt = f"""\
Premises:
{premises_text}

Rules:
{rules_text}

Phenomenon:
{str(phenomenon)}

Possible Cause:
{str(possible_cause)}

Gold Answer:
{gold_answer}

Generate the SFT target.
""".strip()

    return system_prompt, user_prompt, gold_answer


def validate_and_extract_rationale(response_text: str, expected_answer: str) -> str:
    """
    Strictly validates the model response.

    No fuzzy parsing.
    No extraction from malformed text.
    The whole stripped response must exactly match the requested format.
    """
    normalized = response_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    match = STRICT_RESPONSE_RE.fullmatch(normalized)
    if not match:
        raise ValueError(f"Bad format. Raw response was:\n{response_text}")

    answer = match.group("answer")
    if answer != expected_answer:
        raise ValueError(
            f"Answer mismatch. Expected {expected_answer}, got {answer}. "
            f"Raw response was:\n{response_text}"
        )

    rationale = match.group("rationale").strip()
    if not rationale:
        raise ValueError("Empty rationale.")

    return rationale


# =========================
# JSON I/O
# =========================

REQUIRED_FIELD_ALIASES = {
    "Premises": ["Premises", "premises"],
    "Rules": ["Rules", "rules"],
    "Phenomenon": ["Phenomenon", "phenomenon"],
    "PossibleCause": ["PossibleCause", "possible_cause"],
    "Label": ["Label", "label"],
}


def get_required_field(example: Dict[str, Any], canonical_name: str) -> Any:
    return _get_any(example, REQUIRED_FIELD_ALIASES[canonical_name], default=None)


def validate_causelogics_example(example: Dict[str, Any], path: Path, index: int) -> None:
    missing = [
        canonical
        for canonical in REQUIRED_FIELD_ALIASES
        if get_required_field(example, canonical) is None
    ]
    if missing:
        raise ValueError(f"{path} item {index} is missing fields: {missing}")

    premises = get_required_field(example, "Premises")
    rules = get_required_field(example, "Rules")
    label = get_required_field(example, "Label")

    if not isinstance(premises, list):
        raise ValueError(f"{path} item {index} has non-list Premises: {type(premises).__name__}")
    if not isinstance(rules, list):
        raise ValueError(f"{path} item {index} has non-list Rules: {type(rules).__name__}")

    normalize_label(label)


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path} item {i} is not a JSON object.")
        validate_causelogics_example(item, path, i)

    return data


def examples_match_for_resume(input_example: Dict[str, Any], output_example: Dict[str, Any]) -> bool:
    """
    Compare only the input-defining CauseLogics fields.
    This works even when samples do not have an id field.
    """
    for canonical in REQUIRED_FIELD_ALIASES:
        if get_required_field(input_example, canonical) != get_required_field(output_example, canonical):
            return False
    return True


def atomic_write_json(path: Path, data: List[Dict[str, Any]]) -> None:
    """
    Writes a complete valid JSON file atomically.

    This avoids leaving a half-written JSON file if the process is interrupted
    during the write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def output_path_for(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}_enhanced.json"


def initialize_output_data(
    input_data: List[Dict[str, Any]],
    output_path: Path,
    resume: bool,
) -> List[Dict[str, Any]]:
    """
    If resume is enabled and an output file already exists, load and verify it.
    Otherwise, initialize from the input data.

    Supports both:
    - the normal full-length output file used by this script, and
    - a shorter prefix output file, in case you previously saved only completed rows.
    """
    if resume and output_path.exists():
        existing = load_json_list(output_path)

        if len(existing) > len(input_data):
            raise ValueError(
                f"Cannot resume from {output_path}: output has more rows than input. "
                f"Input has {len(input_data)} rows, output has {len(existing)} rows."
            )

        for i, out in enumerate(existing):
            inp = input_data[i]
            if not examples_match_for_resume(inp, out):
                raise ValueError(
                    f"Cannot resume from {output_path}: row mismatch at index {i}. "
                    "The existing output does not match the current input data."
                )

        if len(existing) == len(input_data):
            return existing

        initialized = copy.deepcopy(input_data)
        for i, out in enumerate(existing):
            initialized[i] = out
        atomic_write_json(output_path, initialized)
        return initialized

    initialized = copy.deepcopy(input_data)
    atomic_write_json(output_path, initialized)
    return initialized


def has_completed_rationale(example: Dict[str, Any]) -> bool:
    return isinstance(example.get("rationale"), str) and bool(example["rationale"].strip())


def append_error_log(output_dir: Path, record: Dict[str, Any]) -> None:
    log_path = output_dir / "generation_errors.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def example_identifier(example: Dict[str, Any], index: int) -> str:
    """
    CauseLogics samples may not have a stable id.
    Use id if present; otherwise use the row index plus the phenomenon/cause.
    """
    if "id" in example:
        return str(example["id"])
    if "ID" in example:
        return str(example["ID"])

    phenomenon = _get_any(example, ["Phenomenon", "phenomenon"], default="")
    possible_cause = _get_any(example, ["PossibleCause", "possible_cause"], default="")
    return f"row-{index} | phenomenon={phenomenon!r} | possible_cause={possible_cause!r}"


# =========================
# OpenRouter API
# =========================

def post_json_with_urllib(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout_seconds: int,
) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
            return json.loads(response_body)

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {error_body}") from e

    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error: {e}") from e


def call_openrouter_once(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: int,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if HTTP_REFERER:
        headers["HTTP-Referer"] = HTTP_REFERER
    if X_TITLE:
        headers["X-Title"] = X_TITLE

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "stream": False,
    }

    response_json = post_json_with_urllib(
        url=OPENROUTER_URL,
        payload=payload,
        headers=headers,
        timeout_seconds=timeout_seconds,
    )

    try:
        content = response_json["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Unexpected OpenRouter response shape: {response_json}") from e

    if not isinstance(content, str):
        raise RuntimeError(f"Expected string content, got: {type(content).__name__}")

    return content


def generate_rationale_with_retries(
    api_key: str,
    example: Dict[str, Any],
    max_retries: int,
    timeout_seconds: int,
) -> Tuple[str, str]:
    """
    Returns:
      rationale, raw_valid_response
    """
    system_prompt, user_prompt, expected_answer = create_synthetic_causelogics_prompt(example)

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            raw_response = call_openrouter_once(
                api_key=api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout_seconds=timeout_seconds,
            )
            rationale = validate_and_extract_rationale(raw_response, expected_answer)
            return rationale, raw_response

        except Exception as e:
            last_error = e

            if attempt == max_retries:
                break

            sleep_seconds = min(60.0, 1.5 * (2 ** (attempt - 1)))
            sleep_seconds += random.uniform(0.0, 0.75)
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")


# =========================
# Worker orchestration
# =========================

@dataclass(frozen=True)
class Task:
    split_name: str
    index: int
    example: Dict[str, Any]


@dataclass
class SplitState:
    input_path: Path
    output_path: Path
    data: List[Dict[str, Any]]
    lock: threading.Lock


def process_task(
    task: Task,
    api_key: str,
    max_retries: int,
    timeout_seconds: int,
) -> Tuple[Task, str, str]:
    rationale, raw_response = generate_rationale_with_retries(
        api_key=api_key,
        example=task.example,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
    )
    return task, rationale, raw_response


def build_pending_tasks(split_name: str, data: List[Dict[str, Any]]) -> List[Task]:
    tasks: List[Task] = []
    for i, example in enumerate(data):
        if not has_completed_rationale(example):
            tasks.append(Task(split_name=split_name, index=i, example=copy.deepcopy(example)))
    return tasks


def run_generation(
    train_path: Path,
    val_path: Path,
    output_dir: Path,
    threads: int,
    resume: bool,
    max_retries: int,
    timeout_seconds: int,
) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY environment variable.")

    output_dir.mkdir(parents=True, exist_ok=True)

    train_input = load_json_list(train_path)
    val_input = load_json_list(val_path)

    train_output_path = output_path_for(train_path, output_dir)
    val_output_path = output_path_for(val_path, output_dir)

    train_data = initialize_output_data(train_input, train_output_path, resume=resume)
    val_data = initialize_output_data(val_input, val_output_path, resume=resume)

    states: Dict[str, SplitState] = {
        "train": SplitState(
            input_path=train_path,
            output_path=train_output_path,
            data=train_data,
            lock=threading.Lock(),
        ),
        "val": SplitState(
            input_path=val_path,
            output_path=val_output_path,
            data=val_data,
            lock=threading.Lock(),
        ),
    }

    tasks = build_pending_tasks("train", train_data) + build_pending_tasks("val", val_data)

    total_rows = len(train_data) + len(val_data)
    already_done = total_rows - len(tasks)

    print(f"Train rows: {len(train_data)}")
    print(f"Val rows:   {len(val_data)}")
    print(f"Already completed: {already_done}/{total_rows}")
    print(f"Pending: {len(tasks)}")
    print(f"Output train: {train_output_path}")
    print(f"Output val:   {val_output_path}")
    print(f"Threads: {threads}")
    print("", flush=True)

    if not tasks:
        print("Nothing to do. All rows already have rationales.")
        return

    completed_now = 0
    failed: List[Dict[str, Any]] = []
    progress_lock = threading.Lock()
    error_log_lock = threading.Lock()

    with futures.ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_task = {
            executor.submit(
                process_task,
                task,
                api_key,
                max_retries,
                timeout_seconds,
            ): task
            for task in tasks
        }

        for future in futures.as_completed(future_to_task):
            task = future_to_task[future]
            state = states[task.split_name]

            try:
                finished_task, rationale, raw_response = future.result()

                with state.lock:
                    row = copy.deepcopy(state.data[finished_task.index])
                    row["rationale"] = rationale

                    # Keep this disabled by default to avoid storing extra SFT fields.
                    # Uncomment if you want to audit the exact accepted response later.
                    # row["rationale_raw_response"] = raw_response

                    state.data[finished_task.index] = row

                    # Save after every single successful rationale.
                    atomic_write_json(state.output_path, state.data)

                with progress_lock:
                    completed_now += 1
                    overall_done = already_done + completed_now
                    print(
                        f"[OK] {task.split_name}[{task.index}] "
                        f"id={example_identifier(task.example, task.index)} "
                        f"progress={overall_done}/{total_rows}",
                        flush=True,
                    )

            except Exception as e:
                error_record = {
                    "split": task.split_name,
                    "index": task.index,
                    "id": example_identifier(task.example, task.index),
                    "error": str(e),
                }
                failed.append(error_record)

                with error_log_lock:
                    append_error_log(output_dir, error_record)

                with progress_lock:
                    print(
                        f"[FAILED] {task.split_name}[{task.index}] "
                        f"id={example_identifier(task.example, task.index)} error={e}",
                        file=sys.stderr,
                        flush=True,
                    )

    print("", flush=True)
    print(f"Completed in this run: {completed_now}")
    print(f"Failed in this run:    {len(failed)}")

    if failed:
        print(f"Failure log: {output_dir / 'generation_errors.jsonl'}")
        print(
            "Rows that failed were left without a rationale, so rerunning with "
            "--resume will retry only those unfinished rows."
        )


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate rationale-enhanced CauseLogics train/val JSON files."
    )

    parser.add_argument("--train", required=True, type=Path, help="Path to causelogics_train.json")
    parser.add_argument("--val", required=True, type=Path, help="Path to causelogics_val.json")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for enhanced JSON files")

    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=f"Number of simultaneous API calls. Default: {DEFAULT_THREADS}",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Max attempts per example. Default: {DEFAULT_MAX_RETRIES}",
    )

    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help=f"HTTP request timeout. Default: {DEFAULT_REQUEST_TIMEOUT_SECONDS}",
    )

    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        dest="resume",
        help="Resume from existing enhanced output files.",
    )
    resume_group.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Ignore existing enhanced output files and start fresh.",
    )
    parser.set_defaults(resume=RESUME_FROM_EXISTING_OUTPUTS)

    args = parser.parse_args()

    if args.threads < 1:
        parser.error("--threads must be >= 1")
    if args.max_retries < 1:
        parser.error("--max-retries must be >= 1")
    if args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be >= 1")

    return args


def main() -> None:
    args = parse_args()

    run_generation(
        train_path=args.train,
        val_path=args.val,
        output_dir=args.output_dir,
        threads=args.threads,
        resume=args.resume,
        max_retries=args.max_retries,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    main()